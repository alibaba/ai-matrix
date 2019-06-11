#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
#include "logger.h"
#include "argsParser.h"

const std::string gSampleName = "TensorRT.sample_fasterRCNN";

static samplesCommon::Args gArgs;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// Stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 375;
static const int INPUT_W = 500;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;
static const int NMS_MAX_OUT = 300; // This value needs to be changed as per the nmsMaxOut value set in RPROI plugin parameters in prototxt

const std::string CLASSES[OUTPUT_CLS_SIZE]{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";

struct PPM
{
    std::string magic, fileName;
    int h, w, max;
    uint8_t buffer[INPUT_C * INPUT_H * INPUT_W];
};

struct BBox
{
    float x1, y1, x2, y2;
};

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"images/", "data/faster-rcnn/"};
    return locateFile(input, dirs);
}

// Simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(locateFile(filename), std::ifstream::binary);
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void writePPMFileWithBBox(const std::string& filename, PPM& ppm, const BBox& bbox)
{
    std::ofstream outfile("./output/" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6"
            << "\n"
            << ppm.w << " " << ppm.h << "\n"
            << ppm.max << "\n";
    auto round = [](float x) -> int { return int(std::floor(x + 0.5f)); };
    for (int x = int(bbox.x1); x < int(bbox.x2); ++x)
    {
        // Bbox top border
        ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3] = 255;
        ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 2] = 0;
        // Bbox bottom border
        ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3] = 255;
        ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 2] = 0;
    }
    for (int y = int(bbox.y1); y < int(bbox.y2); ++y)
    {
        // Bbox left border
        ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3] = 255;
        ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 2] = 0;
        // Bbox right border
        ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3] = 255;
        ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 2] = 0;
    }
    outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void caffeToTRTModel(const std::string& deployFile,           // Name for caffe prototxt
                     const std::string& modelFile,            // Name for model
                     const std::vector<std::string>& outputs, // Network outputs
                     unsigned int maxBatchSize,               // Batch size - NB must be at least as large as the batch we want to run with)
                     IHostMemory** trtModelStream)            // Output stream for the TensorRT model
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();

    gLogInfo << "Begin parsing model..." << std::endl;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
                                                              locateFile(modelFile).c_str(),
                                                              *network,
                                                              DataType::kFLOAT);
    gLogInfo << "End parsing model..." << std::endl;
    // Specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(10 << 20); // We need about 6MB of scratch space for the plugin layer for batch size 5

    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    gLogInfo << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);
    gLogInfo << "End building engine..." << std::endl;

    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* inputData, float* inputImInfo, std::vector<float>& outputBboxPred, std::vector<float>& outputClsProb, std::vector<float>& outputRois, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 2 inputs and 3 outputs.
    assert(engine.getNbBindings() == 5);
    void* buffers[5];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
        inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1),
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
        outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
    const int dataSize = batchSize * INPUT_C * INPUT_H * INPUT_W;
    const int imInfoSize = batchSize * IM_INFO_SIZE;
    const int bboxPredSize = batchSize * NMS_MAX_OUT * OUTPUT_BBOX_SIZE;
    const int clsProbSize = batchSize * NMS_MAX_OUT * OUTPUT_CLS_SIZE;
    const int roisSize = batchSize * NMS_MAX_OUT * 4;
    // Create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex0], dataSize * sizeof(float)));      // data
    CHECK(cudaMalloc(&buffers[inputIndex1], imInfoSize * sizeof(float)));    // im_info
    CHECK(cudaMalloc(&buffers[outputIndex0], bboxPredSize * sizeof(float))); // bbox_pred
    CHECK(cudaMalloc(&buffers[outputIndex1], clsProbSize * sizeof(float)));  // cls_prob
    CHECK(cudaMalloc(&buffers[outputIndex2], roisSize * sizeof(float)));     // rois

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, dataSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, imInfoSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputBboxPred.data(), buffers[outputIndex0], bboxPredSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(outputClsProb.data(), buffers[outputIndex1], clsProbSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(outputRois.data(), buffers[outputIndex2], roisSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex0]));
    CHECK(cudaFree(buffers[inputIndex1]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
}

void bboxTransformInvAndClip(std::vector<float>& rois, std::vector<float>& deltas, std::vector<float>& predBBoxes, float* imInfo,
                             const int N, const int nmsMaxOut, const int numCls)
{
    for (int i = 0; i < N * nmsMaxOut; ++i)
    {
        float width = rois[i * 4 + 2] - rois[i * 4] + 1;
        float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
        float ctr_x = rois[i * 4] + 0.5f * width;
        float ctr_y = rois[i * 4 + 1] + 0.5f * height;
        float* imInfo_offset = imInfo + i / nmsMaxOut * 3;
        for (int j = 0; j < numCls; ++j)
        {
            float dx = deltas[i * numCls * 4 + j * 4];
            float dy = deltas[i * numCls * 4 + j * 4 + 1];
            float dw = deltas[i * numCls * 4 + j * 4 + 2];
            float dh = deltas[i * numCls * 4 + j * 4 + 3];
            float pred_ctr_x = dx * width + ctr_x;
            float pred_ctr_y = dy * height + ctr_y;
            float pred_w = exp(dw) * width;
            float pred_h = exp(dh) * height;
            predBBoxes[i * numCls * 4 + j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
        }
    }
}

std::vector<int> nms(std::vector<std::pair<float, int>>& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto i : score_index)
    {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(&bbox[(idx * numClasses + classNum) * 4],
                                           &bbox[(kept_idx * numClasses + classNum) * 4]);
                keep = overlap <= nms_threshold;
            }
            else
                break;
        }
        if (keep)
            indices.push_back(idx);
    }
    return indices;
}

void printHelp(const char* name)
{
    std::cout << "Usage: " << name << "\n"
        << "Optional Parameters:\n"
        << "  -h, --help        Display help information.\n"
        << "  --useDLACore=N    Specify the DLA engine to run on.\n"
        << "  --batch_size=N    Specify the batch size.\n";
}


int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help || !argsOK)
    {
        printHelp(argv[0]);
        return argsOK ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    IHostMemory* trtModelStream{nullptr};
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Batch size
    const int N = gArgs.batch_size;
    // Create a TensorRT model from the caffe model and serialize it to a stream
    caffeToTRTModel("faster_rcnn_test_iplugin.prototxt",
                    "VGG16_faster_rcnn_final.caffemodel",
                    std::vector<std::string>{OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2},
                    N, &trtModelStream);
    assert(trtModelStream != nullptr);

    // Available images
    std::string image;
    std::ifstream infile("images/list.txt");
    // std::vector<std::string> imageList = {"000456.ppm", "000542.ppm", "001150.ppm", "001763.ppm", "004545.ppm"};
    std::vector<std::string> imageList(N);
    std::vector<std::string> images;
    while (getline(infile, image)) {
        images.push_back(image);
    }
    infile.close();
    
    std::vector<PPM> ppms(N);

    float imInfo[N * 3]; // Input im_info
    assert(ppms.size() <= imageList.size());

    float* data = new float[N * INPUT_C * INPUT_H * INPUT_W];
    // Pixel mean used by the Faster R-CNN's author
    float pixelMean[3]{102.9801f, 115.9465f, 122.7717f}; // Also in BGR order

    // Deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    std::vector<float> rois;
    std::vector<float> bboxPreds;
    std::vector<float> clsProbs;
    std::vector<float> predBBoxes;

    // Host memory for outputs
    rois.assign(N * NMS_MAX_OUT * 4, 0);
    bboxPreds.assign(N * NMS_MAX_OUT * OUTPUT_BBOX_SIZE, 0);
    clsProbs.assign(N * NMS_MAX_OUT * OUTPUT_BBOX_SIZE, 0);

    // Predicted bounding boxes
    predBBoxes.assign(N * NMS_MAX_OUT * OUTPUT_BBOX_SIZE, 0);

    // The sample passes if there is at least one detection for each item in the batch
    bool pass = true;

    const int num_batches = images.size() / N;

    for (int iter = 0; iter < num_batches; iter++)
    {
        imageList.clear();

        for (int i = 0; i < N; ++i)
        {
            imageList.push_back(images[iter*N+i]);
        }

        for (int i = 0; i < N; ++i)
        {
            readPPMFile(imageList[i], ppms[i]);
            imInfo[i * 3] = float(ppms[i].h);     // Number of rows
            imInfo[i * 3 + 1] = float(ppms[i].w); // Number of columns
            imInfo[i * 3 + 2] = 1;                // Image scale
        }

        for (int i = 0, volImg = INPUT_C * INPUT_H * INPUT_W; i < N; ++i)
        {
            for (int c = 0; c < INPUT_C; ++c)
            {
                // The color image to input should be in BGR order
                for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
                    data[i * volImg + c * volChl + j] = float(ppms[i].buffer[j * INPUT_C + 2 - c]) - pixelMean[c];
            }
        }

        // Run inference
        doInference(*context, data, imInfo, bboxPreds, clsProbs, rois, N);

        // Unscale back to raw image space
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < NMS_MAX_OUT * 4 && imInfo[i * 3 + 2] != 1; ++j)
                rois[i * NMS_MAX_OUT * 4 + j] /= imInfo[i * 3 + 2];
        }

        bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, NMS_MAX_OUT, OUTPUT_CLS_SIZE);

        const float nms_threshold = 0.3f;
        const float score_threshold = 0.8f;

        for (int i = 0; i < N; ++i)
        {
            float* bbox = predBBoxes.data() + i * NMS_MAX_OUT * OUTPUT_BBOX_SIZE;
            float* scores = clsProbs.data() + i * NMS_MAX_OUT * OUTPUT_CLS_SIZE;
            int numDetections = 0;
            for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // Skip the background
            {
                std::vector<std::pair<float, int>> score_index;
                for (int r = 0; r < NMS_MAX_OUT; ++r)
                {
                    if (scores[r * OUTPUT_CLS_SIZE + c] > score_threshold)
                    {
                        score_index.push_back(std::make_pair(scores[r * OUTPUT_CLS_SIZE + c], r));
                        std::stable_sort(score_index.begin(), score_index.end(),
                                         [](const std::pair<float, int>& pair1,
                                            const std::pair<float, int>& pair2) {
                                             return pair1.first > pair2.first;
                                         });
                    }
                }

                // Apply NMS algorithm
                std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);

                numDetections += static_cast<int>(indices.size());

                // Show results
                for (unsigned k = 0; k < indices.size(); ++k)
                {
                    int idx = indices[k];
                    std::string storeName = CLASSES[c] + "-" + std::to_string(scores[idx * OUTPUT_CLS_SIZE + c]) + ".ppm";
                    // gLogInfo << "Detected " << CLASSES[c] << " in " << ppms[i].fileName << " with confidence " << scores[idx * OUTPUT_CLS_SIZE + c] * 100.0f << "% "
                             // << " (Result stored in " << storeName << ")." << std::endl;

                    BBox b{bbox[idx * OUTPUT_BBOX_SIZE + c * 4], bbox[idx * OUTPUT_BBOX_SIZE + c * 4 + 1], bbox[idx * OUTPUT_BBOX_SIZE + c * 4 + 2], bbox[idx * OUTPUT_BBOX_SIZE + c * 4 + 3]};
                    writePPMFileWithBBox(storeName, ppms[i], b);
                }
            }
            pass &= numDetections >= 1;
        }

        if (iter % 100 == 0)
        {
            gLogInfo << "Number of batches processed: " << iter << std::endl;
        }
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    delete[] data;

    return gLogger.reportTest(sampleTest, pass);
}
