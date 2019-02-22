#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <vector>
#include <map>
#include <random>
#include <iterator>
#include <algorithm>

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace nvonnxparser;
#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}

struct Params
{
    std::string deployFile, modelFile, engine, calibrationCache{"CalibrationTable"};
    std::string uffFile;
    std::string onnxModelFile;
    std::string testList;
    std::string data_folder;
    bool scale{ false };
    std::vector<std::string> outputs;
    std::vector<std::pair<std::string, Dims3> > uffInputs;
    int device{ 0 }, batchSize{ 1 }, workspaceSize{ 23 }, iterations{ 1 }, avgRuns{ 1 };
    bool fp16{ false }, int8{ false }, verbose{ false }, hostTime{ false };
    float pct{99};
    //float pct{0};
} gParams;

static inline int volume(Dims3 dims)
{
    return dims.d[0]*dims.d[1]*dims.d[2];
}

std::vector<std::string> gInputs;
std::map<std::string, Dims3> gInputDimensions;

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage/100) * all);
    if (0 <= exclude && exclude < all)
    {
        std::sort(times.begin(), times.end());
        return times[all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

// Logger for TensorRT info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO || gParams.verbose)
            std::cout << msg << std::endl;
    }
} gLogger;

class RndInt8Calibrator : public IInt8EntropyCalibrator
{
public:
    RndInt8Calibrator(int totalSamples, std::string cacheFile)
        : mTotalSamples(totalSamples)
        , mCurrentSample(0)
        , mCacheFile(cacheFile)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);//it will make the random data to calibrator model
        for(auto& elem: gInputDimensions)
        {
            int elemCount = volume(elem.second);
	    printf("INT8 Calibrator Here..\n");
            std::vector<float> rnd_data(elemCount);
            for(auto& val: rnd_data)
                val = distribution(generator);

            void * data;
            CHECK(cudaMalloc(&data, elemCount * sizeof(float)));
            CHECK(cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float), cudaMemcpyHostToDevice));

            mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
        }
    }

    ~RndInt8Calibrator()
    {
        for(auto& elem: mInputDeviceBuffers)
            CHECK(cudaFree(elem.second));
    }

    int getBatchSize() const override
    {
        return 1;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (mCurrentSample >= mTotalSamples)
            return false;

        for(int i = 0; i < nbBindings; ++i)
            bindings[i] = mInputDeviceBuffers[names[i]];

        ++mCurrentSample;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void* cache, size_t length) override
    {
    }

private:
    int mTotalSamples;
    int mCurrentSample;
    std::string mCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
};


ICudaEngine* caffeToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
                                                              gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(),
                                                              *network,
                                                              gParams.fp16 ? DataType::kHALF:DataType::kFLOAT);


    if (!blobNameToTensor)
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
        std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" <<
        dims.d[2] << std::endl;
    }

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (blobNameToTensor->find(s.c_str()) == nullptr)
        {
            std::cout << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
        std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
        << dims.d[2] << std::endl;
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(size_t(gParams.workspaceSize)<<20);
    if(gParams.fp16)
    {
    	builder->setFp16Mode(gParams.fp16);
	    printf("setFP16 to default Model...\n");
    }

    
    if (gParams.int8)
    {
        RndInt8Calibrator calibrator(1, gParams.calibrationCache); // change here WW
        builder->setInt8Mode(true);
	    printf("setINT8 to default Model...\n");
        builder->setInt8Calibrator(&calibrator);
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

ICudaEngine* uffToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    IUffParser* parser = createUffParser();

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            std::cerr << "Failed to register output " << s << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    for (auto& s : gParams.uffInputs)
    {
        if (!parser->registerInput(s.first.c_str(), s.second, UffInputOrder::kNCHW))
        {
            std::cerr << "Failed to register input " << s.first << std::endl;
            return nullptr;
        }
    }

    if (!parser->parse(gParams.uffFile.c_str(), *network, gParams.fp16 ? DataType::kHALF:DataType::kFLOAT))
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(gParams.workspaceSize<<20);
    builder->setFp16Mode(gParams.fp16);

    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    if (gParams.int8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(&calibrator);
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}
/* outdated in new tensorRT docker 19.01
ICudaEngine* onnxToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // create onnx config file
    nvonnxparser::IOnnxConfig* config = nvonnxparser::createONNXConfig();
    config->setModelFileName(gParams.onnxModelFile.c_str());

    // parse the onnx model to populate the network, then set the outputs
    nvonnxparser::IONNXParser* parser = nvonnxparser::createONNXParser(*config);

    if (!parser->parse(gParams.onnxModelFile.c_str(), DataType::kFLOAT))
    {
        std::cout << "failed to parse onnx file" << std::endl;
        return nullptr;
    }

    // Retrieve the network definition from parser
    if (!parser->convertToTRTNetwork())
    {
        std::cout << "failed to convert onnx network into TRT network" << std::endl;
        return nullptr;
    }

    nvinfer1::INetworkDefinition* network = parser->getTRTNetwork();

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        gParams.outputs.push_back(network->getOutput(i)->getName());
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(gParams.workspaceSize<<20);
    builder->setFp16Mode(gParams.fp16);

    ICudaEngine* engine = builder->buildCudaEngine(*network);

    if (engine == nullptr)
    {
        std::cout << "could not build engine" << std::endl;
        assert(false);
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}*/
void createMemoryFromImage(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name,float* imgFloatData)
{
    size_t bindingIndex = engine.getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
    assert(bindingIndex < buffers.size());
    Dims3 dimensions = static_cast<Dims3&&>(engine.getBindingDimensions((int)bindingIndex));
    size_t eltCount = dimensions.d[0]*dimensions.d[1]*dimensions.d[2]*gParams.batchSize, memSize = eltCount * sizeof(float);
    float* localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++)
        {	

		//localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;
		localMem[i] = imgFloatData[i];
		//printf("ImageData %f\n",localMem[i]);
	}

    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

    delete[] localMem;
    buffers[bindingIndex] = deviceMem;
    printf("Memory Read...Image Ok\n");
}
/*
void createMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
    size_t bindingIndex = engine.getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
    assert(bindingIndex < buffers.size());
    Dims3 dimensions = static_cast<Dims3&&>(engine.getBindingDimensions((int)bindingIndex));
    size_t eltCount = dimensions.d[0]*dimensions.d[1]*dimensions.d[2]*gParams.batchSize, memSize = eltCount * sizeof(float);

    float* localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++)
        localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

    delete[] localMem;
    buffers[bindingIndex] = deviceMem;
    //printf("Memory Read...Random Ok\n");
}
*/
void createMemorySetZero(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
    size_t bindingIndex = engine.getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
    assert(bindingIndex < buffers.size());
    Dims3 dimensions = static_cast<Dims3&&>(engine.getBindingDimensions((int)bindingIndex));
    size_t eltCount = dimensions.d[0]*dimensions.d[1]*dimensions.d[2]*gParams.batchSize, memSize = eltCount * sizeof(float);

    float* localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++)
        localMem[i] = 0;

    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

    delete[] localMem;
    buffers[bindingIndex] = deviceMem;
    printf("Memory Set Zero... Ok\n");
}

void getMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name, std::vector<int> & res_vector)
{
    size_t bindingIndex = engine.getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
    assert(bindingIndex < buffers.size());
    Dims3 dimensions = static_cast<Dims3&&>(engine.getBindingDimensions((int)bindingIndex));
    size_t eltCount = dimensions.d[0]*dimensions.d[1]*dimensions.d[2]*gParams.batchSize, memSize = eltCount * sizeof(float);
    float* localMem = new float[eltCount];

    CHECK(cudaMemcpy(localMem, buffers[bindingIndex], memSize, cudaMemcpyDeviceToHost));

    std::vector <std::pair <float, int>> res;
    
    for (int i=0; i< eltCount; i++){
        res.push_back(std::make_pair(localMem[i], i));
    }
    std::sort (res.begin(), res.end());
    int rank = 0;
    for (int i=eltCount-1; i >= eltCount-5; i--){
        rank ++;
        printf ("Top %d, softmax=%f, class=%d\n", rank, res[i].first, res[i].second);
        res_vector.push_back(res[i].second);
    }


/*
    for(int x = 0; x < eltCount ; x++)
	if(!(gParams.fp16 || gParams.int8))
         printf("Result [%d] is : %.23f\n",x, localMem[x] );
	else if(gParams.fp16)
         printf("Result [%d] is : %.10f\n",x, localMem[x] );
	else if(gParams.int8)
         printf("Result [%d] is : %.8f\n",x, localMem[x] );
*/
    delete[] localMem;
    for(int i=0; i< buffers.size(); i++){
        CHECK(cudaFree(buffers[i]));
    }
     
}
void doInference(ICudaEngine& engine, float* imgFloatData, std::vector<int> &res_vector)
{
    IExecutionContext *context = engine.createExecutionContext();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
	
    //Create the input buffer for H2D 
    std::vector<void*> buffers(gInputs.size() + gParams.outputs.size());
    for (size_t i = 0; i < gInputs.size(); i++)
        createMemoryFromImage(engine, buffers, gInputs[i],imgFloatData);

    //Create the output buffer for H2D
    for (size_t i = 0; i < gParams.outputs.size(); i++)
        createMemorySetZero(engine, buffers, gParams.outputs[i]);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    std::vector<float> times(gParams.avgRuns);
    for (int j = 0; j < gParams.iterations; j++)
    {
        float total = 0, ms;
        for (int i = 0; i < gParams.avgRuns; i++)
        {
            if (gParams.hostTime)
            {
                auto tStart = std::chrono::high_resolution_clock::now();
                context->execute(gParams.batchSize, &buffers[0]);
                auto tEnd = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            }
            else
            {
                cudaEventRecord(start, stream);
                //context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
                context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
                cudaEventRecord(end, stream);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&ms, start, end);
            }
            times[i] = ms;
            total += ms;
        }
        total /= gParams.avgRuns;
        std::cout << "Average over " << gParams.avgRuns << " runs is " << total << " ms (percentile time is " << percentile(gParams.pct, times) << ")." << std::endl;

	
    }
    cudaDeviceSynchronize();
    /*CH-------------*/
    //for (size_t i = 0; i < gParams.outputs.size(); i++)
    
	getMemory(engine, buffers, gParams.outputs[0], res_vector);

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();
     
}

static void printUsage()
{
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --deploy=<file>      Caffe deploy file\n");
    printf("  OR --uff=<file>      UFF file\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

    printf("\nMandatory params for onnx:\n");
    printf("  --onnx=<file>        ONNX Model file\n");

    printf("\nOptional params:\n");

    printf("  --uffInput=<name>,C,H,W Input blob names along with their dimensions for UFF parser\n");
    printf("  --model=<file>       Caffe model file (default = no model, random weights used)\n");
    printf("  --batch=N            Set batch size (default = %d)\n", gParams.batchSize);
    printf("  --device=N           Set cuda device to N (default = %d)\n", gParams.device);
    printf("  --iterations=N       Run N iterations (default = %d)\n", gParams.iterations);
    printf("  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
    printf("  --percentile=P       For each iteration, report the percentile time at P percentage (0<P<=100, default = %.1f%%)\n", gParams.pct);
    printf("  --workspace=N        Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
    printf("  --fp16               Run in fp16 mode (default = false). Permits 16-bit kernels\n");
    printf("  --int8               Run in int8 mode (default = false). Currently no support for ONNX model.\n");
    printf("  --verbose            Use verbose logging (default = false)\n");
    printf("  --hostTime           Measure host time rather than GPU time (default = false)\n");
    printf("  --engine=<file>      Generate a serialized TensorRT engine\n");
    printf("  --calib=<file>       Read INT8 calibration cache file.  Currently no support for ONNX model.\n");
    printf("  --test=<file>        Read the test images from list.\n");
    printf("  --label=<file>       Read the labels for image class.\n");

    fflush(stdout);
}
void MemSet(float* SrcMem, int Size)
{
	for( int x =0 ; x < Size; x++)
		SrcMem[x]=0.0f;
}
bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseBool(const char* arg, const char* name, bool& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
    if (match)
    {
        std::cout << name << std::endl;
        value = true;
    }
    return match;

}

bool parseFloat(const char* arg, const char* name, float& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}


bool parseArgs(int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage();
        return false;
    }

    for (int j = 1; j < argc; j++)
    {
        if (parseString(argv[j], "model", gParams.modelFile) || parseString(argv[j], "deploy", gParams.deployFile) || parseString(argv[j], "engine", gParams.engine))
            continue;

        if (parseString(argv[j], "uff", gParams.uffFile))
            continue;

        if (parseString(argv[j], "onnx", gParams.onnxModelFile))
            continue;

        if (parseString(argv[j], "calib", gParams.calibrationCache))
            continue;

        if (parseString(argv[j], "test", gParams.testList))
            continue;

        if (parseString(argv[j], "data_folder", gParams.data_folder))
            continue;
        
        std::string output;
        if (parseString(argv[j], "output", output))
        {
            gParams.outputs.push_back(output);
            continue;
        }

        std::string uffInput;
        if (parseString(argv[j], "uffInput", uffInput))
        {
            std::vector<std::string> uffInputStrs = split(uffInput, ',');
            if (uffInputStrs.size() != 4)
            {
                printf("Invalid uffInput: %s\n", uffInput.c_str());
                return false;
            }

            gParams.uffInputs.push_back(std::make_pair(uffInputStrs[0], Dims3(atoi(uffInputStrs[1].c_str()), atoi(uffInputStrs[2].c_str()), atoi(uffInputStrs[3].c_str()))));
            continue;
        }

        if (parseInt(argv[j], "batch", gParams.batchSize) || parseInt(argv[j], "iterations", gParams.iterations) || parseInt(argv[j], "avgRuns", gParams.avgRuns)
            || parseInt(argv[j], "device", gParams.device)  || parseInt(argv[j], "workspace", gParams.workspaceSize))
            continue;

        if (parseFloat(argv[j], "percentile", gParams.pct))
            continue;

        if (parseBool(argv[j], "fp16", gParams.fp16) || parseBool(argv[j], "int8", gParams.int8)
            || parseBool(argv[j], "verbose", gParams.verbose) || parseBool(argv[j], "hostTime", gParams.hostTime))
            continue;
        
        if (parseBool(argv[j], "scale", gParams.scale) )
            continue;

        printf("Unknown argument: %s\n", argv[j]);
        return false;
    }
    return true;
}

static ICudaEngine* createEngine()
{
    ICudaEngine *engine;
    if ((!gParams.deployFile.empty()) || (!gParams.uffFile.empty()) || (!gParams.onnxModelFile.empty())) {

        if (!gParams.uffFile.empty())
        {
            engine = uffToTRTModel();
        }
        else if (!gParams.onnxModelFile.empty())
        {
            //engine = onnxToTRTModel(); not needed in our test
        }
        else
        {
            engine = caffeToTRTModel();
        }

        if (!engine)
        {
            std::cerr << "Engine could not be created" << std::endl;
            return nullptr;
        }

        if (!gParams.engine.empty())
        {
            std::ofstream p(gParams.engine);
            if (!p)
            {
                std::cerr << "could not open plan output file" << std::endl;
                return nullptr;
            }
            IHostMemory *ptr = engine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
        }
        return engine;
    }

    // load directly from serialized engine file if deploy not specified
    if (!gParams.engine.empty()) {
        char *trtModelStream{nullptr};
        size_t size{0};
        std::ifstream file(gParams.engine, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

        IRuntime* infer = createInferRuntime(gLogger);
        engine = infer->deserializeCudaEngine(trtModelStream, size, nullptr);
        if (trtModelStream) delete [] trtModelStream;

        // assume input to be "data" for deserialized engine
        gInputs.push_back("data");
        return engine;
    }

    // complain about empty deploy file
    std::cerr << "Deploy file not specified" << std::endl;
    return nullptr;
}
void ImgPreprocessAspectVGG(cv::Mat SrcImage, float* imgRow, Dims3 dimensions, bool scale){
    // first crop the image according to their h-w-ratio
    int resize_h = dimensions.d[1];
    int resize_w = dimensions.d[2];
    assert(resize_h == resize_w);
    int w = SrcImage.cols;
    int h = SrcImage.rows;
    int off, startX, startY, height, width,resize_aspect_h, resize_aspect_w;
    // resize to 256xN or Nx256 whichever smallest side is 256
    if ( h < w){
        resize_aspect_h = 256;
        resize_aspect_w = 256.0/h * w;
    }else{
        resize_aspect_w = 256;
        resize_aspect_h = 256.0/w * h;
    }
    cv::Mat tmp, tmpAspect;
    cv::resize(SrcImage, tmpAspect, cv::Size(resize_aspect_w, resize_aspect_h));
    std::cout << "rh="<<resize_aspect_h<<",rw="<<resize_aspect_w;
    std::cout << "rh="<<h<<",rw="<<w;
    // crop center of 224x224
    startX = (resize_aspect_w - resize_w)/2;
    startY = (resize_aspect_h - resize_h)/2;
    std::cout << "rh="<<resize_h<<",rw="<<resize_w;
    std::cout << "rh="<<startX<<",rw="<<startY;
    cv::Mat ROI(tmpAspect, cvRect(startX,startY,resize_w,resize_h));
    ROI.copyTo(tmp);
    std::cout << "rh=====";
    cv::Mat imgFloat;
    tmp.convertTo(imgFloat, CV_32FC3);
        
    // minus mean and multiply by scale
    imgFloat = imgFloat - (cv::Scalar(103.94,116.78,123.68));
    if (scale){
        imgFloat = imgFloat.mul(cv::Scalar(0.017,0.017,0.017));
    }
        
    float a = imgFloat.at<cv::Vec3f>(0,0)[1];
    for(int c =0; c < dimensions.d[0];c++){
        for(int h =0;h<dimensions.d[1];h++){	 
            for(int w=0; w< dimensions.d[2];w++){
                float a = imgFloat.at<cv::Vec3f>(h,w)[c];
                imgRow[ c*dimensions.d[1]*dimensions.d[2]+ h*dimensions.d[2] + w ]= a;		
            }
        }
    }
}

void ImgPreprocessAspectCut(cv::Mat SrcImage, float* imgRow, Dims3 dimensions, bool scale){
    // first crop the image according to their h-w-ratio
    int resize_h = dimensions.d[1];
    int resize_w = dimensions.d[2];
    assert(resize_h == resize_w);
    int w = SrcImage.cols;
    int h = SrcImage.rows;
    int off, startX, startY, height, width;
    if ( h < w){
        off = (w - h )/2;
        startX = off+1;
        startY = 0;
        height = h;
        width = h;
    }else{
        off = (h - w)/2;
        startX = 0;
        startY = off+1;
        height = w;
        width = w;
    }
    cv::Mat tmp, tmpAspect;
    cv::Mat ROI(SrcImage, cvRect(startX,startY,width,height));
    ROI.copyTo(tmpAspect);
    // resize to 224x224
    cv::resize(tmpAspect, tmp, cv::Size(resize_w, resize_h));

    cv::Mat imgFloat;
    tmp.convertTo(imgFloat, CV_32FC3);
        
    // minus mean and multiply by scale
    imgFloat = imgFloat - (cv::Scalar(103.94,116.78,123.68));
    if (scale){
        imgFloat = imgFloat.mul(cv::Scalar(0.017,0.017,0.017));
    }
        
    float a = imgFloat.at<cv::Vec3f>(0,0)[1];
    for(int c =0; c < dimensions.d[0];c++){
        for(int h =0;h<dimensions.d[1];h++){	 
            for(int w=0; w< dimensions.d[2];w++){
                float a = imgFloat.at<cv::Vec3f>(h,w)[c];
                imgRow[ c*dimensions.d[1]*dimensions.d[2]+ h*dimensions.d[2] + w ]= a;		
            }
        }
    }
}
void ImgPreprocessCenterCropDirect(cv::Mat SrcImage, float* imgRow, Dims3 dimensions, bool scale){
    // get image from default image preprocessing scrip on Caffe website.
    // default resize to 256x256, then center crop to 224x224
    int resize_h = dimensions.d[1];
    int resize_w = dimensions.d[2];
    assert(resize_h == resize_w);
    int w = SrcImage.cols;
    int h = SrcImage.rows;
   
    cv::Mat tmp, tmp256;
   
    cv::resize( SrcImage, tmp, cv::Size(224, 224));
    
     
    cv::Mat imgFloat;
    tmp.convertTo(imgFloat, CV_32FC3);
        
    // minus mean and multiply by scale
    imgFloat = imgFloat - (cv::Scalar(103.94,116.78,123.68));
    if (scale){
        imgFloat = imgFloat.mul(cv::Scalar(0.017,0.017,0.017));
    }
        
    float a = imgFloat.at<cv::Vec3f>(0,0)[1];
    for(int c =0; c < dimensions.d[0];c++){
        for(int h =0;h<dimensions.d[1];h++){	 
            for(int w=0; w< dimensions.d[2];w++){
                float a = imgFloat.at<cv::Vec3f>(h,w)[c];
                imgRow[ c*dimensions.d[1]*dimensions.d[2]+ h*dimensions.d[2] + w ]= a;		
            }
        }
    }
}

void ImgPreprocessCenterCrop(cv::Mat SrcImage, float* imgRow, Dims3 dimensions, bool scale){
    // get image from default image preprocessing scrip on Caffe website.
    // default resize to 256x256, then center crop to 224x224
    int resize_h = dimensions.d[1];
    int resize_w = dimensions.d[2];
    assert(resize_h == resize_w);
    int w = SrcImage.cols;
    int h = SrcImage.rows;
    int off = (256-resize_h)/2;
    cv::Mat tmp, tmp256;
    
    if(w!=256 || h!=256){
        cv::resize( SrcImage, tmp256, cv::Size(256, 256));
        cv::Mat ROI(tmp256, cvRect(off,off,resize_w,resize_h));
        ROI.copyTo(tmp);  
    }
    else{
        cv::Mat ROI(SrcImage, cvRect(off,off,resize_w,resize_h));
        ROI.copyTo(tmp);
    }
    
     
    cv::Mat imgFloat;
    tmp.convertTo(imgFloat, CV_32FC3);
        
    // minus mean and multiply by scale
    imgFloat = imgFloat - (cv::Scalar(103.94,116.78,123.68));  //B,G,R
    if (scale){
        imgFloat = imgFloat.mul(cv::Scalar(0.017,0.017,0.017));
    }
        
    float a = imgFloat.at<cv::Vec3f>(0,0)[1];
    for(int c =0; c < dimensions.d[0];c++){
        for(int h =0;h<dimensions.d[1];h++){	 
            for(int w=0; w< dimensions.d[2];w++){
                float a = imgFloat.at<cv::Vec3f>(h,w)[c];
                imgRow[ c*dimensions.d[1]*dimensions.d[2]+ h*dimensions.d[2] + w ]= a;		
            }
        }
    }
}
void cal_accuracy(std::vector<int> res_vector, int ylabel, int &top1,int& top5){
    int n = res_vector.size();
    if(res_vector[0] == ylabel) top1++;
    if(res_vector[0] == ylabel 
    || res_vector[1] == ylabel
    || res_vector[2] == ylabel
    || res_vector[3] == ylabel
    || res_vector[4] == ylabel
    ) top5++;
}
int main(int argc, char** argv)
{
    // create a TensorRT model from the caffe model and serialize it to a stream

    if (!parseArgs(argc, argv))
        return -1;

    cudaSetDevice(gParams.device);

    if (gParams.outputs.size() == 0 && !gParams.deployFile.empty())
    {
        std::cerr << "At least one network output must be defined" << std::endl;
        return -1;
    }
 
    ICudaEngine* engine = createEngine();
    if (!engine)
    {
        std::cerr << "Engine could not be created" << std::endl;
        return -1;
    }

    if (gParams.uffFile.empty() && gParams.onnxModelFile.empty())
        nvcaffeparser1::shutdownProtobufLibrary();
    else if (gParams.deployFile.empty() && gParams.onnxModelFile.empty())
        nvuffparser::shutdownProtobufLibrary();

    //CH OpenCV
    size_t bindingIndex = engine->getBindingIndex(gInputs[0].c_str());
    Dims3 dimensions = static_cast<Dims3&&>(engine->getBindingDimensions((int)bindingIndex));
    size_t eltCount = dimensions.d[0]*dimensions.d[1]*dimensions.d[2]*gParams.batchSize, memSize = eltCount * sizeof(float);

    std::ifstream fileList(gParams.testList);
    std::string fileLine;
    printf("File:%s\n",gParams.testList);
    float* imgRow=(float*)malloc(memSize);
    int total_image = 0;
    int top1=0, top5=0;
    // the while loop will iterate all the images in the validation set
    while (std::getline(fileList, fileLine))
    {
        total_image++;
	    MemSet(imgRow,eltCount);
    	std::string imgFile;

        std::cout << "Read file path=" << fileLine<<std::endl;
        std::stringstream ss(fileLine);
        std::string filename, class_id;
    
        std::getline(ss, filename, ' ');
        std::getline(ss, class_id, ' ');
        int ylabel = stoi(class_id);
        std::string full_filename = gParams.data_folder + "/" + filename;
        std::cout << "full filename="<< full_filename;
        cv::Mat SrcImage, SrcImageResize;
        // read each image
        SrcImage = cv::imread(full_filename,  CV_LOAD_IMAGE_COLOR);
        if(!SrcImage.empty()) 
            printf("Success to decode image\n");
        else
            printf("Fail to decode image\n");
        printf("Dim0: %d Dim1:%d Dim2:%d \n", dimensions.d[0],dimensions.d[1],dimensions.d[2]);

        // preprocess the image
        ImgPreprocessCenterCrop(SrcImage, imgRow, dimensions, gParams.scale); // 1st method we choose
        //ImgPreprocessAspectVGG(SrcImage, imgRow, dimensions, gParams.scale);
        // ImgPreprocessCenterCropDirect(SrcImage, imgRow, dimensions, gParams.scale);

        std::vector<int> res_vector;
        res_vector.clear();
        // inference process for each image
        doInference(*engine, imgRow, res_vector);
        cal_accuracy(res_vector, ylabel, top1, top5);
        std::cout << "Top1 accuracy="<<top1*1.0/total_image<< " with "<<total_image<<" images"<< std::endl;
        std::cout << "Top5 accuracy="<<top5*1.0/total_image<< " with "<<total_image<<" images"<< std::endl;
    }
    engine->destroy();
    free(imgRow);
    return 0;
}

