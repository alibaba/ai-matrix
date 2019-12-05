/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "test_harness.h"
#include "image_loader.h"

#include <fstream>
#include <iomanip>
#include <time.h>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

#undef VLOG
#include <glog/logging.h>

// Load val_map.txt
bool LoadImageMap(c::ClientData handle, const std::string& dsPath, const std::string& mapFileName) {
  TestHarness* th = (TestHarness*)handle;
  std::ifstream mapFile(dsPath + "/" + mapFileName, std::ifstream::in);

  if (!mapFile.is_open())
    return false;

  while (!mapFile.eof()) {
    std::string fileName;
    mapFile >> fileName;
    th->sampleFileNameToIndexMap[fileName] = th->sampleFileNames.size();
    th->sampleFileNames.push_back(dsPath + "/" + fileName);

    int label;
    mapFile >> label;
  }
  
  return true;
}

// LoadSamplesToRam implementation for QSL
void LoadSamplesToRam(c::ClientData handle, const QuerySampleIndex* qsi, size_t cntQsi) {
  TestHarness* th = (TestHarness*)handle;
  if (!qsi || !cntQsi)
    return;

  VLOG(1) << "Loading " << cntQsi << " samples";

  cv::Size outputDim(th->width, th->height);
  std::vector<cv::Mat> images;

  // Loading images
  for (size_t i=0;i<cntQsi;i++) {
    cv::Mat img;
    
    // Save the qsi sequence to help on tensor slicing 
    th->loadedSamplesIndex.push_back(qsi[i]);
    th->loadedSamplesIndexMap[qsi[i]].push_back(i);
    std::string fileName = th->sampleFileNames[qsi[i]];
    VLOG(3) << "Loading sample[" << i << "]: file[" 
      << qsi[i] << "]: " << th->sampleFileNames[qsi[i]];
    Status loadStatus = LoadImage(fileName, outputDim, img);
    if (!loadStatus.ok()) {
      LOG(ERROR) << "Can't load image " << fileName;
      CHECK(0);
    }
    images.push_back(img);
  }

  /* Preparing the aggregate tensor comprising of all the images in this load.
   * Sections of sub-tensor will be sliced from this aggregate tensor when matching qsi sequence
   * is found from qs of IssueQuery call
   */
  CHECK(!th->loadedSamplesTensor) << "Loaded samples are not unloaded before another LoadSamplesToRam";
  int imageChannels = images[0].channels();
  if (th->quantized) {
    CHECK(cntQsi < th->maxBatch) << "Count of Loaded Samples " << cntQsi << " exceed max loadable batch size";
    /* Load pre-processed image data into host memory accessible by HanGuang NPU
     * https://github.com/mlperf/inference_policies/issues/55
     * https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#9-faq
     *   Q: Can we preload image data in host memory somewhere that is mapped into accelerator memory?
     *   A: Yes, provided the image data isn’t eventually cached on the device.  
     */
    ratelDataType dtype = RATEL_INT8;
    size_t size = cntQsi*th->height*th->width*imageChannels*sizeof(char);

    ratelDims dims;
    dims.nbDims = 4;
    dims.d[0] = cntQsi;
    dims.d[1] = th->width;
    dims.d[2] = th->height;
    dims.d[3] = imageChannels;
     
    ratelTensorInfo desc;
    memset(&desc, 0, sizeof(ratelTensorInfo));
    desc.dims = dims;
    desc.npuType = dtype;
    desc.usage = RATEL_USAGE_INPUT;
    desc.layout = RATEL_LINEARLAYOUT;
    desc.batches = 1;

    // Create a int8 hgai aggregate tensor residing in host memory
    ratelTensorCreate(&desc, nullptr, RATEL_MEM_HOST, &th->npuTensor);

    /* Get alignment requirement for the specific tensor
     * https://github.com/mlperf/inference_policies/issues/51
     *   You should feel free to pad to the hardware's required alignment during 
     *   untimed pre-processing
     */
    ratelTensorSize npuSize = {0};
    ratelTensorGetSize(th->npuTensor, &npuSize);

    // Get the tensor start data pointer
    int8_t* pTensorStart = nullptr;
    ratelTensorMap(th->npuTensor, (void**)&pTensorStart);

    // Create a corresponding tensorflow tensor with matching layout post alignment
    auto alignWidth = npuSize.rowStride;
    auto alignHeight = npuSize.sliceStride / npuSize.rowStride;
    CHECK(alignHeight == th->height);
    th->loadedSamplesTensor = new Tensor(tensorflow::DT_INT8,
      tensorflow::TensorShape({cntQsi,alignHeight,alignWidth,1}));
    if (!th->loadedSamplesTensor) {
      LOG(ERROR) << "Can't create tensor with shape ["
        << cntQsi << ","
        << alignHeight << ","
        << alignWidth << "]";
      CHECK(0);
    }

    // Load pre-processed image data into the aggregate tensor
    for (int n = 0; n < cntQsi; n++) {
      cv::Mat imgInt8;

      /* https://github.com/mlperf/inference_policies/issues/78
       *   WG approves Multiplying by a constant as part of pre-processing is fine.
       */
      if (th->quantScale != 1.0f)
        images[n] *= th->quantScale;

      // Convert from float to int8
      images[n].convertTo(imgInt8, CV_8SC3);

      int8_t* dst = pTensorStart + n * npuSize.sliceStride;
      int8_t* src = (int8_t*)imgInt8.data;

      for (int h = 0; h < th->height; h++) {
        memcpy(dst, src, th->width * imageChannels);
        dst += npuSize.rowStride;
        src += th->width * imageChannels;          
      }
    }
    ratelTensorUnmap(th->npuTensor);

    /* Bind the hgai tensor to tensorflow tensor so that hgai tensor is used for 
     * input once bound tensorflow tensor is detected
     */
    const void* data = 
      reinterpret_cast<const void*>((th->loadedSamplesTensor)->flat<signed char>().data());
    ratelTensorBindClient(th->npuTensor, data);
  } else {
    // Create a float tensorflow aggregate tensor
    th->loadedSamplesTensor = new Tensor(tensorflow::DT_FLOAT,
      tensorflow::TensorShape({cntQsi,th->height,th->width,imageChannels}));
    if (!th->loadedSamplesTensor) {
      CHECK(0) << "Can't create tensor with shape ["
        << cntQsi << ","
        << th->height << ","
        << th->width << ","
        << imageChannels << "]";
    }

    auto tensorMapped = th->loadedSamplesTensor->tensor<float, 4>();
    for (int n = 0; n < cntQsi; n++) {
      for (int h = 0; h < th->height; h++) {
        for (int w = 0; w < th->width; w++) {
          for (int c = 0; c < imageChannels; c++) {
            const float* src = (float*)images[n].data +
              h * th->width * imageChannels + w * imageChannels + c;
            tensorMapped(n, h, w, c) = *src;
          }
        }
      }
    }
  }
}

// UnloadSamplesFromRam implementation for QSL
void UnloadSamplesFromRam(c::ClientData handle, const QuerySampleIndex* qsi, size_t cntQsi) {
  TestHarness* th = (TestHarness*)handle;
  if (!qsi || !cntQsi)
    return;

  VLOG(1) << "Unloading " << cntQsi << " samples";
  if (th->npuTensor) {
    ratelTensorDestroy(th->npuTensor);
    th->npuTensor = NULL;
  }
  delete th->loadedSamplesTensor;
  th->loadedSamplesIndex.clear();
  for (auto it = th->loadedSamplesIndexMap.begin(); 
       it != th->loadedSamplesIndexMap.end(); ++it) {
    it->second.clear();
  }
  th->loadedSamplesIndexMap.clear();
  th->loadedSamplesTensor = NULL;
}

/* Worker threads monitor the work item queue and once any of them detect work item available
 * in the work item queue, it will start the inference for the work item. After the inference
 * is done, it will wait if qsr structure is not ready, or it will fill the result into qsr
 * and report back to LoadGen right away.
 */
void worker(c::ClientData handle, int id) {
  VLOG(1) << "worker[" << id << "]: online";
  TestHarness* th = (TestHarness*)handle;
  bool warmUp = false;
  auto startTime = std::chrono::system_clock::now();
  int lastBatch = 0;
  while (!th->shutdown) {
    std::unique_lock<std::mutex> lk(th->qiQueueMutex);

    // Decrease the total batch count on the fly from last work item
    if (lastBatch) {
      th->totalBatch -= lastBatch;
      lastBatch = 0;
    }

    /* The clear condition of the big wait:
     *   1. work item queue is not empty
     *   2. and total fly around batch including the coming work item not exceeding the total
     *      batch count that can reside on the HanGuangAI accessible host memory
     *   3. If in the warm up mode, this thread has not done warming up
     *   4. queue lock is acquired
     */
    if (!th->qiQueue.empty() && 
        ((th->qiQueue.front())->qri.cntQs+th->totalBatch<=th->maxBatch) && 
        (!th->warmup || !warmUp) ||
        // worker idle time statistics shall start right after first issuequery is sent
        th->qiQueueCV.wait_for(lk, std::chrono::milliseconds(1), 
          [&th, &warmUp, &startTime, &id] { 
            bool ready = 
              !th->qiQueue.empty() && 
              ((th->qiQueue.front())->qri.cntQs+th->totalBatch<=th->maxBatch) && 
              (!th->warmup || !warmUp);
            th->statMutex.lock();
            th->workerIdle += std::chrono::system_clock::now()-startTime;
            startTime = std::chrono::system_clock::now();
            th->statMutex.unlock();
            return ready;
          })) {
      bool failed = false;
      QueryItem* qitem = th->qiQueue.front();
      lastBatch = qitem->qri.cntQs;
      th->totalBatch += lastBatch;
      th->qiQueue.pop();
      // Releasing the lock right after a work item is acquired
      lk.unlock();

      VLOG(2) << "worker[" << id << "]: is on [" 
        << qitem->qri.qid << ":" << qitem->qri.iid << "]" ;

      // Infer the sliced tensor of this work item
      std::vector<Tensor> outputs;
      Status runStatus = th->session->Run({{th->inputLayer, qitem->tensor}},
                                           {th->outputLayer}, {}, &outputs);

      if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
        failed = true;
      }

      // Get the inference result from output tensor
      Tensor indices = outputs[0];
      tensorflow::TTypes<tensorflow::int64>::Flat indicesFlat = 
        indices.flat<tensorflow::int64>();
      
      VLOG(2) << "worker[" << id << "]: is done running [" 
        << qitem->qri.qid << ":" << qitem->qri.iid << "]" ;
      {
        // Wait for qsr readiness. qsr structure allocation and initialization is done in qsr preparer
        std::unique_lock<std::mutex> lkQsi(qitem->qri.m);
        qitem->qri.cv.wait(lkQsi, [&qitem] { return qitem->qri.ready;});

        VLOG(4) << "worker[" << id << "]: starts to report [" 
          << qitem->qri.qid << ":" << qitem->qri.iid << "]" ;
        if (qitem->qri.qsr) {
          if (!failed) {
            // Filling result
            for (int i=0;i<qitem->qri.cntQs;i++) {
              int result = indicesFlat(i)-1;
              *(int*)(qitem->qri.qsr[i].data) = result;
            }
          }
          /* For warm up and calibration, query samples are generated by test harness instead of 
           * LoadGen, thus no need to call QuerySamplesComplete 
           */
          if (!th->warmup && !th->calibration) {
            c::QuerySamplesComplete(qitem->qri.qsr, qitem->qri.cntQs);
          } else if (th->calibration) {
            /* Accumulate the completed calibration count for test harness to determine whether
             * the whole calibration is done or not
             */
            th->warmupMutex.lock();
            th->cntDoneCalibration += qitem->qri.cntQs;
            th->warmupMutex.unlock();
          } else {
            /* Mark warm up done for itself and also decrease the warm up count for test harness
             * to determine whether the warm up process is done or not
             */
            warmUp = true;
            th->warmupMutex.lock();
            th->cntWarmup--;
            th->warmupMutex.unlock();
          }
          delete qitem->qri.qsr;
        }
        VLOG(3) << "worker[" << id << "]: done [" 
          << qitem->qri.qid << ":" << qitem->qri.iid << "]" ;
        delete qitem;
      }
      startTime = std::chrono::system_clock::now();
    }
  }
  VLOG(1) << "worker[" << id << "]: offline";
}

/* qsr preparer thread monitors the qsr queue and once a qsr preparation task is ready, it will 
 * allocate and initialize the qsr structure, which is part of a work item. 
 */
void qsrPreparer(c::ClientData handle) {
  VLOG(1) << "qsr preparer online";
  TestHarness* th = (TestHarness*)handle;
  while (!th->shutdown) {
    std::unique_lock<std::mutex> lk(th->qriQueueMutex);
    // Wait till task is coming in
    if (!th->qriQueue.empty() ||
        th->qriQueueCV.wait_for(lk, std::chrono::milliseconds(1), [&th]{ return !th->qriQueue.empty();})) {
      QueryResponseItem* qri = th->qriQueue.front();
      th->qriQueue.pop();
      // Releasing the lock right after a task is acquired
      lk.unlock();

      // Allocating the qsr structure
      VLOG(4) << "qsr preparer is on ["<< qri->qid << ":" << qri->iid << "]" ;
      void* mem = malloc((sizeof(QuerySampleResponse) +
                          sizeof(int)) * qri->cntQs);
      if (mem) {
        // Initialize the qsr structure
        qri->qsr = (QuerySampleResponse*)mem;
        uintptr_t data = (uintptr_t)(qri->qsr + qri->cntQs);
        memset((void*)data, 0, sizeof(int)*qri->cntQs);
        for (int i=0;i<qri->cntQs;i++) {
          qri->qsr[i].id = qri->qs[i].id;
          qri->qsr[i].data = data + i * sizeof(int);
          qri->qsr[i].size = sizeof(int);  
        }
      }
      VLOG(4) << "qsr preparer is done ["<< qri->qid << ":" << qri->iid << "]" ;
      qri->m.lock();
      qri->ready = true;
      qri->m.unlock();
      qri->cv.notify_one();
    }
  }
  VLOG(1) << "qsr preparer offline";
}

// IssueQuery implementation for SUT
void IssueQuery(c::ClientData handle, const QuerySample* qs, size_t cntQs) {
  TestHarness* th = (TestHarness*)handle;

  // Allocate a new query index
  int qid = th->cntQ++;

  VLOG(1) << "IssueQuery[" << qid << "]: " << cntQs << " samples";
  Query* q = new Query;
  if (!q) {
    LOG(ERROR) << "Error allocating query object";
    return;
  }
  q->qid = qid;
  q->qs = qs;
  q->cntQs = cntQs;

  // Send the query to query queue and return right after
  th->qQueueMutex.lock();
  th->qQueue.push(q);
  th->qQueueMutex.unlock();
  th->qQueueCV.notify_one();
}

/* Statistics worker monitor worker idle time for the duration of its sleep to see if 
 * all the workers are really busy while overall on-the-fly batch count is really small
 * If similar situation happens for serveral sleep consecutively, meaning the workers 
 * are too busy handling massive small batche queries, a new worker will be launched to
 * relieve the pressure.
 */
void statWorker(c::ClientData handle) {
  VLOG(1) << "statWorker: online";
  TestHarness* th = (TestHarness*)handle;
  std::chrono::duration<double> duration;
  struct timespec t, tmp;
  t.tv_sec = 0;
  t.tv_nsec = 10000000;

  th->statMutex.lock();
  auto lastIdle = th->workerIdle;
  th->statMutex.unlock();

  auto start = std::chrono::system_clock::now();
  auto curIdle = lastIdle;
  /* count down for same busy signal happens for consecutive sleeps, worker count square
   * is used to make it more difficult to launch a new worker when the worker count 
   * increases.
   */
  int countDown = th->workers.size()*th->workers.size()/8;

  while (!th->shutdown) {
    nanosleep(&t, &tmp);

    th->statMutex.lock();
    curIdle = th->workerIdle;
    th->statMutex.unlock();

    duration = (std::chrono::system_clock::now() - start);
    start = std::chrono::system_clock::now();
    
    if (!th->shutdown) {
      auto workerIdleDiff = curIdle-lastIdle;
      float workerIdlePercentage = workerIdleDiff.count()/duration.count()/th->workers.size();

      if (workerIdlePercentage < 0.002f && th->totalBatch < 32) {
        countDown--;
        if (!countDown) {
          // Launch new worker
          std::thread* t = new std::thread(worker, handle, th->workers.size());
          if (t) {
            th->workers.push_back(t);
          }
          countDown = th->workers.size()*th->workers.size()/8;
        }
      } else {
        // reset the count down as last signal is not set
        countDown = th->workers.size()*th->workers.size()/8;
      }
      
      lastIdle = curIdle;
    }
  }

  VLOG(1) << "statWorker: offline";
}

// Launch statistics worker 
bool LaunchStatThread(c::ClientData handle) {
  TestHarness* th = (TestHarness*)handle;
  th->statWorker = new std::thread(statWorker, handle);
  if (!th->statWorker)
    return false;
  return true;
}

/* IssueQuery workers monitor the query queue and once a new query is acquired, it will break the
 * query into sections containing consecutive qsi and create new work items and tensor slices of 
 * the aggregate tensor for the sections. New work items will be sent to work item queue, the qsr
 * in the work item will be sent to qsr queue. So the qsr structure allocation and initialization 
 * can be done along side the work item inference.
 */
void iqWorker(c::ClientData handle, int id) {
  VLOG(1) << "iqWorker[" << id << "]: online";
  TestHarness* th = (TestHarness*)handle;

  while (!th->shutdown) {
    std::unique_lock<std::mutex> lk(th->qQueueMutex);
    // Wait till task is coming in
    if (!th->qQueue.empty() || 
        th->qQueueCV.wait_for(lk, std::chrono::milliseconds(1), 
        [&th] { 
          return !th->qQueue.empty();
        })) {
      Query* q = th->qQueue.front();
      th->qQueue.pop();
      // Releasing the lock right after a task is acquired
      lk.unlock();

      int qid = q->qid;
      const QuerySample* qs = q->qs;
      size_t cntQs = q->cntQs;

      VLOG(2) << "iqWorker[" << id << "]: got query[" << qid <<"]";

      int cntI = 0;
      while (cntQs) {
        // Find the maximal batch for the next section
        int maxCnt = (cntQs<th->batchPerQueryItem)?cntQs:th->batchPerQueryItem;

        /* Assume a single query index has multiple occurance of the loaded sample sequence.
         * For example, when QSL count is smaller than sample count in a single query. Might happen?
         * Anyway, from the first occurance of the index in the loaded sample sequence, find
         * the maximal consecutive index match starting from current position in the query
         * to create a new section.
         */
        const std::vector<int>& map = th->loadedSamplesIndexMap[qs[0].index];
        int start = map[0];
        int qiBatchCnt = 1;

        for (size_t location=0;location<map.size();location++) {
          int cnt = 1;
          while (cnt < maxCnt && (qs[cnt].index == th->loadedSamplesIndex[map[location]+cnt]))
            ++cnt;
          if (cnt > qiBatchCnt) {
            start = map[location];
            qiBatchCnt = cnt;
          }
        }

        // Create a new work item for the section
        QueryItem* qi = new QueryItem;
        if (!qi) {
          LOG(ERROR) << "Error allocating QuestItem";
          return ;
        }

        qi->qri.qsr = NULL;
        qi->qri.qs = qs;
        qi->qri.cntQs = qiBatchCnt;
        qi->qri.ready = false;
        qi->qri.qid = qid;
        qi->qri.iid = cntI++;
        // Slice the tensor for the section
        qi->tensor = th->loadedSamplesTensor->Slice(start, start+qiBatchCnt);

        VLOG(2) << "iqWorker[" << id << "]: query[" << qid << "]: [" << qi->qri.iid 
          << "]: dispatch t[" << start << ":" << start+qiBatchCnt << "]";

        // Send the work item to work item queue
        th->qiQueueMutex.lock();
        th->qiQueue.push(qi);
        th->qiQueueMutex.unlock();
        th->qiQueueCV.notify_one();

        // At the same time, send the qsr to the qsr preparation queue
        th->qriQueueMutex.lock();
        th->qriQueue.push(&qi->qri);
        th->qriQueueMutex.unlock();
        th->qriQueueCV.notify_one();

        // Move the position in the query for the next section
        qs += qiBatchCnt;
        cntQs -= qiBatchCnt;
      }

      delete q;
    }
  }
  VLOG(1) << "iqWorker[" << id << "]: offline";
}

// Launch workers, IssueQuery workers and qsr preparer before the test 
bool LaunchThreads(c::ClientData handle) {
  TestHarness* th = (TestHarness*)handle;
  for (int i=0;i<th->cntIqWorkers;i++) {
    std::thread* t = new std::thread(iqWorker, handle, i);
    if (t) {
      th->iqWorkers.push_back(t);
    }
  }
  for (int i=0;i<th->cntWorkers;i++) {
    std::thread* t = new std::thread(worker, handle, i);
    if (t) {
      th->workers.push_back(t);
    }
  }
  th->qsrPreparer = new std::thread(qsrPreparer, handle);
  if (th->iqWorkers.empty() || th->workers.empty() || !th->qsrPreparer)
    return false;
  return true;
}

/* Shutdown all launched threads, including workers, IssueQuery workers, 
   qsr preparer and statistics workers */
void ShutdownThreads(c::ClientData handle) {
  TestHarness* th = (TestHarness*)handle;
  th->shutdown = true;

  for (size_t i=0;i<th->iqWorkers.size();i++) {
    th->iqWorkers[i]->join();
    delete th->iqWorkers[i];
  }
  th->iqWorkers.clear();
  for (size_t i=0;i<th->workers.size();i++) {
    th->workers[i]->join();
    delete th->workers[i];
  }
  th->workers.clear();
  if (th->qsrPreparer) {
    th->qsrPreparer->join();
    delete th->qsrPreparer;
    th->qsrPreparer = NULL;
  }
  if (th->statWorker) {
    th->statWorker->join();
    delete th->statWorker;
    th->statWorker = NULL;
  }
  th->shutdown = false;
}

// FlushQueries implementation for SUT
void FlushQueries() {
  VLOG(1) << "FlushQueries";
}

// ReportLatencyResults implementation for SUT
void ReportLatencyResults(c::ClientData handle, const int64_t* latencies, size_t cntQueries) {
  VLOG(1) << "ReportLatencyResults: " << cntQueries << " queries";
  TestHarness* th = (TestHarness*)handle;
  th->latencies.resize(cntQueries);
  memcpy(th->latencies.data(), latencies, cntQueries * sizeof(int64_t));
}
