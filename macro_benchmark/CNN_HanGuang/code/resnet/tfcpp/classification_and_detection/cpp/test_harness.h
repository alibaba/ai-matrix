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

#ifndef __TEST_HARNESS_H__
#define __TEST_HARNESS_H__

#include <map>
#include <mutex>
#include <memory>
#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <opencv2/core/core.hpp>
#include "c_api.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "alinpu_ratelrt_c_api_pub.h"
using tensorflow::Tensor;

using namespace mlperf;

// Defines all the information needed for qsr preparer to prepare Loadgen's qsr
typedef struct _QueryResponseItem { 
  int qid;                     /* Query index of the work item */
  int iid;                     /* Item index of the work item */
  QuerySampleResponse* qsr;    /* Array of qsr to return result back to LoadGen */
  const QuerySample* qs;       /* Array of qs from LoadGen for this work item */
  size_t cntQs;                /* Array count of both qsr and qs */
  std::mutex m;                /* Mutex to control the access to qsr structure */
  std::condition_variable cv;  /* CV to help to monitor the qsr readiness */
  bool ready;                  /* The readiness of qsr structure */
} QueryResponseItem; 
  
// Defines a work item for consecutive part of query from loadgen
typedef struct _QueryItem {
  Tensor tensor;               /* The sliced tensor for the work item */
  QueryResponseItem qri;       /* The information needed for worker to do its work */
} QueryItem;

// Defines a query from LoadGen
typedef struct _Query {
  int qid;                     /* The universal index of the query */
  const QuerySample* qs;       /* The qs from IssueQuery */
  size_t cntQs;                /* The cntQs from IssueQuery */
} Query;

// Defines informations used by test harness
typedef struct _TestHarness {
  int height;                  /* Input image height */
  int width;                   /* Input image width */
  std::string inputLayer;      /* Input layer name in the resnet50 model */
  std::string outputLayer;     /* Output layer name in the resnet50 model */
  std::unique_ptr<tensorflow::Session> session;
                               /* Tensorflow session created for image inferences of the test */

  int cntWorkers;              /* Count of workers  */
  int cntIqWorkers;            /* Count of IssueQuery workers */
  int batchPerQueryItem;       /* Maximal batch count per work item */
  int maxBatch;                /* Maximal batch fly around >= batchPerQueryItem*cntWorkers */

  std::vector<std::string> sampleFileNames;
                               /* Corresponding file names loaded from val_map.txt */
  std::map<std::string, int> sampleFileNameToIndexMap;
                               /* File name to sample index map */
  std::vector<QuerySampleIndex> loadedSamplesIndex;
                               /* Query sample index set from last LoadSamplesToRam */
  std::map<QuerySampleIndex, std::vector<int>> loadedSamplesIndexMap;
                               /* Map qsi content to the qsi index of last LoadSamplesToRam */
  Tensor* loadedSamplesTensor; /* Tensorflow tensor created corresponding to last LoadSamplesToRam */
                               
  ratelTensor npuTensor;          /* hgai tensor created corresponding to last LoadSamplesToRam */

  bool quantized;              /* whether type of input tensor shall be the int8 or not */
  float quantScale;            /* The quantization scale value to scale original pre-processed data
                                  into int8 value range */

  std::vector<std::thread*> iqWorkers;
                               /* IssueQuery workers whose responsibility is to break queries from 
                                  query queue into work items in work item queue */
  std::queue<Query*> qQueue;   /* Query queue */
  std::mutex qQueueMutex;      /* Mutex to control the access of query queue */
  std::condition_variable qQueueCV;
                               /* CV to help to monitor the condition of the query queue */

  std::vector<std::thread*> workers; 
                               /* workers whose responsibility is to handle work item from 
                                  work item queue */
  std::thread* qsrPreparer;    /* qsr preparer thread whose responsibility is to allocate and
                                  prepare qsr in qsr queue before workers fills result in 
                                  each qsr corresponding work item */
  std::queue<QueryItem*> qiQueue;
                               /* Work item queue */
  std::mutex qiQueueMutex;     /* Mutex to control the access of work item queue */
  int totalBatch;              /* Total batches currently processed by HanGuangAI */
  std::condition_variable qiQueueCV;
                               /* CV to help to monitor the condition of work item queue */
  std::queue<QueryResponseItem*> qriQueue;
                               /* qsr queue */
  std::mutex qriQueueMutex;    /* Mutex to control the access of qsr queue */
  std::condition_variable qriQueueCV;
                               /* CV to help to monitor the condition of qsr queue */

  bool shutdown;               /* Shutdown signal to notify all threads to exit */

  bool warmup;                 /* Whether the test harness is in warm up mode or not */
  bool calibration;            /* Whether the test harness is in calibration mode or not */
  int cntWarmup;               /* How many workers complete their warm up process */
  int cntDoneCalibration;      /* How many calibration sample inference has completed */
  std::mutex warmupMutex;      /* Mutex to control the access of cntWarmup or cntDoneCalibration
                                  warm up process and calibration process is mutually exclusive */
  int cntQ;                    /* Total count of received queries */

  std::vector<int64_t> latencies;
                               /* Storage to hold latency reports */
  std::chrono::duration<double> workerIdle;
                               /* Total accumulated idle time of workers within a duration */
  std::mutex statMutex;        /* Mutex to control the access of workerIdle */
  std::thread* statWorker;     /* Statistics worker whose responsibility is to monitor the busy 
                                  status of the workers and launch more workers if needed */
} TestHarness;

// Launch workers, IssueQuery workers and qsr preparer before the test 
bool LaunchThreads(c::ClientData);
// Launch statistics worker 
bool LaunchStatThread(c::ClientData);
/* Shutdown all launched threads, including workers, IssueQuery workers, 
   qsr preparer and statistics workers */
void ShutdownThreads(c::ClientData);

// Load val_map.txt
bool LoadImageMap(c::ClientData, const std::string&, const std::string&);

// LoadSamplesToRam implementation for QSL
void LoadSamplesToRam(c::ClientData, const QuerySampleIndex*, size_t);
// UnloadSamplesFromRam implementation for QSL
void UnloadSamplesFromRam(c::ClientData, const QuerySampleIndex*, size_t);

// IssueQuery implementation for SUT
void IssueQuery(c::ClientData, const QuerySample*, size_t);
// FlushQueries implementation for SUT
void FlushQueries();
// ReportLatencyResults implementation for SUT
void ReportLatencyResults(c::ClientData, const int64_t*, size_t);
#endif
