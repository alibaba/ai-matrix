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

#include "op.h"
#include "image_loader.h"
#include "test_harness.h"

#undef VLOG
#include <glog/logging.h>

#include <fstream>
#include <iomanip>

using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::int32;

// Check whether a hgai's HgEngine op is in the graph
bool hasNpuEngineOp(tensorflow::GraphDef& graphDef) {
  for (size_t n=0;n<graphDef.node_size();n++) {
    tensorflow::NodeDef nodeDef = graphDef.node(n);
    if (nodeDef.op() == "AliNPUEngineOp")
      return true;
  }
  return false;
}

// Load quantization scale to convert from pre-processed data into int8 value range
bool loadQuantScale(std::string modelFileName, float* quantScale) {
  /* Try to load the input_quant_nodes.txt from the same directory of quantized 
   * model file
   */
  std::string quantScaleFilePath;
  size_t modelFilePathEndLoc = modelFileName.rfind('/');
  if (modelFilePathEndLoc != std::string::npos)
    quantScaleFilePath = modelFileName.substr(0, modelFilePathEndLoc) + "/";
  quantScaleFilePath += "input_quant_nodes.txt";
  std::ifstream quantScaleFile(quantScaleFilePath.c_str());
  if (quantScaleFile.is_open()) {
    bool foundScale = false;
    char buf[256];
    // Find the line containing in_scale: in the file
    do {
      quantScaleFile.getline(buf, 256);
      if (!quantScaleFile.gcount()) {
        break;
      }
      std::string line(buf);
      size_t loc = line.find("in_scale");
      if (loc != std::string::npos) {
        loc = line.find(':');
        if (loc != std::string::npos) {
          *quantScale = std::stof(line.substr(loc+1));
          foundScale = true;
          break;
        }
      }
    } while (!quantScaleFile.eof());
    return foundScale;
  }
  return false;
}

// Load the tensorflow model
Status LoadModel(TestHarness* th, std::string modelFileName) {
  tensorflow::GraphDef graphDef;
  Status loadModelStatus =
    ReadBinaryProto(tensorflow::Env::Default(), modelFileName, &graphDef);
  if (!loadModelStatus.ok()) {
    return tensorflow::errors::NotFound("Failed to load model at '",
                                        modelFileName, "'");
  }
  // Check if the model contains hgai HgEngine op
  bool hasEngineOp = hasNpuEngineOp(graphDef);

  // If it does, then try to load the quantization scale
  if (hasEngineOp)
    th->quantized = loadQuantScale(modelFileName, &th->quantScale);

  // Create a new tensorflow session from the graph
  th->session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status sessionCreateStatus = th->session->Create(graphDef);
  if (!sessionCreateStatus.ok()) {
    return sessionCreateStatus;
  }
  return Status::OK();
}

// Extract scenarios
std::vector<TestScenario> GetScenarios(std::string& scenarioStr) {
  std::vector<std::string> scenarioStrs;
  std::vector<TestScenario> scenarios;
  size_t pos;
  while ((pos = scenarioStr.find(',')) != std::string::npos) {
    if (pos) {
      scenarioStrs.push_back(scenarioStr.substr(0, pos));
    }
    scenarioStr = scenarioStr.substr(pos+1);
  }
  scenarioStrs.push_back(scenarioStr);
  for (size_t i=0;i<scenarioStrs.size();i++) {
    if (scenarioStrs[i] == "SingleStream")
      scenarios.push_back(TestScenario::SingleStream);
    else if (scenarioStrs[i] == "MultiStream")
      scenarios.push_back(TestScenario::MultiStream);
    else if (scenarioStrs[i] == "Server")
      scenarios.push_back(TestScenario::Server);
    else if (scenarioStrs[i] == "Offline")
      scenarios.push_back(TestScenario::Offline);
  }
  return scenarios;
}

// Get scenario string from scenario enum
std::string GetScenarioStr(TestScenario scenario) {
  switch(scenario) {
  case TestScenario::SingleStream:
    return std::string("SingleStream");
  case TestScenario::MultiStream:
    return std::string("MultiStream");
  case TestScenario::Server:
    return std::string("Server");
  case TestScenario::Offline:
    return std::string("Offline");
  }
  CHECK(0) << "Unknown scenario " << (int)scenario;
  return std::string("");
}

// Warm up process
void warmup(TestHarness* th) {
  int cntWarmup = th->cntWorkers;
  int batchSize = 1;
  std::vector<QuerySampleIndex> qsi(batchSize);
  std::vector<QuerySample> qs(batchSize);
  for (size_t i=0;i<qsi.size();i++) {
    qsi[i] = (QuerySampleIndex)i;
    qs[i].id = 0;
    qs[i].index = (QuerySampleIndex)i;
  }
  LoadSamplesToRam((c::ClientData)th, qsi.data(), batchSize);

  th->cntWarmup = cntWarmup;
  th->warmup = true;
  
  for (int i=0;i<cntWarmup;i++) {
    IssueQuery((c::ClientData)th, qs.data(), batchSize);   
  }

  th->warmupMutex.lock();
  while (th->cntWarmup) {
    th->warmupMutex.unlock();
    sleep(0);
    th->warmupMutex.lock();
  }
  th->warmupMutex.unlock();
  th->warmup = false;

  // Reset some variables affected by the warm up process
  th->cntQ = 0;

  UnloadSamplesFromRam((c::ClientData)th, qsi.data(), batchSize);
}

// Calibration process
void calibration(TestHarness* th, std::string calListFileName) {
  VLOG(1) << "start calibration ...";
  th->calibration = true;
  th->cntDoneCalibration = 0;

  // Load the calibration set list file
  std::ifstream listFile(calListFileName, std::ifstream::in);
  if (!listFile.is_open()) {
    LOG(ERROR) << "can't load calibration list file " << calListFileName;
    return;
  }

  std::vector<QuerySampleIndex> qsi;
  while (!listFile.eof()) {
    std::string fileName;
    listFile >> fileName;
    if (fileName != "" &&
        th->sampleFileNameToIndexMap.find(fileName) != th->sampleFileNameToIndexMap.end()) {
      qsi.push_back((QuerySampleIndex)(th->sampleFileNameToIndexMap[fileName]));
    }
  }

  std::vector<QuerySample> qs(qsi.size());
  for (size_t i=0;i<qsi.size();i++) {
    qs[i].id = 0;
    qs[i].index = qsi[i];
  }
  LoadSamplesToRam((c::ClientData)th, qsi.data(), qsi.size());
  IssueQuery((c::ClientData)th, qs.data(), qs.size());

  th->warmupMutex.lock();
  while (th->cntDoneCalibration < qsi.size()) {
    th->warmupMutex.unlock();
    sleep(0);
    th->warmupMutex.lock();
  }
  th->warmupMutex.unlock();
  UnloadSamplesFromRam((c::ClientData)th, qsi.data(), qsi.size());
  VLOG(1) << "done calibration";
}


int main(int argc, char* argv[]) {
  // Load and register HgAI custom op into tensorflow framework
  Status loadStatus = LoadHgaiOpLibrary();
  if (!loadStatus.ok()) {
    LOG(ERROR) << "can't load lib_ops.so";
    return -1;
  }

  /* 3 workers is enough for large batch, small batch test will cause statistics worker
   * to launch more workers
   */
  int cntWorkers = 3;

  // 3 IssueQuery worker is enough to handle all the necessary tasks
  int cntIqWorkers = 3;

  // Get the magic number of maximal batch count that host memory accessible for HanGuangAI can hold
  int maxBatch = 4800;

  /* Though we have logic to prevent the total batch fly-around exceed the maximal batch
   * but why not starting from a balanced point
   */
  int batchPerQueryItem = maxBatch/cntWorkers;

  // Initial value for command line flags
  std::string dataset = std::getenv("DATA_DIR");
  std::string model = std::string(std::getenv("MODEL_DIR")) + "/fp32.pb";
  int32 width = 224;
  int32 height = 224;
  std::string inputLayer = "input_tensor";
  std::string outputLayer = "ArgMax";

  std::string mlperfConfigFilePath;
  std::string auditConfigFilePath;
  std::string userConfigFilePath;
  std::string calibrationListFilePath;
  std::string scenarioStr = "SingleStream";
  bool accuracyMode = false;
  bool submissionMode = false;
  int runDuration = 0;
  int count = batchPerQueryItem;
  int latencySingle = 0;
  int queriesSingle = 1024;
  int qpsMulti = 24;
  int latencyMulti = 0;
  int streamCnt = 4;
  int queriesMulti = 1024;
  int qpsServer = 3000;
  int latencyServer = 0;
  int queriesServer = 1024;
  int qpsOffline = 40000;
  int queriesOffline = 2000000;
  bool skipWarmup = false;

  std::vector<Flag> flagList = {
      Flag("dataset", &dataset, "dataset location"),
      Flag("model", &model, "model to be executed"),
      Flag("width", &width, "resize image to this width in pixels"),
      Flag("height", &height, "resize image to this height in pixels"),
      Flag("input", &inputLayer, "name of input layer"),
      Flag("output", &outputLayer, "name of output layer"),
      Flag("threads", &cntWorkers, "worker thread count"),
      Flag("threads-issuequery", &cntIqWorkers, "issue query worker thread count"),
      Flag("scenario", &scenarioStr, "mlperf benchmark scenario"),
      Flag("accuracy", &accuracyMode, "enable accuracy mode"),
      Flag("submission", &submissionMode, "enable submission mode"),
      Flag("time", &runDuration, "time to scan in seconds"),
      Flag("count", &count, "dataset items to use"),
      Flag("mlperf-config", &mlperfConfigFilePath, "file path for mlperf.conf file for loadgen settings"),
      Flag("user-config", &userConfigFilePath, "file path for user.conf file for loadgen settings"),
      Flag("audit-config", &auditConfigFilePath, "file path for audit.conf file for loadgen settings"),
      Flag("calibration", &calibrationListFilePath, "file path for calibration list file for calibration"),

      // Single Stream
      Flag("latency-single", &latencySingle, "expected latency for SingleStream in ms, 0 means default"),
      Flag("queries-single", &queriesSingle, "mlperf number of queries for SingleStream"),

      // Multi Stream
      Flag("qps-multi", &qpsMulti, "target qps for MultiStream mode (fps each stream)"),
      Flag("latency-multi", &latencyMulti, "expected latency for MultiStream in ms, 0 means default"),
      Flag("streams", &streamCnt, "stream count in multi stream mode"),
      Flag("queries-multi", &queriesMulti, "mlperf number of queries for MultiStream"),

      // Server
      Flag("qps-server", &qpsServer, "target qps for Server mode"),
      Flag("latency-server", &latencyServer, "expected latency for MultiStream in ms, 0 means default"),
      Flag("queries-server", &queriesServer, "mlperf number of queries for Server"),

      // Offline
      Flag("qps-offline", &qpsOffline, "target qps for Offline mode"),
      Flag("queries-offline", &queriesOffline, "mlperf number of queries for Offline"),

      Flag("batch", &batchPerQueryItem, "batch size per query item"),
      Flag("max-batch", &maxBatch, "max batch size SUT can hold at the same time"),
      Flag("skip-warmup", &skipWarmup, "skip warmup"),
  };

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flagList);
  const bool parseResult = tensorflow::Flags::Parse(&argc, argv, flagList);
  if (!parseResult) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // Get all the scenarios from command line flag
  std::vector<TestScenario> scenarios = GetScenarios(scenarioStr);

  // Maximal batch per work item can not exceed QSL sample count
  if (batchPerQueryItem > count)
    batchPerQueryItem = count;

  // No need to warm up for accuracy mode
  if (accuracyMode)
    skipWarmup = true;

  // Initialize test harness
  TestHarness th;
  th.width = width;
  th.height = height;
  th.inputLayer = inputLayer;
  th.outputLayer = outputLayer;
  th.cntWorkers = cntWorkers;
  th.cntIqWorkers = cntIqWorkers;
  th.batchPerQueryItem = batchPerQueryItem;
  th.maxBatch = maxBatch;
  th.shutdown = false;
  th.warmup = false;
  th.calibration = false;
  th.cntQ = 0;
  th.quantized = false;
  th.quantScale = 1.0f;
  th.loadedSamplesTensor = NULL;
  th.totalBatch = 0;
  th.workerIdle = std::chrono::duration<double>::zero();
  th.statWorker = NULL;
  th.npuTensor = NULL;

  // Load val_map.txt from imagenet
  std::string mapFileName("val_map.txt");
  if (!LoadImageMap((c::ClientData)&th, dataset, mapFileName)) {
    LOG(ERROR) << "Failed to load dataset map file " << dataset << "/" << mapFileName;
    return -1;
  }   

  // First we load and initialize the model.
  Status loadGraphStatus = LoadModel(&th, model);
  if (!loadGraphStatus.ok()) {
    LOG(ERROR) << loadGraphStatus;
    return -1;
  }

  /* Doing calibration process if requested, basically running all sample images
   * in the calibration list and exit right after
   */
  if (calibrationListFilePath != "") {
    if (!LaunchThreads((c::ClientData)&th)) {
      LOG(ERROR) << "Failed to launch worker threads";
      ShutdownThreads((c::ClientData)&th);
      return -1;
    }
    calibration(&th, calibrationListFilePath);
    ShutdownThreads((c::ClientData)&th);
    return 0;
  }

  // Loop over scenarios the command line flag wants to run
  for (int i=0;i<scenarios.size();i++) {
    TestSettings settings;
    settings.scenario = scenarios[i];

    // Launch workers, IssueQuery workers and qsr preparer before the test 
    if (!LaunchThreads((c::ClientData)&th)) {
      LOG(ERROR) << "Failed to launch worker threads";
      ShutdownThreads((c::ClientData)&th);
      return -1;
    }

    // Skip warm up if requested
    if (!skipWarmup)
      warmup(&th);
    
    // Set the test mode
    settings.mode = TestMode::PerformanceOnly;
    if (accuracyMode)
      settings.mode = TestMode::AccuracyOnly;
    if (submissionMode)
      settings.mode = TestMode::SubmissionRun;

    // Set settings from command line option
    if (runDuration) {
      settings.min_duration_ms = runDuration * 1000;
    }
    if (scenarios[i] == TestScenario::SingleStream) {
      settings.min_query_count = queriesSingle;
      if (latencySingle)
        settings.single_stream_expected_latency_ns = latencySingle * 1000000;
    } else if (scenarios[i] == TestScenario::MultiStream) {
      settings.multi_stream_target_qps = qpsMulti;
      settings.min_query_count = queriesMulti;
      settings.multi_stream_samples_per_query = streamCnt;
      settings.multi_stream_max_async_queries = cntWorkers;
      settings.multi_stream_target_latency_percentile = 0.99;
      if (latencyMulti)
        settings.multi_stream_target_latency_ns = latencyMulti * 1000000;
    } else if (scenarios[i] == TestScenario::Offline) {
      settings.min_query_count = queriesOffline;
      if (qpsOffline)
        settings.offline_expected_qps = qpsOffline;
    } else {
      settings.server_target_qps = qpsServer;
      settings.min_query_count = queriesServer;
      settings.server_max_async_queries = 6000;
      if (latencyServer) 
        settings.server_target_latency_ns = latencyServer * 1000000;
    }

    // Overwrite settings from mlperf.conf file
    if (mlperfConfigFilePath != "") {
      LOG(INFO) << "Loading settings from mlperf.conf file " << mlperfConfigFilePath;
      std::string profile = "resnet50";
      if (settings.FromConfig(mlperfConfigFilePath, 
                              profile,
                              GetScenarioStr(scenarios[i]))) {
        LOG(ERROR) << "Invalid config file " << mlperfConfigFilePath;
      }
    }

    // Overwrite settings from audit.conf file
    if (auditConfigFilePath != "") {
      LOG(INFO) << "Loading settings from audit.conf file " << auditConfigFilePath;
      std::string profile = "resnet50";
      if (settings.FromConfig(auditConfigFilePath, 
                              profile,
                              GetScenarioStr(scenarios[i]))) {
        LOG(ERROR) << "Invalid config file " << auditConfigFilePath;
      }
    }

    // Overwrite settings from user.conf file
    if (userConfigFilePath != "") {
      LOG(INFO) << "Loading settings from user.conf file " << userConfigFilePath;
      std::string profile = "resnet50";
      if (settings.FromConfig(userConfigFilePath, 
                              profile,
                              GetScenarioStr(scenarios[i]))) {
        LOG(ERROR) << "Invalid config file " << userConfigFilePath;
      }
    }

    // Construct SUT
    void* sut = c::ConstructSUT((c::ClientData)&th,
                                NULL,
                                0,
                                IssueQuery,
                                FlushQueries,
                                ReportLatencyResults);

    if (!sut) {
      LOG(ERROR) << "Failed to construct system under test";
      ShutdownThreads((c::ClientData)&th);
      return -1;
    }

    // Construct QSL
    void* qsl = c::ConstructQSL((c::ClientData)&th,
                                NULL,
                                0,
                                count,
                                /* rules requires minimal performance count of 
                                 * samples for QSL for resnet is 1024 */
                                (batchPerQueryItem>1024?batchPerQueryItem:1024),
                                LoadSamplesToRam,
                                UnloadSamplesFromRam);

    if (!qsl) {
      LOG(ERROR) << "Failed to construct query sample library";
      c::DestroySUT(sut);
      ShutdownThreads((c::ClientData)&th);
      return -1;
    }

    // Launch statistics worker 
    LaunchStatThread((c::ClientData)&th);

    // Start the test
    c::StartTest(sut, qsl, settings);

    // Destroy QSL and SUT after the test
    c::DestroyQSL(qsl);
    c::DestroySUT(sut);

    // Shutdown all the launched threads
    ShutdownThreads((c::ClientData)&th);
  }

  return 0;
}
