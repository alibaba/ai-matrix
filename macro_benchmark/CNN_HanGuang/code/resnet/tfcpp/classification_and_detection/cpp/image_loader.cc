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

#include <stdio.h>
#include <setjmp.h>
#include <iostream>
#include <jpeglib.h>

#include "tensorflow/cc/ops/standard_ops.h"
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

#include "test_harness.h"
#include "image_loader.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;

Status DecodeJPEGtoMatTF(std::string fileName, cv::Mat& img) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string inputName = "fileReader";
  auto fileReader = tensorflow::ops::ReadFile(root.WithOpName(inputName),
                                              fileName);
  // Now try to figure out what kind of file it is and decode it.
  const int wantedChannels = 3;
  tensorflow::Output imageReader;
  imageReader = DecodeJpeg(root.WithOpName("jpegReader"), fileReader,
                           DecodeJpeg::Channels(wantedChannels));
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dimsExpander = ExpandDims(root.WithOpName("dimsExpander"), imageReader, 0);

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::vector<Tensor> outTensors;
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {"dimsExpander"}, {}, &outTensors));
  Tensor tensor = outTensors[0];
  cv::Mat newImg(tensor.shape().dim_size(1), tensor.shape().dim_size(2), CV_8UC3);
  
  uint8* dstData = (uint8*)newImg.data;
  auto tensorMapped = tensor.tensor<uint8, 4>();

  // copying the data into the corresponding tensor
  for (int y = 0; y < tensor.shape().dim_size(1); ++y) {
    uint8* dstRow = dstData + (y * tensor.shape().dim_size(2) * tensor.shape().dim_size(3));
    for (int x = 0; x < tensor.shape().dim_size(2); ++x) {
      uint8* dstPixel = dstRow + (x * tensor.shape().dim_size(3));
      for (int c = 0; c < tensor.shape().dim_size(3); ++c) {
        uint8* dstValue = dstPixel + c;
        *dstValue = tensorMapped(0, y, x, c);
      }
    }
  }
  newImg.copyTo(img);
  return Status::OK();
}

Status LoadImage(std::string fileName, 
                 cv::Size& outputDim, cv::Mat& img) {
  cv::Mat newImg;
  TF_RETURN_IF_ERROR(DecodeJPEGtoMatTF(fileName, newImg));

  // Part of imagenet preprocessing process copied from MLPerf Reference Implementation
  // 
  // Resizing
  cv::Size s = newImg.size();
  float scale = 87.5;
  int newHeight = (int)(100.0f * outputDim.height/ scale);
  int newWidth = (int)(100.0f * outputDim.width/ scale);
  int w, h;
  if (s.height > s.width) {
    w = newWidth;
    h = (int)(newHeight * s.height / s.width);
  } else {
    h = newHeight;
    w = (int)(newWidth * s.width / s.height);
  }
  cv::resize(newImg, newImg, cv::Size(w, h), 0, 0, cv::INTER_AREA);

  // Croping
  s = newImg.size();
  int left = (int)((s.width-outputDim.width)/2);
  int top  = (int)((s.height-outputDim.height)/2);
  cv::Mat cropedNewImg = newImg(cv::Rect(left, top, outputDim.width, outputDim.height));

  // Converting to float and subtracting by predefined mean value
  cv::Mat floatImg;
  cropedNewImg.convertTo(floatImg, CV_32FC1);
  floatImg -= cv::Scalar(123.68, 116.78, 103.94);

  floatImg.copyTo(img);
  return Status::OK();
}

