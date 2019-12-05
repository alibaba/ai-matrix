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

#include "op.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
Status LoadLibrary(const char* library_filename, void** result,
                   const void** buf, size_t* len);
}

Status LoadHgaiOpLibrary() {
  void* handle = NULL;
  const void* buf = NULL;
  size_t len = 0;
  std::string filename = "lib_ops.so.";
  filename += std::to_string(TF_MAJOR_VERSION) + "." +
              std::to_string(TF_MINOR_VERSION);

  TF_RETURN_IF_ERROR(tensorflow::LoadLibrary(filename.c_str(), &handle, &buf, &len));

  filename = "lib_ops_pub.so.";
  filename += std::to_string(TF_MAJOR_VERSION) + "." +
              std::to_string(TF_MINOR_VERSION);
  TF_RETURN_IF_ERROR(tensorflow::LoadLibrary(filename.c_str(), &handle, &buf, &len));
}
