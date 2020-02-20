# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.05-py3
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace/object_detection

COPY requirements.txt .
RUN pip install --no-cache-dir https://github.com/mlperf/training/archive/6289993e1e9f0f5c4534336df83ff199bd0cdb75.zip#subdirectory=compliance \
 && pip install --no-cache-dir -r requirements.txt

# Copy detectron code and build
COPY . .
RUN pip install -e .

ENV OMP_NUM_THREADS=1
ENV OPENCV_FOR_THREADS_NUM=1
