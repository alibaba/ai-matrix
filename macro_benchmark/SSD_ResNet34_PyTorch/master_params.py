# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import torch
import apex_C

def check_type_split(buckets):
    for bucket in buckets:
        this_type = bucket[0].type()
        for param in bucket:
            if param.type() != this_type:
                raise ValueError("Each bucket must contain only params of the same type.")


def create_flat_master(model_buckets):
    # Ideally, we'd like to flatten the model params as well, and reset the float params' .data
    # attributes to point directly into the flattened master buffers.  However, my version that does
    # so is yielding CUDNN_STATUS_BAD_PARAM errors when running with distributed and nhwc.
    # I ended up making the safe choice of not altering what the params' .data members point to.
    check_type_split(model_buckets)

    flat_master_buckets = [apex_C.flatten([p.detach().clone().float() for p in model_bucket])
                           for model_bucket in model_buckets]

    for flat_master in flat_master_buckets:
        flat_master.requires_grad_()

    return flat_master_buckets
