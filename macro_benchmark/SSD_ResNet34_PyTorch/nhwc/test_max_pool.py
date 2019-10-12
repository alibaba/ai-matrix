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

from nhwc.conv import *
from nhwc.max_pool import *
import torch

torch.backends.cudnn.benchmark=True

N = 64
C = 256
H = 56
W = 56

kernel = 3
stride = 1
padding = 1

H_out = 1 + (H + 2*padding - kernel) // stride
W_out = 1 + (W + 2*padding - kernel) // stride

pool = MaxPool2d_NHWC(kernel_size=kernel, stride=stride, padding=padding)

# Make a NCHW copy of everything
pool_nchw = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding).cuda()

x = torch.randn(N, C, H, W).cuda().half()

# Copy input tensor
x_copy = x.clone()
x_copy.requires_grad = True

# Transpose -> NHWC
x = x.permute(0,2,3,1).contiguous()
x.requires_grad = True

g0 = torch.randn(N, H_out, W_out, C).cuda().half()
g0_nchw = g0.clone().permute(0,3,1,2).contiguous()

out = pool(x)
#out = out.relu_()
out.backward(g0)

out_nchw = pool_nchw(x_copy)
#out_nchw = out_nchw.relu_()

out_nchw.backward(g0_nchw)

out_nhwc = out.permute(0,3,1,2)
#print(out_nhwc)
#print(out_nchw)

print(torch.allclose(out_nhwc, out_nchw, atol=1e-5, rtol=1e-3))
print(torch.allclose(x.grad.permute(0,3,1,2), x_copy.grad, atol=1e-5, rtol=1e-3))

