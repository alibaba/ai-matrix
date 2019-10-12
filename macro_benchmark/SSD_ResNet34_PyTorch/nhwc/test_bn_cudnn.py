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

from conv import *
from batch_norm import *
import torch

torch.backends.cudnn.benchmark=True

N = 64
C = 256
H = 56
W = 56

fuse_relu = True
fuse_add = True

bn = BatchNorm2d_NHWC(C, fuse_relu=fuse_relu).cuda()

# Make a NCHW copy of everything
bn_nchw = torch.nn.BatchNorm2d(C).cuda()
# Need to set this for consistency
bn_nchw.eps = 1e-4

# copy the parameters
bn_nchw.weight = torch.nn.Parameter(bn.weight.clone())
bn_nchw.bias = torch.nn.Parameter(bn.bias.clone())

# generate random inputs
x = torch.randn(N, C, H, W).cuda().half()
z = torch.randn(N, C, H, W).cuda().half()

# Copy input tensors
x_copy = x.clone()
z_copy = z.clone()
x_copy.requires_grad = True
z_copy.requires_grad = True


# Transpose -> NHWC
x = x.permute(0,2,3,1).contiguous()
z = z.permute(0,2,3,1).contiguous()
x.requires_grad = True
z.requires_grad = True

# generate a random signal to backprop, copy
g0 = torch.randn(N, H, W, C).cuda().half()
g0_nchw = g0.clone().permute(0,3,1,2).contiguous()

# Run NHWC fwd
out = bn(x, z if fuse_add else None)
# Run NHWC bwd
out.backward(g0)

# Run NCHW fwd
out_nchw = bn_nchw(x_copy)
if fuse_add:
    out_nchw += z_copy
if fuse_relu:
    out_nchw = out_nchw.relu()

# Run NCHW bwd
out_nchw.backward(g0_nchw)

# Permute NHWC results -> NCHW for comparison
out_nhwc = out.permute(0,3,1,2)
x_grad_nhwc = x.grad.permute(0,3,1,2)
if fuse_add:
    z_grad_nhwc = z.grad.permute(0,3,1,2)

atol = 1e-5
rtol = 1e-3
print('X: ', torch.allclose(out_nhwc, out_nchw, atol=atol, rtol=rtol))
print('dS: ', torch.allclose(bn.weight.grad, bn_nchw.weight.grad, atol=atol, rtol=rtol))
print('dB: ', torch.allclose(bn.bias.grad, bn_nchw.bias.grad, atol=atol, rtol=rtol))
print('dX: ', torch.allclose(x_grad_nhwc, x_copy.grad, atol=atol, rtol=rtol))
if fuse_add:
    print('dZ: ', torch.allclose(z_grad_nhwc, z_copy.grad, atol=atol, rtol=rtol))
