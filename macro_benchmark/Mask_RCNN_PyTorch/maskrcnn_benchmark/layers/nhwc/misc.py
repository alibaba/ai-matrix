# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

"""
helper class that supports empty tensors on some nhwc functions
"""

import math
import torch
from torch.nn.modules.utils import _ntuple
from maskrcnn_benchmark.layers.nhwc import conv
from maskrcnn_benchmark.layers.nhwc import transforms


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class Conv2d_NHWC(conv.Conv2d_NHWC):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d_NHWC, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class ConvTranspose2d_NHWC(conv.ConvTranspose2d_NHWC):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d_NHWC, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

def nhwc_to_nchw_transform(x):
    if x.numel() == 0:
        return x
    op = transforms.NHWCToNCHW()
    return op(x)

def nchw_to_nhwc_transform(x):
    if x.numel() == 0:
        return x
    op = transforms.NCHWToNHWC()
    return op(x)
