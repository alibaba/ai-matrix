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
from torch.nn.modules.conv import _ConvNd

import SSD._C as C

import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class conv2d_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1):
        # Save constants for bprop
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.need_bias_grad = bias is not None
        ctx.save_for_backward(x, w)

        if bias is None:
            return C.cudnn_convolution_nhwc(x, w,
                                            padding, stride, dilation,
                                            groups,
                                            torch.backends.cudnn.benchmark, False)
        else:
            return C.cudnn_convolution_with_bias_nhwc(x, w, bias,
                                            padding, stride, dilation,
                                            groups,
                                            torch.backends.cudnn.benchmark, False)

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_variables

        if ctx.need_bias_grad:
            dx, dw, db = C.cudnn_convolution_backward_with_bias_nhwc(x, grad_y, w,
                                                       ctx.padding, ctx.stride, ctx.dilation, ctx.groups,
                                                       True, False,
                                                       list(ctx.needs_input_grad[0:3]))
            if ctx.needs_input_grad[0]:
                return dx, dw, db, None, None, None, None
            else:
                return None, dw, db, None, None, None, None
        else:
            dx, dw = C.cudnn_convolution_backward_nhwc(x, grad_y, w,
                                                       ctx.padding, ctx.stride, ctx.dilation, ctx.groups,
                                                       True, False,
                                                       list(ctx.needs_input_grad[0:2]))
            if ctx.needs_input_grad[0]:
                return dx, dw, None, None, None, None, None
            else:
                return None, dw, None, None, None, None, None

class Conv2d_NHWC(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_NHWC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias=bias, padding_mode='zeros')

        # permute filters
        self.weight = torch.nn.Parameter(self.weight.permute(0, 2, 3, 1).contiguous())

    def forward(self, x):
        return conv2d_NHWC_impl.apply(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

