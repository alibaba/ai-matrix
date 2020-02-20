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
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin

from maskrcnn_benchmark import NHWC
from apex import amp

from torch.autograd.function import once_differentiable
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
            return NHWC.cudnn_convolution_nhwc(x, w,
                                            padding, stride, dilation,
                                            groups,
                                            torch.backends.cudnn.benchmark, False)
        else:
            return NHWC.cudnn_convolution_with_bias_nhwc(x, w, bias,
                                            padding, stride, dilation,
                                            groups,
                                            torch.backends.cudnn.benchmark, False)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        x, w = ctx.saved_variables
        if ctx.need_bias_grad:
            dx, dw, db = NHWC.cudnn_convolution_backward_with_bias_nhwc(x, grad_y, w,
                                                       ctx.padding, ctx.stride, ctx.dilation, ctx.groups,
                                                       torch.backends.cudnn.benchmark, False,
                                                       list(ctx.needs_input_grad[0:3]))
            if ctx.needs_input_grad[0]:
                return dx, dw, db, None, None, None, None
            else:
                return None, dw, db, None, None, None, None
        else:
            if (not ctx.needs_input_grad[1] ):
                return None, None, None, None, None, None, None  
            dx, dw = NHWC.cudnn_convolution_backward_nhwc(x, grad_y, w,
                                                       ctx.padding, ctx.stride, ctx.dilation, ctx.groups,
                                                       torch.backends.cudnn.benchmark, False,
                                                       list(ctx.needs_input_grad[0:2]))
            if (not ctx.needs_input_grad[1]):
                return None, None, None, None, None, None, None  
            elif ctx.needs_input_grad[0]:
                return dx, dw, None, None, None, None, None
            else:
                return None, dw, None, None, None, None, None


class conv2d_transpose_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, bias=None, stride=(1,1), padding=(0,0), output_padding=(0,0), dilation=(1,1), groups=1):
        # Save constants for bprop
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.need_bias_grad = bias is not None
        ctx.save_for_backward(x, w)
        if bias is None:
            return NHWC.cudnn_convolution_transpose_nhwc(x, w,
                                            padding, output_padding, stride, dilation,
                                            groups,
                                            torch.backends.cudnn.benchmark, False)
        else:
            return NHWC.cudnn_convolution_transpose_with_bias_nhwc(x, w, bias,
                                            padding, output_padding, stride, dilation,
                                            groups,
                                            torch.backends.cudnn.benchmark, False)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        x, w = ctx.saved_variables
        if ctx.need_bias_grad:
            dx, dw, db = NHWC.cudnn_convolution_transpose_backward_with_bias_nhwc(x, grad_y, w,
                                                       ctx.padding, ctx.output_padding, ctx.stride, ctx.dilation, ctx.groups,
                                                       torch.backends.cudnn.benchmark, False,
                                                       list(ctx.needs_input_grad[0:3]))
            if ctx.needs_input_grad[0]:
                return dx, dw, db, None, None, None, None, None
            else:
                return None, dw, db, None, None, None, None, None
        else:
            if (not ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]):
                return None, None, None, None, None, None, None  
            dx, dw = NHWC.cudnn_convolution_transpose_backward_nhwc(x, grad_y, w,
                                                       ctx.padding, ctx.output_padding, ctx.stride, ctx.dilation, ctx.groups,
                                                       torch.backends.cudnn.benchmark, False,
                                                       list(ctx.needs_input_grad[0:2]))
            if (not ctx.needs_input_grad[1]):
                return None, None, None, None, None, None, None  
            elif ctx.needs_input_grad[0]:
                return dx, dw, None, None, None, None, None
            else:
                return None, dw, None, None, None, None, None

amp.register_half_function(conv2d_NHWC_impl,'apply')
amp.register_half_function(conv2d_transpose_NHWC_impl,'apply')

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

class ConvTranspose2d_NHWC(_ConvTransposeMixin, _ConvNd):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d_NHWC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)
        # permute filters
        self.weight = torch.nn.Parameter(self.weight.permute(0, 2, 3, 1).contiguous())

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return conv2d_transpose_NHWC_impl.apply(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.dilation, self.groups)
