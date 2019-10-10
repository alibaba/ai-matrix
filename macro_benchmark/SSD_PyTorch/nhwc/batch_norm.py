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
from torch.nn.modules.batchnorm import _BatchNorm

import SSD._C as C

import collections
from itertools import repeat

class bn_NHWC_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, b, rm, riv, mom, epsilon, fuse_relu=False, is_train=True, z=None):
        if is_train:
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.fuse_relu = fuse_relu

            ctx.fuse_add = False if z is None else True

            if z is not None:
                y, save_mean, save_var, reserve = C.bn_add_fwd_nhwc_cudnn(x, z, s, b, rm, riv, mom, epsilon, fuse_relu)
            else:
                y, save_mean, save_var, reserve = C.bn_fwd_nhwc_cudnn(x, s, b, rm, riv, mom, epsilon, fuse_relu)

            ctx.save_for_backward(x, y, s, b, rm, riv, save_mean, save_var, reserve)

            return y
        else:
            if z is not None:
                return C.bn_add_fwd_eval_nhwc_cudnn(x, z, s, b, rm, riv, mom, epsilon, fuse_relu)
            else:
                return C.bn_fwd_eval_nhwc_cudnn(x, s, b, rm, riv, mom, epsilon, fuse_relu)

    @staticmethod
    def backward(ctx, grad_y):
        x, y, s, b, rm, riv, save_mean, save_var, reserve = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        fuse_relu = ctx.fuse_relu
        fuse_add = ctx.fuse_add

        if ctx.fuse_add:
            dx, dz, dscale, dbias = C.bn_add_bwd_nhwc_cudnn(x, y, grad_y, s, b, rm, riv, save_mean, save_var, reserve, mom, epsilon, fuse_relu)
        else:
            dx, _, dscale, dbias = C.bn_bwd_nhwc_cudnn(x, y, grad_y, s, b, rm, riv, save_mean, save_var, reserve, mom, epsilon, fuse_relu)
            dz = None

        return dx, dscale, dbias, None, None, None, None, None, None, dz



class BatchNorm2d_NHWC(_BatchNorm):
    def __init__(self, num_features, fuse_relu=False):
        super(BatchNorm2d_NHWC, self).__init__(num_features)

        self.fuse_relu = fuse_relu

    def forward(self, x, z=None):
        return bn_NHWC_impl.apply(x,
                                  self.weight, self.bias,
                                  self.running_mean, self.running_var,
                                  self.momentum,
                                  self.eps, self.fuse_relu, self.training, z)

