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

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import Conv2d_NHWC
from batch_norm import BatchNorm2d_NHWC
from max_pool import MaxPool2d_NHWC

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_NHWC(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d_NHWC(planes, fuse_relu=True)
        self.conv2 = Conv2d_NHWC(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d_NHWC(planes, fuse_relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_NHWC(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                BatchNorm2d_NHWC(self.expansion*planes, fuse_relu=False)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_NHWC(in_planes, planes, kernel_size=1)
        self.bn1 = BatchNorm2d_NHWC(planes, fuse_relu=True)
        self.conv2 = Conv2d_NHWC(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm2d_NHWC(planes, fuse_relu=True)
        self.conv3 = Conv2d_NHWC(planes, self.expansion*planes, kernel_size=1)
        self.bn3 = BatchNorm2d_NHWC(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_NHWC(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                BatchNorm2d_NHWC(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_NHWC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_NHWC, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d_NHWC(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d_NHWC(64, fuse_relu=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

	# Move back to NCHW for final parts
        out = out.permute(0, 3, 1, 2).contiguous()

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_NHWC():
    return ResNet_NHWC(BasicBlock, [2,2,2,2])

def ResNet34_NHWC():
    return ResNet_NHWC(BasicBlock, [3,4,6,3])

def ResNet50_NHWC():
    return ResNet_NHWC(Bottleneck, [3,4,6,3])

def ResNet101_NHWC():
    return ResNet_NHWC(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

