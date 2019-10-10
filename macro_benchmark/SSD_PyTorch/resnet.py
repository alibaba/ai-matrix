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

import torch                    # for torch.cat and torch.zeros
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from nhwc.conv import Conv2d_NHWC
from nhwc.batch_norm import BatchNorm2d_NHWC
from nhwc.max_pool import MaxPool2d_NHWC

# Group batch norm
from apex.parallel import SyncBatchNorm as gbn
# Persistent group BN for NHWC case
from apex.contrib.groupbn.batch_norm import BatchNorm2d_NHWC as gbn_persistent
import apex.parallel

__all__ = ['resnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class MyLayers:
    Conv2d = nn.Conv2d
    MaxPool = nn.MaxPool2d

    def __init__(self, bn_group, **kwargs):
        super(MyLayers, self).__init__()
        self.nhwc = False
        self.bn_group = bn_group

    def build_bn(self, planes, fuse_relu=False):
        return self.BnAddRelu(planes, fuse_relu, self.bn_group)

    class BnAddRelu(gbn):
        def __init__(self, planes, fuse_relu=False, bn_group=1):
            super(MyLayers.BnAddRelu, self).__init__(
                planes,
                process_group=apex.parallel.create_syncbn_process_group(bn_group))
            self.fuse_relu_flag = fuse_relu

        def forward(self, x, z=None):
            out = super().forward(x)
            if z is not None:
                out = out + z
            if self.fuse_relu_flag:
                out = out.relu_()
            return out


class MyLayers_NHWC:
    Conv2d = Conv2d_NHWC
    MaxPool = MaxPool2d_NHWC

    class BnAddRelu(gbn_persistent):
        def __init__(self, planes, fuse_relu=False, bn_group=1):
            super(MyLayers_NHWC.BnAddRelu, self).__init__(planes,
                                                          fuse_relu,
                                                          bn_group=bn_group)

    def __init__(self, bn_group, **kwargs):
        super(MyLayers_NHWC, self).__init__()
        self.nhwc = True
        self.bn_group = bn_group

    def build_bn(self, planes, fuse_relu):
        return self.BnAddRelu(planes, fuse_relu, self.bn_group)



def conv1x1(layer_types, in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer_types.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                           bias=False)

def conv3x3(layer_types, in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return layer_types.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, layerImpls, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(layerImpls, inplanes, planes, stride=stride)
        self.bn1 = layerImpls.build_bn(planes, fuse_relu=True)
        self.conv2 = conv3x3(layerImpls, planes, planes)
        self.bn2 = layerImpls.build_bn(planes, fuse_relu=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out, residual)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, layerImpls, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(layerImpls, inplanes, planes)
        self.bn1 = layerImpls.build_bn(planes, fuse_relu=True)
        self.conv2 = conv3x3(layerImpls, planes, planes, stride=stride)
        self.bn2 = layerImpls.build_bn(planes, fuse_relu=True)
        self.conv3 = conv1x1(layerImpls, planes, planes * self.expansion)
        self.bn3 = layerImpls.build_bn(planes * self.expansion, fuse_relu=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out, residual)

        return out

class Classifier(nn.Module):
    def __init__(self, block_expansion, num_classes=1000, use_nhwc=False):
        super(Classifier, self).__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block_expansion, num_classes)
        self.is_nhwc=use_nhwc

    def forward(self, x):
        if self.is_nhwc:
            # Permute back to NCHW for AvgPool and the rest
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet(nn.Module):

    def __init__(self, layerImpls, block, layers, num_classes=1000,
                 pad_input=False, ssd_mods=False, use_nhwc=False,
                 bn_group=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if pad_input:
            input_channels = 4
        else:
            input_channels = 3
        self.conv1 = layerImpls.Conv2d(input_channels, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
        self.bn1 = layerImpls.build_bn(64, fuse_relu=True)
        self.maxpool = layerImpls.MaxPool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(layerImpls, block, 64, layers[0])
        self.layer2 = self._make_layer(layerImpls, block, 128, layers[1], stride=2)
        if ssd_mods:
            self.layer3 = self._make_layer(layerImpls, block, 256, layers[2], stride=1)
        else:
            self.layer3 = self._make_layer(layerImpls, block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(layerImpls, block, 512, layers[3], stride=2)

            self.classifier = Classifier(block.expansion, num_classes=1000, use_nhwc=use_nhwc)

        # FIXME! This (a) fails for nhwc, and (b) is irrelevant if the user is
        # also loading pretrained data (which we don't know about here, but
        # know about in the caller (the "resnet()" function below).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, layerImpls, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layerImpls.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                layerImpls.build_bn(planes * block.expansion, fuse_relu=False),
            )

        layers = []
        layers.append(block(layerImpls, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layerImpls, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x

def _transpose_state(state, pad_input=False):
    for k in state.keys():
        if len(state[k].shape) == 4:
            if pad_input and "conv1.weight" in k and not 'layer' in k:
                s = state[k].shape
                state[k] = torch.cat([state[k], torch.zeros([s[0], 1, s[2], s[3]])], dim=1)
            state[k] = state[k].permute(0, 2, 3, 1).contiguous()
    return state

model_blocks = {
    'resnet18': BasicBlock,
    'resnet34': BasicBlock,
    'resnet50': Bottleneck,
    'resnet101': Bottleneck,
    'resnet152': Bottleneck,
}

model_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}

def resnet(model_name, pretrained=False, nhwc=False, ssd_mods=False, **kwargs):
    """Constructs a ResNet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if nhwc:
        layerImpls = MyLayers_NHWC(**kwargs)
    else:
        layerImpls = MyLayers(**kwargs)

    block = model_blocks[model_name]
    layer_list = model_layers[model_name]
    model = ResNet(layerImpls, block, layer_list, ssd_mods=ssd_mods, use_nhwc=nhwc, **kwargs)
    if pretrained:
        orig_state_dict = model_zoo.load_url(model_urls[model_name])
        if ssd_mods:
            # the ssd model doesn't have layer4
            state_dict = {k:orig_state_dict[k] for k in orig_state_dict if (not k.startswith('layer4') and not k.startswith('fc'))}
        else:
            state_dict = orig_state_dict
        pad_input = kwargs.get('pad_input', False)
        if nhwc:
            state_dict = _transpose_state(state_dict, pad_input)

        model.load_state_dict(state_dict)
    if ssd_mods:
        return nn.Sequential(model.conv1, model.bn1, model.maxpool, model.layer1, model.layer2, model.layer3)
    else:
        return model
