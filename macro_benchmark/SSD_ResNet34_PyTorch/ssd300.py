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
import torch.nn as nn
from base_model import L2Norm, ResNet

from nhwc.conv import Conv2d_NHWC

class SSD300(nn.Module):
    """
        Build a SSD module to take 300x300 image input,
        and output 8732 per class bounding boxes

        vggt: pretrained vgg16 (partial) model
        label_num: number of classes (including background 0)
    """
    def __init__(self, label_num, backbone='resnet34', use_nhwc=False, pad_input=False, bn_group=1):

        super(SSD300, self).__init__()

        self.label_num = label_num
        self.use_nhwc = use_nhwc
        self.pad_input = pad_input
        self.bn_group = bn_group

        if backbone == 'resnet18':
            out_channels = 256
            out_size = 38
            self.out_chan = [out_channels, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            out_channels = 256
            out_size = 38
            self.out_chan = [out_channels, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            out_channels = 1024
            out_size = 38
            self.l2norm4 = L2Norm()
            self.out_chan = [out_channels, 1024, 512, 512, 256, 256]
        else:
            print('Invalid backbone chosen')

        self.model = ResNet(backbone, self.use_nhwc, self.pad_input, self.bn_group)

        self._build_additional_features(out_size, self.out_chan)

        # after l2norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2
        # classifer 1, 2, 3, 4, 5 ,6

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.mbox = []
        self.padding_amounts = []

        if self.use_nhwc:
            conv_fn = Conv2d_NHWC
        else:
            conv_fn = nn.Conv2d
        for nd, oc in zip(self.num_defaults, self.out_chan):
            # Horizontally fuse loc and conf convolutions
            my_num_channels = nd*(4+self.label_num)
            if self.use_nhwc:
                # Want to manually pad to get HMMA kernels in NHWC case
                padding_amount = 8 - (my_num_channels % 8)
            else:
                padding_amount = 0
            self.padding_amounts.append(padding_amount)
            self.mbox.append(conv_fn(oc, my_num_channels + padding_amount, kernel_size=3, padding=1))

        self.mbox = nn.ModuleList(self.mbox)

        # intitalize all weights
        with torch.no_grad():
            self._init_weights()

    def _build_additional_features(self, input_size, input_channels):
        idx = 0
        if input_size == 38:
            idx = 0
        elif input_size == 19:
            idx = 1
        elif input_size == 10:
            idx = 2

        self.additional_blocks = []

        if self.use_nhwc:
            conv_fn = Conv2d_NHWC
        else:
            conv_fn = nn.Conv2d

        #
        if input_size == 38:
            self.additional_blocks.append(nn.Sequential(
                conv_fn(input_channels[idx], 256, kernel_size=1),
                nn.ReLU(inplace=True),
                conv_fn(256, input_channels[idx+1], kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ))
            idx += 1

        self.additional_blocks.append(nn.Sequential(
            conv_fn(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_fn(256, input_channels[idx+1], kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv9_1, conv9_2
        self.additional_blocks.append(nn.Sequential(
            conv_fn(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_fn(128, input_channels[idx+1], kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv10_1, conv10_2
        self.additional_blocks.append(nn.Sequential(
            conv_fn(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_fn(128, input_channels[idx+1], kernel_size=3),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # Only necessary in VGG for now
        if input_size >= 19:
            # conv11_1, conv11_2
            self.additional_blocks.append(nn.Sequential(
                conv_fn(input_channels[idx], 128, kernel_size=1),
                nn.ReLU(inplace=True),
                conv_fn(128, input_channels[idx+1], kernel_size=3),
                nn.ReLU(inplace=True),
            ))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        addn_blocks = [
            *self.additional_blocks]
        layers = [ *self.mbox ]
            # *self.loc, *self.conf]

        # Need to handle additional blocks differently in NHWC case due to xavier initialization
        for layer in addn_blocks:
            for param in layer.parameters():
                if param.dim() > 1:
                    if self.use_nhwc:
                        # xavier_uniform relies on fan-in/-out, so need to use NCHW here to get
                        # correct values (K, R) instead of the correct (K, C)
                        nchw_param_data = param.data.permute(0, 3, 1, 2).contiguous()
                        nn.init.xavier_uniform_(nchw_param_data)
                        # Now permute correctly-initialized param back to NHWC
                        param.data.copy_(nchw_param_data.permute(0, 2, 3, 1).contiguous())
                    else:
                        nn.init.xavier_uniform_(param)

        for layer, default, padding in zip(layers, self.num_defaults, self.padding_amounts):
            for param in layer.parameters():
                if param.dim() > 1 and self.use_nhwc:
                    # Need to be careful - we're initialising [loc, conf, pad] with
                    # all 3 needing to be treated separately
                    conf_channels = default * self.label_num
                    loc_channels  = default * 4
                    pad_channels  = padding
                    # Split the parameter into separate parts along K dimension
                    conf, loc, pad = param.data.split([conf_channels, loc_channels, pad_channels], dim=0)

                    # Padding should be zero
                    pad_data = torch.zeros_like(pad.data)

                    def init_loc_conf(p):
                        p_data = p.data.permute(0, 3, 1, 2).contiguous()
                        nn.init.xavier_uniform_(p_data)
                        p_data = p_data.permute(0, 2, 3, 1).contiguous()
                        return p_data

                    # Location and confidence data
                    loc_data = init_loc_conf(loc)
                    conf_data = init_loc_conf(conf)

                    # Put the full weight together again along K and copy
                    param.data.copy_(torch.cat([conf_data, loc_data, pad_data], dim=0))
                elif param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, mbox):
        locs = []
        confs = []
        for s, m, num_defaults, pad in zip(src, mbox, self.num_defaults, self.padding_amounts):
            mm = m(s)
            conf_channels = num_defaults * self.label_num
            loc_channels  = num_defaults * 4

            if self.use_nhwc:
                conf, loc, _ = mm.split([conf_channels, loc_channels, pad], dim=3)
                conf, loc = conf.contiguous(), loc.contiguous()
                # We now have unfused [N, H, W, C]
                # Layout is a little awkward here.
                # Take C = c * d, then we actually have:
                # [N, H, W, c*d]
                # flatten HW first:
                #   [N, H, W, c*d] -> [N, HW, c*d]
                locs.append(
                    loc.view(s.size(0), -1, 4 * num_defaults).permute(0, 2, 1).contiguous().view(loc.size(0), 4, -1))
                confs.append(
                    conf.view(s.size(0), -1, self.label_num * num_defaults).permute(0, 2, 1).contiguous().view(conf.size(0), self.label_num, -1))
            else:
                conf, loc = mm.split([conf_channels, loc_channels], dim=1)
                conf, loc = conf.contiguous(), loc.contiguous()
                # flatten the anchors for this layer
                locs.append(loc.view(s.size(0), 4, -1))
                confs.append(conf.view(s.size(0), self.label_num, -1))

        cat_dim = 2 # 1 if self.use_nhwc else 2
        locs, confs = torch.cat(locs, cat_dim), torch.cat(confs, cat_dim)
        if False: # self.use_nhwc:
            # Input here is [N, HW, sum(c*d)], pemute to [N, sum(c*d), HW]
            locs = locs.permute(0, 2, 1).contiguous()
            confs = confs.permute(0, 2, 1).contiguous()
        return locs, confs

    def forward(self, data):

        layers = self.model(data)

        # last result from network goes into additional blocks
        x = layers[-1]
        # If necessary, transpose back to NCHW
        additional_results = []
        for i, l in enumerate(self.additional_blocks):
            x = l(x)
            additional_results.append(x)

        # do we need the l2norm on the first result?
        src = [*layers, *additional_results]
        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4

        locs, confs = self.bbox_view(src, self.mbox)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs

