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

import torch.nn as nn
import torch

from resnet import resnet

class ResNet(nn.Module):
    def __init__(self, resnet_base, use_nhwc=False, pad_input=False, bn_group=1):
        super().__init__()

        rn_opts = {
            'nhwc' : use_nhwc,
            'pretrained' : True,
            'ssd_mods' : True,
            'pad_input' : pad_input,
            'bn_group' : bn_group,
        }
        self.rn = resnet(resnet_base, **rn_opts)

    def forward(self, data):
        # package the data in a list because for some varieties of SSD we
        # actually want to take the 38x38 from somewhere in the middle of the
        # stack, rather than just the final output
        return [self.rn(data)]

class L2Norm(nn.Module):
    """
       Scale shall be learnable according to original paper
       scale: initial scale number
       chan_num: L2Norm channel number (norm over all channels)
    """
    def __init__(self, scale=20, chan_num=512):
        super(L2Norm, self).__init__()
        # Scale across channels
        self.scale = \
            nn.Parameter(torch.Tensor([scale]*chan_num).view(1, chan_num, 1, 1))

    def forward(self, data):
        # normalize accross channel
        return self.scale*data*data.pow(2).sum(dim=1, keepdim=True).clamp(min=1e-12).rsqrt()

class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        #print(gxy.sum(), gwh.sum())
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """

        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        con_neg[con_neg!=con_neg] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*(mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        #print((sl1*num_mask/pos_num).mean().item(), \
        #     ((con*mask.float()).sum(dim=1)*num_mask/pos_num).mean().item(), \
        #     ((con*neg_mask.float()).sum(dim=1)*num_mask/pos_num).mean().item(), )
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        #del mask, vec_gd, sl1, con, con_neg, con_idx, con_rank, neg_num, neg_mask, closs, total_loss, num_mask, pos_num
        return ret
