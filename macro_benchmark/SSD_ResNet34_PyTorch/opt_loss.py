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

class OptLoss(torch.jit.ScriptModule):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    __constants__ = ['scale_xy', 'scale_wh', 'dboxes_xy', 'dboxes_wh']

    def __init__(self, dboxes):
        super(OptLoss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = torch.nn.SmoothL1Loss(reduce=False)
        self.dboxes = torch.nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        self.register_buffer('dboxes_xy', self.dboxes[:, :2, :])
        self.register_buffer('dboxes_wh', self.dboxes[:, 2:, :])
        self.dboxes_xy.requires_grad = False
        self.dboxes_wh.requires_grad = False
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = torch.nn.CrossEntropyLoss(reduce=False)

    @torch.jit.script_method
    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        # loc_wh, loc_xy = loc[:, :2, :], loc[:, 2:, :]
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes_xy) / self.dboxes_wh
        # gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :] / self.dboxes_wh).log()
        # gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        #print(gxy.sum(), gwh.sum())
        return torch.cat((gxy, gwh), dim=1).contiguous()

    @torch.jit.script_method
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
        sl1 = (mask.type_as(sl1) * sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        # con_neg[mask] = 0
        con_neg.masked_fill_(mask, 0)
        # con_neg[con_neg!=con_neg] = 0
        con_neg.masked_fill_(con_neg!=con_neg, 0)
        con_s, con_idx = con_neg.sort(dim=1, descending=True)
        r = torch.arange(0, con_neg.size(1), dtype=torch.long, device='cuda').expand(con_neg.size(0), -1)
        con_rank = r.scatter(1, con_idx, r)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        closs = (con*(mask.type_as(con_s) + neg_mask.type_as(con_s))).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).type_as(closs)
        pos_num = pos_num.type_as(closs).clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret
