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

import torch
from torch.autograd import Variable
from SSD import _C as C

from train import dboxes300_coco
from utils import Encoder

import numpy as np

import random
import time

b1 = np.array([[ 0.6994,  0.4620,  0.9620,  0.7748],
      [ 0.4149,  0.6118,  0.4581,  0.7141],
      [ 0.8627,  0.6390,  0.8968,  0.6859],
      [ 0.4626,  0.6786,  0.7066,  0.8359],
      [ 0.0583,  0.7838,  0.1761,  0.9167],
      [ 0.0197,  0.7093,  0.1063,  0.7821],
      [ 0.7446,  0.4397,  0.7807,  0.5353],
      [ 0.5805,  0.6577,  0.6299,  0.6958],
      [ 0.6137,  0.6411,  0.6635,  0.6889],
      [ 0.0006,  0.7387,  0.1155,  0.8454],
      [ 0.9154,  0.6114,  1.0000,  0.7110],
      [ 0.7788,  0.4582,  0.8123,  0.4984],
      [ 0.3279,  0.0832,  0.6601,  0.5236],
      [ 0.0408,  0.0898,  0.3308,  0.5199],
      [ 0.6602,  0.1074,  0.9500,  0.4404],
      [ 0.2352,  0.8385,  0.4033,  0.9892],
      [ 0.0000,  0.7924,  0.0718,  0.8829]])

b1b = np.array([[ 0.6994,  0.4620,  0.9620,  0.7748],
               [ 0.4149,  0.6118,  0.4581,  0.7141],
               [ 0.8627,  0.6390,  0.8968,  0.6859],
               [ 0.4626,  0.6786,  0.7066,  0.8359],
               [ 0.0583,  0.7838,  0.1761,  0.9167],
               [ 0.0197,  0.7093,  0.1063,  0.7821],
               [ 0.7446,  0.4397,  0.7807,  0.5353],
               [ 0.5805,  0.6577,  0.6299,  0.6958],
               [ 0.6137,  0.6411,  0.6635,  0.6889],
               [ 0.0006,  0.7387,  0.1155,  0.8454],
               [ 0.9154,  0.6114,  1.0000,  0.7110],
               [ 0.7788,  0.4582,  0.8123,  0.4984],
               [ 0.3279,  0.0832,  0.6601,  0.5236],
               [ 0.0408,  0.0898,  0.3308,  0.5199],
               [ 0.6602,  0.1074,  0.9500,  0.4404],
               [ 0.2352,  0.8385,  0.4033,  0.9892],
               [ 0.0000,  0.7924,  0.0718,  0.8829]])

b2 = np.array([[ 0.2166,  0.0987,  0.6985,  0.9252],
      [ 0.0078,  0.5031,  0.7645,  0.9830]])

def load_bboxes(box_list, random_rows=True):
    tensor_list = []

    for b in box_list:
        if random_rows:
            n_rows = random.randint(1, b.shape[0])
            t = torch.tensor(np.array(b[0:n_rows, :]).astype(np.float32)).cuda()
        else:
            t = torch.tensor(np.array(b).astype(np.float32)).cuda()
        tensor_list.append(t)

    return generate_bbox_stacks(tensor_list)

# bboxes is a list of tensors of size [N_i, 4]
# outputs N_boxes, stacked boxes, offsets into box_i, original boxes
def generate_bbox_stacks(bboxes):
    offsets = [0]

    for bbox in bboxes:
        offsets.append(bbox.shape[0] + offsets[-1])

    offsets = torch.tensor(np.array(offsets).astype(np.int32)).cuda()

    return len(bboxes), torch.cat(bboxes), offsets, bboxes

def calc_iou_tensor(box1, box2):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    #mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    #mask1 = ~mask1
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])
    #mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    #mask2 = ~mask2

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]
    #*mask1.float()*mask2.float()

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

def test_iou_calc():
    torch.cuda.device(0)

    # source boxes
    N, bboxes_cat, offsets, bboxes = load_bboxes([b1, b1b])

    # target boxes
    _, _, _, targets = load_bboxes([b2])

    ious = C.calc_ious(N, bboxes_cat, offsets, *targets)
    # print('cuda:\n', ious)

    ref_ious = []
    for source in bboxes:
        ref_iou = calc_iou_tensor(source.cpu(), targets[0].cpu())
        ref_ious.append(ref_iou)

    # print('ref:\n ')
    ref_cat = []
    for ref in ref_ious:
        ref_cat.append(ref)

    ref_cat = torch.cat(ref_cat)

    print('calc_iou pass: {}'.format(np.allclose(ref_cat.numpy(), ious.cpu().numpy())))

def test_box_encoder():
    torch.cuda.device(0)

    # np.random.seed(0)

    # source boxes
    box_list = []
    for _ in range(128):
        box_list.append(b1)
    N, bboxes_cat, offsets, bboxes = load_bboxes(box_list, True)
    # N, bboxes_cat, offsets, bboxes = load_bboxes([b1[:2,:], b1[:2,:]])

    print(N, bboxes_cat, offsets)

    label_numpy = np.random.randn(offsets[-1])*10
    labels = torch.tensor(label_numpy.astype(np.int64)).cuda()

    # target boxes are default boxes from SSD
    dboxes = dboxes300_coco()
    dboxes = torch.tensor(np.array(dboxes(order='ltrb')).astype(np.float32))

    # print(dboxes[:10, :])

    start = time.time()
    bbox_out, label_out = C.box_encoder(N, bboxes_cat, offsets, labels, dboxes.cuda(), 0.5)
    torch.cuda.synchronize()
    end = time.time()

    cuda_time = end - start

    # print('bbox_out: {}'.format(bbox_out.shape))
    # print(bbox_out.cpu())

    # print('label_out: {}'.format(label_out.shape))
    # print(label_out.cpu())

    # reference
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    labels_ref = torch.tensor(label_numpy.astype(np.int64))
    start = time.time()

    ref_boxes = []
    ref_labels = []
    for i, bbox in enumerate(bboxes):
        label_slice = labels_ref[offsets[i]:offsets[i+1]]
        bbox_ref_out, label_ref_out = encoder.encode(bbox.cpu(), label_slice.cpu(), criteria = 0.5)
        ref_boxes.append(bbox_ref_out)
        ref_labels.append(label_ref_out)
    end = time.time()
    ref_time = end - start

    ref_boxes = torch.cat(ref_boxes)
    ref_labels = torch.cat(ref_labels)

    # print('ref bbox: {}'.format(ref_boxes.shape))
    # print(bbox_ref_out)

    r = np.isclose(ref_boxes.numpy(), bbox_out.cpu().numpy())
    # r = np.isclose(bbox_ref_out.numpy(), bbox_out.cpu().numpy())

    num_fail = 0
    for i, res in enumerate(r):
        if not res.any():
            num_fail += 1
            print(i, res, ref_boxes[i,:], bbox_out[i, :])

    print('{} bboxes failed'.format(num_fail))

    label_out = label_out.cpu().numpy()
    torch.cuda.synchronize()
    # r2 = np.isclose(label_out, label_ref_out.cpu().numpy())
    r2 = np.isclose(label_out, ref_labels.cpu().numpy())
    num_fail = 0
    for i, res in enumerate(r2):
        if not res:
            num_fail += 1
            print('label: ', i, res, label_out[i], ref_labels.numpy()[i])

    print('{} labels failed'.format(num_fail))

    print('CUDA took {}, numpy took: {}'.format(cuda_time, ref_time))


if __name__ == "__main__":
    test_iou_calc()

    test_box_encoder()

