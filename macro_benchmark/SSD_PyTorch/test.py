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

import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import os

# necessary pytorch imports
import torch.utils.data.distributed
import torch.distributed as dist
from torch.autograd import Variable

# Apex imports
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

# DALI import
from coco_pipeline import COCOPipeline, DALICOCOIterator

from SSD import _C as C


def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco/coco2017',
                        help='path to test and training data files')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='number of examples for each iteration')
    #parser.add_argument('--checkpoint', type=str, default=None,
    #                    help='path to model checkpoint file', required=True)
    parser.add_argument('--backbone', type=str, choices=['vgg16', 'vgg16bn',
                        'resnet18', 'resnet34', 'resnet50'], default='resnet34')
    parser.add_argument('--num-workers', type=int, default=3)
    parser.add_argument('--fbu', type=int, default=1)
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--use-train-dataset', action='store_true')

    # Distributed stuff
    parser.add_argument('--local_rank', default=0, type=int,
			help='Used for multi-process training. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')

    return parser.parse_args()

def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def test_coco(args):
    # For testing purposes we have to use CUDA
    use_cuda = True

    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)

        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    if args.use_train_dataset:
        annotate = os.path.join(args.data, "annotations/instances_train2017.json")
        coco_root = os.path.join(args.data, "train2017")
        img_number = 118287
    else:
        annotate = os.path.join(args.data, "annotations/instances_val2017.json")
        coco_root = os.path.join(args.data, "val2017")
        img_number = 5000

    pipe = COCOPipeline(args.batch_size, args.local_rank, coco_root,
                    annotate, N_gpu, num_threads=args.num_workers)
    pipe.build()
    test_run = pipe.run()
    dataloader = DALICOCOIterator(pipe, img_number / N_gpu)

    # Build the model
    ssd300 = SSD300(81, backbone=args.backbone, model_path='', dilation=False)

    """
    # Note: args.checkpoint is required, so this can never be false
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)

        # remove proceeding 'module' from checkpoint
        model = od["model"]
        for k in list(model.keys()):
            if k.startswith('module.'):
                model[k[7:]] = model.pop(k)
        ssd300.load_state_dict(model)
    """


    ssd300.cuda()
    ssd300.eval()
    loss_func = Loss(dboxes)
    loss_func.cuda()

    # parallelize
    if args.distributed:
        ssd300 = DDP(ssd300)

    if args.use_fp16:
        ssd300 = network_to_half(ssd300)

    if args.use_train_dataset and args.local_rank == 0:
        print('Image 000000320612.jpg is in fact PNG and it will cause fail if ' +
                'used with nvJPEGDecoder in coco_pipeline')

    for epoch in range(2):
        if epoch == 1 and args.local_rank == 0:
            print("Performance computation starts")
            s = time.time()
        for i, data in enumerate(dataloader):

            with torch.no_grad():
                # Get data from pipeline
                img = data[0][0][0]
                bbox = data[0][1][0]
                label = data[0][2][0]
                label = label.type(torch.cuda.LongTensor)
                bbox_offsets = data[0][3][0]
                bbox_offsets = bbox_offsets.cuda()

                # Encode labels
                N = img.shape[0]
                if bbox_offsets[-1].item() == 0:
                    print("No labels in batch")
                    continue
                bbox, label = C.box_encoder(N, bbox, bbox_offsets, label,
                                            encoder.dboxes.cuda(), 0.5)

                # Prepare tensors for computing loss
                M = bbox.shape[0] // N
                bbox = bbox.view(N, M, 4)
                label = label.view(N, M)
                trans_bbox = bbox.transpose(1,2).contiguous()
                gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                               Variable(label, requires_grad=False)

                if args.use_fp16:
                    img = img.half()

                for _ in range(args.fbu):
                    ploc, plabel = ssd300(img)
                    ploc, plabel = ploc.float(), plabel.float()
                    loss = loss_func(ploc, plabel, gloc, glabel)

        if epoch == 1 and args.local_rank == 0:
            e = time.time()
            print("Performance achieved: {:.2f} img/sec".format(img_number / (e - s)))

        dataloader.reset()

def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    test_coco(args)

if __name__ == "__main__":
    main()
