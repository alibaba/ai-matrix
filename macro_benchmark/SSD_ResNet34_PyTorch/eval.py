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
from utils import DefaultBoxes, Encoder, COCODetection, SSDCropping
from PIL import Image
from base_model import Loss
from utils import SSDTransformer
from mlperf_logger import set_seeds
from ssd300 import SSD300
from sampler import GeneralDistributedSampler
from master_params import create_flat_master
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import io
import random
import torchvision.transforms as transforms
from mlperf_compliance import mlperf_log

import sys

# necessary pytorch imports
import torch.utils.data.distributed
import torch.distributed as dist

# Apex imports
try:
    import apex_C
    from apex.parallel.LARC import LARC
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

from contextlib import redirect_stdout

from SSD import _C as C

# DALI import
from coco_pipeline import COCOPipeline, DALICOCOIterator

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--eval-batch-size', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=0.212,
                        help='stop training early at threshold')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--backbone', type=str, choices=['vgg16', 'vgg16bn', 'resnet18', 'resnet34', 'resnet50'], default='resnet34')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--nhwc', action='store_true')
    parser.add_argument('--pad-input', action='store_true')
    parser.add_argument('--lr', type=float, default=1.25e-3)
    # Distributed stuff
    parser.add_argument('--local_rank', default=0, type=int,
			help='Used for multi-process training. Can either be manually set ' +
			'or automatically set by using \'python -m multiproc\'.')

    return parser.parse_args()

# make sure that arguments are all self-consistent
def validate_arguments(args):
    # nhwc can only be used with fp16
    if args.nhwc:
        assert(args.use_fp16)

    # input padding can only be used with NHWC
    if args.pad_input:
        assert(args.nhwc)

    return

def print_message(rank, *print_args):
    if rank == 0:
        print(*print_args)

def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def load_checkpoint(model, checkpoint):
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'module' from checkpoint
    saved_model = od["model"]
    for k in list(saved_model.keys()):
        if k.startswith('module.'):
            saved_model[k[7:]] = saved_model.pop(k)
    model.load_state_dict(saved_model)


def coco_eval(model, coco, cocoGt, encoder, inv_map, threshold, epoch, iteration, batch_size,
              use_cuda=True, use_fp16=False, local_rank=-1, N_gpu=1,
              use_nhwc=False, pad_input=False):
    from pycocotools.cocoeval import COCOeval
    print("")

    distributed = False
    if local_rank >= 0:
        distributed = True

    ret = []
    overlap_threshold = 0.50
    nms_max_detections = 200
    start = time.time()
    for nbatch, (img, img_id, img_size, _, _) in enumerate(coco):
        print("Parsing batch: {}/{}".format(nbatch, len(coco)), end='\r')
        with torch.no_grad():
            inp = img.cuda()

            if pad_input:
                s = inp.shape
                inp = torch.cat([inp, torch.zeros([s[0], 1, s[2], s[3]], device=inp.device)], dim=1)

            if use_nhwc:
                inp = inp.permute(0, 2, 3 ,1).contiguous()
            if use_fp16:
                inp = inp.half()

            # Get predictions
            ploc, plabel = model(inp)
            ploc, plabel = ploc.float(), plabel.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, overlap_threshold, nms_max_detections)[0]
                except:
                    #raise
                    print("No object detected in idx: {}".format(idx))
                    continue

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0]*wtot, \
                                        loc_[1]*htot,
                                        (loc_[2] - loc_[0])*wtot,
                                        (loc_[3] - loc_[1])*htot,
                                        prob_,
                                        inv_map[label_]])

    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    # Multi-GPU eval
    if distributed:
        # NCCL backend means we can only operate on GPU tensors
        ret_copy = torch.tensor(ret).cuda()

        # Everyone exchanges the size of their results
        ret_sizes = [torch.tensor(0).cuda() for _ in range(N_gpu)]
        torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())

        # Get the maximum results size, as all tensors must be the same shape for
        # the all_gather call we need to make
        max_size = 0
        sizes = []
        for s in ret_sizes:
            max_size = max(max_size, s.item())
            sizes.append(s.item())

        # Need to pad my output to max_size in order to use in all_gather
        ret_pad = torch.cat([ret_copy, torch.zeros(max_size-ret_copy.shape[0], 7, dtype=torch.float32).cuda()])

        # allocate storage for results from all other processes
        other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(N_gpu)]
        # Everyone exchanges (padded) results
        torch.distributed.all_gather(other_ret, ret_pad)

        # Now need to reconstruct the _actual_ results from the padded set using slices.
        cat_tensors = []
        for i in range(N_gpu):
            cat_tensors.append(other_ret[i][:sizes[i]][:])

        final_results = torch.cat(cat_tensors).cpu().numpy()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
        # eval size per worker
        eval_tensor = torch.LongTensor([(len(coco)-1) * batch_size + ploc.shape[0]]).to(device)
        torch.distributed.all_reduce(eval_tensor)
        eval_size = eval_tensor.item()
    else:
        # Otherwise full results are just our results
        final_results = ret
        eval_size = (len(coco)-1) * batch_size + ploc.shape[0]

    print_message(local_rank, "Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(final_results)

    if local_rank == 0 or local_rank == -1:
        E = COCOeval(cocoGt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
    else:
        # fix for cocoeval indiscriminate prints
        with redirect_stdout(io.StringIO()):
            E = COCOeval(cocoGt, cocoDt, iouType='bbox')
            E.evaluate()
            E.accumulate()
            E.summarize()

    current_accuracy = E.stats[0]
    return current_accuracy>= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]

# setup everything (model, etc) to run eval
def run_eval(args):
    from coco import COCO
    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    local_seed = set_seeds(args)
    # start timing here
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    input_size = 300
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    if args.distributed:
        val_sampler = GeneralDistributedSampler(val_coco, pad=False)
    else:
        val_sampler = None

    val_dataloader   = DataLoader(val_coco,
                                  batch_size=args.eval_batch_size,
                                  shuffle=False, # Note: distributed sampler is shuffled :(
                                  sampler=val_sampler,
                                  num_workers=args.num_workers)

    ssd300_eval = SSD300(val_coco.labelnum, backbone=args.backbone, use_nhwc=args.nhwc, pad_input=args.pad_input).cuda()
    if args.use_fp16:
        ssd300_eval = network_to_half(ssd300_eval)
    ssd300_eval.eval()

    if args.checkpoint is not None:
        load_checkpoint(ssd300_eval, args.checkpoint)

    coco_eval(ssd300_eval,
              val_dataloader,
              cocoGt,
              encoder,
              inv_map,
              args.threshold,
              0, # epoch
              0, # iter_num
              args.eval_batch_size,
              use_fp16=args.use_fp16,
              local_rank=args.local_rank if args.distributed else -1,
              N_gpu=N_gpu,
              use_nhwc=args.nhwc,
              pad_input=args.pad_input)

if __name__ == "__main__":
    args = parse_args()
    validate_arguments(args)

    torch.backends.cudnn.benchmark = True

    run_eval(args)


