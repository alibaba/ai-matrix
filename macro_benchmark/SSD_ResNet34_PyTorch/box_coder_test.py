import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection, SSDCropping
from PIL import Image
from base_model import Loss
from utils import SSDTransformer
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

import sys

from SSD import _C as C

def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


if __name__ == "__main__":
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    saved_inputs = torch.load('inputs.pth')

    bboxes = saved_inputs['bbox'].float()
    scores = saved_inputs['scores'].float()
    criteria = float(saved_inputs['criteria'])
    max_num = int(saved_inputs['max_output'])

    print('bboxes: {}, scores: {}'.format(bboxes.shape, scores.shape))

    for i in range(bboxes.shape[0]):
        box, label, score = encoder.decode_batch(bboxes[i, :, :].unsqueeze(0), scores[i, :, :].unsqueeze(0), criteria, max_num)[0]
        print('r: {}'.format(label))
