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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import numpy as np
from time import time
import os
import random
import time
import io
import json

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from utils import SSDTransformer
from ssd300 import SSD300
from train import load_checkpoint, dboxes300_coco

def parse_args():
    parser = ArgumentParser(description="Visualize models predictions on image")
    parser.add_argument('--images', '-i', nargs='*', type=str,
                        help='path to jpg image')
    parser.add_argument('--model', '-m', type=str, default='iter_240000.pt',
                        help='path to trained model')
    parser.add_argument('--threshold', '-t', type=float, default=0.10,
                        help='threshold for predictions probabilities')
    parser.add_argument('--annotations', '-a', type=str,
                        default='/coco/annotations/instances_val2017.json',
                        help='path to json with annotations')
    return parser.parse_args()

def print_image(image, model, encoder, inv_map, name_map, category_id_to_color, threshold):
    # Open image for printing
    im = Image.open(image)
    W, H = im.size

    # Prepare tensor input for model
    tmp = im.copy()
    tmp = tmp.resize((300, 300))
    img = transforms.ToTensor()(tmp)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img).unsqueeze(dim = 0)

    # Find predictions
    with torch.no_grad():
        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()

        ret = []
        for idx in range(ploc.shape[0]):
            # ease-of-use for specific predictions
            ploc_i = ploc[idx, :, :].unsqueeze(0)
            plabel_i = plabel[idx, :, :].unsqueeze(0)

            try:
                result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
            except:
                print("No object detected in image {}".format(image))
                continue

            htot, wtot = (H, W)
            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([0, loc_[0]*wtot, \
                                    loc_[1]*htot,
                                    (loc_[2] - loc_[0])*wtot,
                                    (loc_[3] - loc_[1])*htot,
                                    prob_,
                                    inv_map[label_]])

        ret = np.array(ret).astype(np.float32)

    # Choose bounding boxes for printing
    bboxes = []
    for re in ret:
        if re[5] > threshold:
            bboxes.append(re)

    print("Bounding boxes detected in image {}:".format(image))
    print(bboxes)

    # Prepare image for plotting
    img = transforms.ToTensor()(im)
    img = img.permute(1, 2, 0)
    H = img.shape[0]
    W = img.shape[1]
    fig,ax = plt.subplots(1)
    ax.imshow(img)

    # Add bboxes with labels
    used = set()
    for bbox in bboxes:
        if (bbox[6] in used):
            rect = patches.Rectangle((bbox[1], bbox[2]), bbox[3], bbox[4],
                                    edgecolor=category_id_to_color[bbox[6]],
                                    linewidth=2, facecolor='none')
        else:
            rect = patches.Rectangle((bbox[1], bbox[2]), bbox[3], bbox[4],
                                    label = name_map[bbox[6]],
                                    edgecolor=category_id_to_color[bbox[6]],
                                    linewidth=2, facecolor='none')
            used.add(bbox[6])
        ax.add_patch(rect)

    # Show image
    plt.legend(ncol=1, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()

def main():
    # Parse arguments
    args = parse_args()

    # Get categories names
    with open(args.annotations,'r') as anno:
        js = json.loads(anno.read())
        coco_names = js['categories']

    # Prepare map of COCO labels to COCO names
    name_map = {}
    for name in coco_names:
        name_map[name['id']] = name['name']

    # Prepare map of SSD to COCO labels
    deleted = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
    inv_map = {}
    cnt = 0
    for i in range(1, 81):
        while i + cnt in deleted:
            cnt += 1
        inv_map[i] = i + cnt

    # Prepare colors for categories
    category_id_to_color = dict([(cat_id, [random.uniform(0, 1) ,random.uniform(0, 1), random.uniform(0, 1)]) for cat_id in range(1, 91)])

    # Set math plot lib size
    plt.rcParams["figure.figsize"] = (12, 8)

    # Build and load SSD model
    ssd300 = SSD300(81, backbone="resnet34", model_path=None, dilation=None)
    load_checkpoint(ssd300, args.model)
    ssd300.eval()

    # Prepare encoder
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    # Print images
    for image in args.images:
        print_image(image, ssd300, encoder, inv_map, name_map, category_id_to_color, args.threshold)

if __name__ == "__main__":
    main()
