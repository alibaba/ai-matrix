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
from PIL import Image

from SSD import _C as C

import time
import random
import numpy as np
from math import cos, sin, pi

def hue_saturation_matrix(hue, saturation):
    const_mat = np.array([[.299, .299, .299, 0.0],
                          [.587, .587, .587, 0.0],
                          [.114, .114, .114, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    sch_mat = np.array([[.701, -.299, -.300, 0.0],
                           [-.587, .413, -.588, 0.0],
                           [-.114, -.114, .886, 0.0],
                           [0.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    ssh_mat = np.array([[.168, -.328, 1.25, 0.0],
                           [.330, .035, -1.05, 0.0],
                           [-.497, .292, -.203, 0.0],
                           [0.0, 0.0, 0.0, 0.0]]).astype(np.float32)

    sch = saturation * cos(hue * pi / 180.0)
    ssh = saturation * sin(hue * pi / 180.0)
    m = (const_mat + sch * sch_mat + ssh * ssh_mat)
    return m

def contrast_matrix(contrast, center = 0.):
    scale = contrast + 1.
    bias = (-.5 * scale + .5) * (center * 2.)
    m = np.array([scale, 0., 0., 0.,
                  0., scale, 0., 0.,
                  0., 0., scale, 0.,
                  bias, bias, bias, 1.]).astype(np.float32)
    m = np.reshape(m, [4, 4])
    return m

def brightness_matrix(offset):
    m = np.array([1., 0., 0., 0.,
                  0., 1., 0., 0.,
                  0., 0., 1., 0.,
                  offset, offset, offset, 1.]).astype(np.float32)
    m = np.reshape(m, [4, 4])
    return m

def get_transform_matrix(hue, saturation, contrast, brightness):
    ctm = np.eye(4).astype(np.float32)

    hue_saturation_mat = hue_saturation_matrix(hue, saturation)
    ctm = np.dot(ctm, hue_saturation_mat)

    contrast_mat = contrast_matrix(contrast)
    ctm = np.dot(ctm, contrast_mat)

    brightness_mat = brightness_matrix(brightness)
    ctm = np.dot(ctm, brightness_mat)

    return ctm

def get_random_transform_matrix(hue = 0.05, saturation = 0.5,
                                contrast = 0.5, brightness = 0.125):
    # Defaults from original call : ColorJitter(brightness=0.125, contrast=0.5,
    #                                           saturation=0.5, hue=0.05
    #                                          )
    # brightness, contrast, saturation chosen from [max(0, 1-q), 1 + q]
    # hue chosen from [-h, h]
    # see: https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ColorJitter
    s = random.uniform(max(0, 1. - saturation), 1 + saturation)
    c = random.uniform(max(0, 1. - contrast), 1 + contrast)
    b = random.uniform(max(0, 1. - brightness), 1 + brightness)
    h = random.uniform(-hue, hue)

    return get_transform_matrix(h, s, c, b)

# img is HWC, ctm is 4x4
def apply_image_transform(img, ctm, bias):
    img = np.asarray(img).astype(np.float32)

    return C.apply_transform(*img.shape, img, ctm)

class FusedColorJitter(object):
    def __init__(self):
        pass

    def __call__(self, img):
        s = time.time()
        ctm = get_random_transform_matrix()
        e = time.time()
        ctm_time = e - s

        img = apply_image_transform(img, ctm, ctm)

        return img

