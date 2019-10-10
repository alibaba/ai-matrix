# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from SSD import _C as C

# Abstraction over coco_pipeline.SingleDaliIterator (no_dali=False)
# or torch.utils.data.DataLoader with utils.SSDTransformer (no_dali=True)
# so the input_iterator returns a 4-tuple: (img, unencoded bboxes, unencoded labels, unencoded bbox_offsets)
# this iterator returns a 3-tuple (img, encoded bboxes, encoded labels)
class EncodingInputIterator(object):
    def __init__(self, input_it, dboxes, nhwc, fake_input=False, no_dali=False):
        self._input_it    = input_it
        self._dboxes      = dboxes
        self._nhwc        = nhwc
        self._fake_input  = fake_input
        self._no_dali     = no_dali
        self._saved_batch = None

    def __next__(self):
        # special case for fake_input for all iterations after first
        if self._saved_batch is not None:
            return self._saved_batch

        (img, bbox, label, bbox_offsets) = self._input_it.__next__()

        # non-dali path is cpu, move tensors to gpu
        if self._no_dali:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()

        if bbox_offsets[-1].item() == 0:
            img   = None
            bbox  = None
            label = None
        else:
            # dali path doesn't do horizontal flip, do it here
            if not self._no_dali:
                img, bbox = C.random_horiz_flip(img, bbox, bbox_offsets, 0.5, self._nhwc)

            # massage raw ground truth into form used by loss function
            bbox, label = C.box_encoder(img.shape[0], # <- batch size
                                        bbox, bbox_offsets,
                                        label.type(torch.cuda.LongTensor),
                                        self._dboxes, 0.5)
            bbox = bbox.transpose(1,2).contiguous().cuda()
            label = label.cuda()

        if self._fake_input:
            self._saved_batch = (img, bbox, label)

        return (img, bbox, label)


    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def reset(self):
        if self._no_dali:
            # torch.utils.data.DataLoader doesn't need/want reset
            return None
        else:
            return self._input_it.reset()


# Abstraction over EncodingInputIterator to allow the input pipeline to run at
# a larger batch size than the training pipeline
class RateMatcher(object):
    def __init__(self, input_it, output_size):
        self._input_it = input_it
        self._output_size = output_size

        self._img = None
        self._bbox = None
        self._label = None
        self._offset_offset = 0

    def __next__(self):
        if (self._img is None) or (self._offset_offset >= len(self._img)):
            self._offset_offset = 0
            (self._img, self._bbox, self._label) = self._input_it.__next__()
            if self._img is not None and len(self._img) == 0:
                self._img = None
            if self._img is None:
                return (None, None, None)
            # make sure all three tensors are same size
            assert (len(self._img) == len(self._bbox)) and (len(self._img) == len(self._label))

            # semantics of split() are that it will make as many chunks as
            # necessary with all chunks of output_size except possibly the last
            # if the input size is not perfectly divisble by the the output_size
            self._img   = self._img.split(self._output_size, dim=0)
            self._bbox  = self._bbox.split(self._output_size, dim=0)
            self._label = self._label.split(self._output_size, dim=0)

        output = (self._img[self._offset_offset],
                  self._bbox[self._offset_offset],
                  self._label[self._offset_offset])
        self._offset_offset = self._offset_offset + 1
        return output

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def reset(self):
        self._img = None
        self._bbox = None
        self._label = None
        self._offset_offset = 0

        return self._input_it.reset()

