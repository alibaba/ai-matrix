# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import logging
import os

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict, is_layer_nhwc_eligible
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return
        nhwc = kwargs.get("nhwc", False)
        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)
        # transpose to NCHW before saving as checkpoint if NHWC is used
        if nhwc:
            transpose_checkpoint_model_state_nhwc_to_nchw(data["model"])
            transpose_optimizer_state_nhwc_to_nchw(self.model, data["optimizer"])
        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)
        # Convert back to NHWC if NHWC layout is used, needed for optimizer buffers
        if nhwc:
            if self.optimizer is not None:
                transpose_optimizer_state_nchw_to_nhwc(self.model, self.optimizer.state_dict()) 

    def load(self, f=None, nhwc=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, nhwc)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if nhwc:
                transpose_optimizer_state_nchw_to_nhwc(self.model, self.optimizer.state_dict())
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, nhwc):
        load_state_dict(self.model, checkpoint.pop("model"), nhwc)


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


def transpose_checkpoint_model_state_nhwc_to_nchw(model_dict):
    for k in model_dict:
        param_tensor = model_dict[k]
        needs_transpose = is_layer_nhwc_eligible(k) and len(param_tensor.shape)==4
        if needs_transpose:
            model_dict[k] = model_dict[k].permute(0,3,1,2).contiguous()

def transpose_optimizer_state_nhwc_to_nchw(model, optimizer_dict):
    layer_id_to_name_map = {}
    for name, param in model.named_parameters():
        layer_id_to_name_map[id(param)] = name
    for k in optimizer_dict['state']:
        needs_transpose = is_layer_nhwc_eligible(layer_id_to_name_map[k])
        needs_transpose = needs_transpose and  \
                          len(optimizer_dict['state'][k]['momentum_buffer'].shape) == 4
        if needs_transpose:    
            optimizer_dict['state'][k]['momentum_buffer'] =  \
                        optimizer_dict['state'][k]['momentum_buffer'].permute(0,3,1,2).contiguous()

def transpose_optimizer_state_nchw_to_nhwc(model, optimizer_dict):
    layer_id_to_name_map = {}
    for name, param in model.named_parameters():
        layer_id_to_name_map[id(param)] = name
    for k in optimizer_dict['state']:
        needs_transpose = is_layer_nhwc_eligible(layer_id_to_name_map[k])
        needs_transpose = needs_transpose and  \
                          len(optimizer_dict['state'][k]['momentum_buffer'].shape) == 4
        if needs_transpose:    
            optimizer_dict['state'][k]['momentum_buffer'] =  \
                        optimizer_dict['state'][k]['momentum_buffer'].permute(0,2,3,1).contiguous()
