# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import functools
import logging
import random
import datetime
import time
import gc

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.tester import test
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.mlperf_logger import mlperf_print, generate_seeds, broadcast_seeds, barrier
from maskrcnn_benchmark.utils.async_evaluator import init, get_evaluator, set_epoch_tag

import mlperf_compliance
from mlperf_compliance import constants
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')
# Loop over all finished async results, return a dict of { tag : (bbox_map, segm_map) }
def check_completed_tags():
    # Evaluator is only valid on master rank - all others will have nothing.
    # So, assemble lists of completed runs on master
    if is_main_process():
        evaluator = get_evaluator()

        # loop over all all epoch, result pairs that have finished
        all_results = {}
        for t, r in evaluator.finished_tasks().items():
            # Note: one indirection due to possibility of multiple test datasets
            # we only care about the first
            map_results = r# [0]
            bbox_map = map_results.results["bbox"]['AP']
            segm_map = map_results.results["segm"]['AP']
            all_results.update({ t : (bbox_map, segm_map) })

        return all_results
    
    return {}


def mlperf_test_early_exit(iteration, iters_per_epoch, tester, model, distributed, min_bbox_map, min_segm_map):
    # Note: let iters / epoch == 10k, at iter 9999 we've finished epoch 0 and need to test
    if iteration > 0 and (iteration + 1)% iters_per_epoch == 0:
        synchronize()
        epoch = iteration // iters_per_epoch + 1

        mlperf_print(key=constants.EPOCH_STOP, metadata={"epoch_num": epoch})
        mlperf_print(key=constants.BLOCK_STOP, metadata={"first_epoch_num": epoch})
        mlperf_print(key=constants.EVAL_START, metadata={"epoch_num":epoch})
        # set the async evaluator's tag correctly
        set_epoch_tag(epoch)

        # Note: No longer returns anything, underlying future is in another castle
        tester(model=model, distributed=distributed)
        # necessary for correctness
        model.train()
    else:
        # Otherwise, check for finished async results
        results = check_completed_tags()

        # on master process, check each result for terminating condition
        # sentinel for run finishing
        finished = 0
        if is_main_process():
            for result_epoch, (bbox_map, segm_map) in results.items():
                # mlperf_print(key=constants.EVAL_TARGET, value={"BBOX": min_bbox_map,
                #                                                 "SEGM": min_segm_map})
                logger = logging.getLogger('maskrcnn_benchmark.trainer')
                logger.info('bbox mAP: {}, segm mAP: {}'.format(bbox_map, segm_map))

                mlperf_print(key=constants.EVAL_ACCURACY, value={"accuracy":{"BBOX" : bbox_map, "SEGM" : segm_map}}, metadata={"epoch_num" : result_epoch} )
                mlperf_print(key=constants.EVAL_STOP, metadata={"epoch_num": result_epoch})
                # terminating condition
                if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
                    logger.info("Target mAP reached, exiting...")
                    finished = 1
                    #return True

        # We now know on rank 0 whether or not we should terminate
        # Bcast this flag on multi-GPU
        if get_world_size() > 1:
            with torch.no_grad():
                finish_tensor = torch.tensor([finished], dtype=torch.int32, device = torch.device('cuda'))
                torch.distributed.broadcast(finish_tensor, 0)
    
                # If notified, end.
                if finish_tensor.item() == 1:
                    return True
        else:
            # Single GPU, don't need to create tensor to bcast, just use value directly
            if finished == 1:
                return True

    # Otherwise, default case, continue
    return False

def mlperf_log_epoch_start(iteration, iters_per_epoch):
    # First iteration:
    #     Note we've started training & tag first epoch start
    if iteration == 0:
        mlperf_print(key=constants.BLOCK_START, metadata={"first_epoch_num":1, "epoch_count":1})
        mlperf_print(key=constants.EPOCH_START, metadata={"epoch_num":1})
        return
    if iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch + 1
        mlperf_print(key=constants.BLOCK_START, metadata={"first_epoch_num": epoch, "epoch_count": 1})
        mlperf_print(key=constants.EPOCH_START, metadata={"epoch_num": epoch})

from maskrcnn_benchmark.layers.batch_norm import FrozenBatchNorm2d
def cast_frozen_bn_to_half(module):
    if isinstance(module, FrozenBatchNorm2d):
        module.half()
    for child in module.children():
        cast_frozen_bn_to_half(child)
    return module

def train(cfg, local_rank, distributed, random_number_generator=None):
    # Model logging
    mlperf_print(key=constants.GLOBAL_BATCH_SIZE, value=cfg.SOLVER.IMS_PER_BATCH)
    mlperf_print(key=constants.NUM_IMAGE_CANDIDATES, value=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    # Optimizer logging
    # mlperf_print(key=constants.OPT_NAME, value="sgd_with_momentum")
    mlperf_print(key=constants.OPT_BASE_LR, value=cfg.SOLVER.BASE_LR)
    mlperf_print(key=constants.OPT_LR_WARMUP_STEPS, value=cfg.SOLVER.WARMUP_ITERS)
    mlperf_print(key=constants.OPT_LR_WARMUP_FACTOR, value=cfg.SOLVER.WARMUP_FACTOR)

    scheduler = make_lr_scheduler(cfg, optimizer)

    # disable the garbage collection
    gc.disable()

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    # amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level) # , verbose=cfg.AMP_VERBOSE)

    if distributed:
        model = DDP(model, delay_allreduce=True)

    arguments = {}
    arguments["iteration"] = 0
    arguments["nhwc"] = cfg.NHWC
    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    arguments["save_checkpoints"] = cfg.SAVE_CHECKPOINTS
    
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, cfg.NHWC)
    arguments.update(extra_checkpoint_data)

    # At this point we've loaded relevant checkpoint(s) and can now cast 
    # FrozenBatchNorm2d layers to half() if necessary.
    # This allows us to move parameter casting logic out of the BN code itself,
    # which was preventing us from annotating bn.forward with @script_method, which 
    # was preventing the cross-module fusion of BN with ReLU / Add-ReLU.
    if use_mixed_precision:
        model = cast_frozen_bn_to_half(model)

    mlperf_print(key=constants.INIT_STOP, sync=True)
    mlperf_print(key=constants.RUN_START, sync=True)
    barrier()

    data_loader, iters_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        random_number_generator=random_number_generator,
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # set the callback function to evaluate and potentially
    # early exit each epoch
    if cfg.PER_EPOCH_EVAL:
        per_iter_callback_fn = functools.partial(
                mlperf_test_early_exit,
                iters_per_epoch=iters_per_epoch,
                tester=functools.partial(test, cfg=cfg),
                model=model,
                distributed=distributed,
                min_bbox_map=cfg.MLPERF.MIN_BBOX_MAP,
                min_segm_map=cfg.MLPERF.MIN_SEGM_MAP)
    else:
        per_iter_callback_fn = None

    start_train_time = time.time()

    success = do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        cfg.DISABLE_REDUCED_LOGGING,
        per_iter_start_callback_fn=functools.partial(mlperf_log_epoch_start, iters_per_epoch=iters_per_epoch),
        per_iter_end_callback_fn=per_iter_callback_fn,
    )

    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print(
            "&&&& MLPERF METRIC THROUGHPUT = {:.4f} iterations / s".format((arguments["iteration"] * cfg.SOLVER.IMS_PER_BATCH) / total_training_time)
    )

    return model, success



def main():

    mlperf_compliance.mlperf_log.LOGGER.propagate = False

    mlperf_compliance.mlperf_log.setdefault(
        root_dir=os.path.dirname(os.path.abspath(__file__)),
        benchmark=constants.MASKRCNN,
        stack_offset=1,
        extra_print=False
        )

    mlperf_print(key=mlperf_compliance.constants.INIT_START)

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    # if is_main_process:
    #     # Setting logging file parameters for compliance logging
    #     os.environ["COMPLIANCE_FILE"] = '/MASKRCNN_complVv0.5.0_' + str(datetime.datetime.now())
    #     constants.LOG_FILE = os.getenv("COMPLIANCE_FILE")
    #     constants._FILE_HANDLER = logging.FileHandler(constants.LOG_FILE)
    #     constants._FILE_HANDLER.setLevel(logging.DEBUG)
    #     constants.LOGGER.addHandler(constants._FILE_HANDLER)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        # setting seeds - needs to be timed, so after RUN_START
        if is_main_process():
            master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
            seed_tensor = torch.tensor(master_seed, dtype=torch.float32, device=torch.device("cuda"))
        else:
            seed_tensor = torch.tensor(0, dtype=torch.float32, device=torch.device("cuda"))

        torch.distributed.broadcast(seed_tensor, 0)
        master_seed = int(seed_tensor.item())
    else:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)

    # actually use the random seed
    args.seed = master_seed
    # random number generator with seed set to master_seed
    random_number_generator = random.Random(master_seed)
    # mlperf_print(key=constants.RUN_SET_RANDOM_SEED, value=master_seed)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(random_number_generator, torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

    # todo sharath what if CPU
    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device='cuda')

    # Setting worker seeds
    logger.info("Worker {}: Setting seed {}".format(args.local_rank, worker_seeds[args.local_rank]))
    torch.manual_seed(worker_seeds[args.local_rank])


    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Initialise async eval
    init()

    model, success = train(cfg, args.local_rank, args.distributed, random_number_generator)

    if success is not None:
        if success:
            mlperf_print(key=constants.RUN_STOP, metadata={"status": "success"}, sync=True)
        else:
            mlperf_print(key=constants.RUN_STOP, metadata={"status": "aborted"}, sync=True)

if __name__ == "__main__":
    start = time.time()
    torch.set_num_threads(1)
    main()
    print("&&&& MLPERF METRIC TIME=", time.time() - start)
