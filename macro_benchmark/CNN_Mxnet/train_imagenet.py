# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
import random
import signal

import mxnet as mx
import numpy as np
from mlperf_compliance import constants as mlperf_constants
from mlperf_compliance.mlperf_log import LOGGER
import horovod.mxnet as hvd
from common import find_mxnet, dali, fit
from mlperf_log_utils import mx_resnet_print, mpiwrapper, mlperf_submission_log

def add_general_args(parser):
    parser.add_argument('--verbose', type=int, default=0,
                        help='turn on reporting of chosen algos for convolution, etc.')
    parser.add_argument('--seed', type=int, default=None,
                        help='set the seed for python, nd and mxnet rngs')
    parser.add_argument('--custom-bn-off', type=int, default=0,
                        help='disable use of custom batchnorm kernel')
    parser.add_argument('--fuse-bn-relu', type=int, default=0,
                        help='have batchnorm kernel perform activation relu')
    parser.add_argument('--fuse-bn-add-relu', type=int, default=0,
                        help='have batchnorm kernel perform add followed by activation relu')
    parser.add_argument('--input-layout', type=str, default='NCHW',
                        help='the layout of the input data (e.g. NCHW)')
    parser.add_argument('--conv-layout', type=str, default='NCHW',
                        help='the layout of the data assumed by the conv operation (e.g. NCHW)')
    parser.add_argument('--conv-algo', type=int, default=-1,
                        help='set the convolution algos (fwd, dgrad, wgrad)')
    parser.add_argument('--force-tensor-core', type=int, default=0,
                        help='require conv algos to be tensor core')
    parser.add_argument('--batchnorm-layout', type=str, default='NCHW',
                        help='the layout of the data assumed by the batchnorm operation (e.g. NCHW)')
    parser.add_argument('--batchnorm-eps', type=float, default=2e-5,
                        help='the amount added to the batchnorm variance to prevent output explosion.')
    parser.add_argument('--batchnorm-mom', type=float, default=0.9,
                        help='the leaky-integrator factor controling the batchnorm mean and variance.')
    parser.add_argument('--pooling-layout', type=str, default='NCHW',
                        help='the layout of the data assumed by the pooling operation (e.g. NCHW)')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='key-value store type')

def _get_gpu(gpus):
    idx = hvd.local_rank()
    gpu = gpus.split(",")[idx]
    return gpu

class MLPerfInit(mx.init.Xavier):
    def _init_weight(self, name, arg):
        if name.startswith("fc"):
            mx.ndarray.random.normal(0, 0.01, out=arg)
        else:
            return super()._init_weight(name, arg)

class BNZeroInit(mx.init.Xavier):
    def _init_gamma(self, name, arg):
        if name.endswith("bn3_gamma"):
            arg[:] = 0.0
        else:
            arg[:] = 1.0

if __name__ == '__main__':
    LOGGER.propagate = False
    mx_resnet_print(key=mlperf_constants.INIT_START, uniq=False)
    # parse args
    parser = argparse.ArgumentParser(description="MLPerf RN50v1.5 training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_general_args(parser)
    fit.add_fit_args(parser)
    dali.add_dali_args(parser)

    parser.set_defaults(
        # network
        network          = 'resnet-v1b',
        num_layers       = 50,

        # data
        resize           = 256,
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        # train
        num_epochs       = 100,
        lr_step_epochs   = '30,60,80',
        dtype            = 'float32'
    )
    args = parser.parse_args()

    if 'horovod' in args.kv_store:
        # initialize Horovod with mpi4py comm
        hvd.init(mpiwrapper.get_comm())
        args.gpus = _get_gpu(args.gpus)
        kv = None
        local_rank = hvd.local_rank()

        # dummy Horovod ops to initialize resources
        ctx=mx.gpu(local_rank)
        tensor1 = mx.nd.zeros(shape=(1), dtype='float16', ctx=ctx)
        tensor2 = mx.nd.zeros(shape=(1), dtype='float32', ctx=ctx)
        summed1 = hvd.allreduce(tensor1, average=False)
        summed2 = hvd.allreduce(tensor2, average=False)

    framework = 'MxNet NGC {}'.format(os.environ["NVIDIA_MXNET_VERSION"])

    #mlperf_submission_log(
    #    benchmark=mlperf_constants.RESNET,
    #    framework=framework,
    #)

    # Load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)

    # Initialize seed + random number generators
    if args.seed is None:
        args.seed = int(random.SystemRandom().randint(0, 2**16 - 1))

    if 'horovod' in args.kv_store:
        np.random.seed(args.seed)
        all_seeds = np.random.randint(2**16, size=(hvd.size()))
        args.seed = int(all_seeds[hvd.rank()])
    else:
        kv = mx.kvstore.create(args.kv_store)

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
       mx.gpu(int(i)) for i in args.gpus.split(',')]

    # Load symbol definiton and create model
    sym = net.get_symbol(**vars(args))

    model = mx.mod.Module(context=devs, symbol=sym)

    # Weights init
    initializer = MLPerfInit(
                        rnd_type='gaussian', factor_type="in", magnitude=2) if not args.bn_gamma_init0 else BNZeroInit(rnd_type='gaussian', factor_type="in", magnitude=2)

    # Start DALI pipeline
    lambda_fnc_dali_get_rec_iter=dali.build_input_pipeline(args, kv)

    arg_params, aux_params = None, None

    # Model fetch and broadcast
    if 'horovod' in args.kv_store:
        # Create dummy data shapes and bind them to the model
        data_shapes  = [mx.io.DataDesc('data',(args.batch_size, 224, 224, 4),'float16')]
        label_shapes = [mx.io.DataDesc('softmax_label',(args.batch_size,),'float32')]
        model.bind(data_shapes=data_shapes, label_shapes=label_shapes)

        # Horovod: fetch and broadcast parameters
        model.init_params(initializer, arg_params=arg_params, aux_params=aux_params)
        (arg_params, aux_params) = model.get_params()

        if arg_params is not None:
            hvd.broadcast_parameters(arg_params, root_rank=0)

        if aux_params is not None:
            hvd.broadcast_parameters(aux_params, root_rank=0)

        model.set_params(arg_params=arg_params, aux_params=aux_params)

    mx.ndarray.waitall()
    mx_resnet_print(key=mlperf_constants.INIT_STOP, sync=True)

    # Start training
    mx_resnet_print(key=mlperf_constants.RUN_START, sync=True)
    fit.fit(args, kv, model, initializer, lambda_fnc_dali_get_rec_iter, devs, arg_params, aux_params)





