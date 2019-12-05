# Copyright 2019 The MLPerf Authors. All Rights Reserved.
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
# =============================================================================

import os
import sys
import argparse
from google.protobuf import text_format
import ratelnn
from ratelnn.frontend.tensorflow import converter
import tensorflow as tf
import json
import shutil
import pudb
# initialize HanGuangAI
ratelnn.tf_init()

def make_path(fpth):
    """make sure directory exists before read and write

    Arguments:
        fpth {str} -- path to desired file
    """

    folder = os.path.dirname(fpth)
    if folder is None or folder == '':
        return
    if not os.path.isdir(folder):
        os.makedirs(folder)

# read graphdef from tensorflow pb file
def read_proto(pb_fname: str, as_text: bool = False):
    graph_def = tf.GraphDef()
    with open(pb_fname, "rb") as f:
        if not as_text:
            graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), graph_def)
    return graph_def

# write graphdef to tensorflow pb file
def write_proto(graph_def, pb_fname, as_text: bool = False):
    make_path(pb_fname)

    if as_text:
        with open(pb_fname, 'w') as f:
            f.write(str(graph_def))
    else:
        with open(pb_fname, 'wb') as f:
            f.write(graph_def.SerializeToString())

# initialize and update config settings for HanGuangAI quantization process
def update_config(output_type):
    config = {}
    config['input_shapes'] = [[1, 224, 224, 3]]
    config['image_shape'] = [1, 224, 224, 3]
    config['output_dir'] = 'output'
    config['model_dir'] = os.getenv('MODEL_DIR')
    config['data_dir'] = os.getenv('DATA_DIR')
    config['cal_list_dir'] = os.getenv('LG_PATH') + '/../calibration/ImageNet/cal_image_list_option_1.txt'
    config['quant_cfg'] = {}
    for item in ['avgpool_input', 
                 'avgpool_output',
                 'matmul_activations',
                 'matmul_weights',
                 'matmul_output',
                 'mean_input',
                 'mean_output', 
                 'maxpool_input', 
                 'maxpool_output', 
                 'conv_activations', 
                 'conv_weights', 
                 'conv_output', 
                 'reshape_input', 
                 'reshape_output']:
        config['quant_cfg'][item] = {}
        config['quant_cfg'][item]['is_per_channel'] = False
        config['quant_cfg'][item]['num_bits'] = 8
        config['quant_cfg'][item]['data_format'] = 'NHWC'
    for item in ['conv_weights', 
                 'conv_output',
                 'matmul_weights',
                 'matmul_output']:
        config['quant_cfg'][item]['is_per_channel'] = True
    config['quant_cfg']['conv_weights']['data_format'] = 'HWIO'
    config['quant_cfg']['matmul_activations']['data_format'] = 'NC'
    config['quant_cfg']['matmul_weights']['data_format'] = 'IO'
    config['quant_cfg']['matmul_output']['data_format'] = 'NC'
    config['quant_cfg']['reshape_output']['data_format'] = 'NC'
    config['input_nodes'] = ['input_tensor']
    config['output_nodes'] = ['ArgMax']
        
    config['fp32_model_path']  = config['model_dir'] + '/fp32.pb'
    config['qfp32_quant_path'] = config['model_dir'] + '/quant.pb'
    config['qfp32_cal_path']   = config['model_dir'] + '/cal.pb'
    config['npu_model_path']   = config['model_dir'] + '/npu.pb'
    config['output_dir']       = 'output'
    config['output_type']      = output_type

    config['cal_cmd'] = '../classification_and_detection/cpp/classification --dataset={} --calibration={}'.format(config['data_dir'], config['cal_list_dir']) + ' --model={}'
    config['test_cmd'] = '../classification_and_detection/cpp/classification --dataset={} --scenario=SingleStream --queries-single=1 --skip-warmup --count=1 --accuracy'.format(config['data_dir']) + ' --model={}'

    assert output_type in ['qfp32', 'npu'], "output_type should in [qfp32, npu]"

    config['use_prequant'] = True
    config['enable_EMA'] = False
    config['remove_input_quant'] = True
    
    return config
    
# HanGuangAI quantization process
def quantize(config):
    # create directory for tmp files generated during process
    cal_data_path = config['output_dir']
    if os.path.exists(cal_data_path):
        shutil.rmtree(cal_data_path)
    os.makedirs(cal_data_path, exist_ok=True)
    
    # read original graphdef from fp32 model
    graph_def = read_proto(config['fp32_model_path'])

    # create HanGuangAI converter object
    c = converter(graph_def, config)
    
    # calibration
    cal_model = c.to_cal()
    write_proto(cal_model, config['qfp32_cal_path'])
    cal_cmd = config['cal_cmd'].format(config['qfp32_cal_path'])
    print('cal: ' + cal_cmd)
    os.system(cal_cmd)

    # quantization
    quant_model = c.to_quant()
    write_proto(quant_model, config['qfp32_quant_path'])
    test_cmd = config['test_cmd'].format(config['qfp32_quant_path'])
    print('qfp32: ' + test_cmd)

    input_quant_path = '{}/input_quant_nodes.txt'.format(config['model_dir'])
    if os.path.exists('input_quant_nodes.txt'):
        os.remove('input_quant_nodes.txt')
    if os.path.exists(input_quant_path):
        os.remove(input_quant_path)

    os.system(test_cmd)
    
    # compilation and finalization
    npu_model = c.to_npu()
    write_proto(npu_model, config['npu_model_path'])

    if os.path.exists('input_quant_nodes.txt'):
        os.rename('input_quant_nodes.txt', input_quant_path)

    # final test to make sure the generated npu.pb working
    test_cmd = config['test_cmd'].format(config['npu_model_path'])
    print('npu: ' + test_cmd)
    os.system(test_cmd)

    
if __name__ == "__main__":
    # command line option parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_type', default='qfp32', help='quantize output type, [qfp32, npu]')
    parser.add_argument('--debug', action='store_true', help='enable debug')
    args = parser.parse_args()

    # enable debugging if needed
    if args.debug:
        pu.db

    # initialize and update config settings
    c = update_config(args.output_type)

    # HanGuangAI quantization process
    quantize(c)
