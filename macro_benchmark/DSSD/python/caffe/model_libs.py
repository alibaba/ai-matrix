import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  #assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in range(0, num):
        ret.append(var[0])
    else:
      for i in range(0, num):
        ret.append(var)
    return ret


def BNLayer(net, from_layer, out_layer, use_relu,
    use_scale=True, eps=0.001,
    bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias'):

  # parameters for batchnorm layer.
  bn_kwargs = {
    'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
    'eps': eps,
  }
  # parameters for scale bias layer after batchnorm.
  if use_scale:
    sb_kwargs = {
        'bias_term': True,
        'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
        'filler': dict(type='constant', value=1.0),
        'bias_filler': dict(type='constant', value=0.0),
        }
  else:
    bias_kwargs = {
        'param': [dict(lr_mult=1, decay_mult=0)],
        'filler': dict(type='constant', value=0.0),
        }


  bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
  net[bn_name] = L.BatchNorm(net[from_layer], in_place=True, **bn_kwargs)
  last_layer = bn_name
  if use_scale:
    sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
    net[sb_name] = L.Scale(net[last_layer], in_place=True, **sb_kwargs)
    last_layer = sb_name
  else:
    bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
    net[bias_name] = L.Bias(net[last_layer], in_place=True, **bias_kwargs)
    last_layer = bias_name

  if use_relu:
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[last_layer], in_place=True)


def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1, decay_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=decay_mult)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    use_global_stats = bn_params.get('use_global_stats', False)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=decay_mult),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def FullyBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    use_scale=True, eps=0.001,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias'):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  net[conv_name] = L.InnerProduct(net[from_layer], num_output=num_output, **kwargs)

  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def DeconvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, eps=0.001,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias'):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
           'param': [dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=1, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    deconv_param={
            'num_output':num_output,
            'kernel_size':kernel_h,
            'pad':pad_h,
            'stride':stride_h,
            'dilation':dilation,
            }
  else:
    deconv_param={
            'num_output':num_output,
            'kernel_h':kernel_h,
            'kernel_w':kernel_w,
            'pad_h':pad_h,
            'pad_w':pad_w,
            'stride_h':stride_h,
            'stride_w':stride_w,
            'dilation':dilation,
            }

  param = {'param':kwargs['param']}
  del kwargs['param']
  deconv_param.update(kwargs)
  net[conv_name] = L.Deconvolution(net[from_layer], convolution_param=deconv_param, **param )
#  if dilation > 1:
#    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)



def ResBody2Layer(net, from_layer, block_name, out2a, out2b, stride, use_branch1, dilation=1, kernel=1, pad=0,lr_mult=1, decay_mult=1, **bn_param):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        lr_mult=lr_mult, decay_mult=decay_mult,
        **bn_param)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix,
      lr_mult=lr_mult, decay_mult=decay_mult,
      **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix,
      lr_mult=lr_mult, decay_mult=decay_mult,
      **bn_param)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)



def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1, kernel=1,pad=0,lr_mult=1, decay_mult=1, **bn_param):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=kernel, pad=pad, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        lr_mult=lr_mult, decay_mult=decay_mult,
        **bn_param)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=kernel, pad=pad, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix,
      lr_mult=lr_mult, decay_mult=decay_mult,
      **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  if dilation == 1:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        lr_mult=lr_mult, decay_mult=decay_mult,
        **bn_param)
  else:
    pad_for_dilation = int((3 + (dilation - 1) * 2) - 1) // 2
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad_for_dilation, stride=1, use_scale=use_scale,
        dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix,
        lr_mult=lr_mult, decay_mult=decay_mult,
        **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix,
      lr_mult=lr_mult, decay_mult=decay_mult,
      **bn_param)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)





# Preactivation version of Residual Network
# Identity Mappings in Deep Residual Networks
def PreResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True


  BNLayer(net, from_layer, '{}preact'.format(conv_prefix), use_relu=True,
  use_scale=True, eps=0.001,
  bn_prefix=bn_prefix, bn_postfix=bn_postfix,
  scale_prefix=scale_prefix, scale_postfix=scale_postfix)

  last_layer = '{}preact_relu'.format(conv_prefix)

  branch_name = 'branch2a'
  ConvBNLayer(net, last_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  if dilation == 1:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) // 2
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=False, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)


  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, last_layer, branch_name, use_bn=False, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])


def DeconvResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'deconv/res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'deconv/bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'deconv/scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    DeconvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2a, kernel_size=stride, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer


  branch_name = 'branch2c'
  DeconvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2c, kernel_size=stride, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  if dilation == 1:
    DeconvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) // 2
    DeconvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2a'
  DeconvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2a, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'deconv/res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)




def InceptionTower(net, from_layer, tower_name, layer_params, **bn_param):
  use_scale = False
  for param in layer_params:
    tower_layer = '{}/{}'.format(tower_name, param['name'])
    del param['name']
    if 'pool' in tower_layer:
      net[tower_layer] = L.Pooling(net[from_layer], **param)
    else:
      param.update(bn_param)
      ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
          use_scale=use_scale, **param)
    from_layer = tower_layer
  return net[from_layer]

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='', anno_type=None,
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
        }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source),
        ntop=ntop, **kwargs)


def CreateAnnotatedPoseDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='',
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    if output_label:
        data, label, joint_label = L.AnnotatedPoseData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=3, **kwargs)
        return [data, label, joint_label]
    else:
        data = L.AnnotatedPoseData(name="data",
            annotated_data_param=dict(label_map_file=label_map_file,
                batch_sampler=batch_sampler),
            data_param=dict(batch_size=batch_size, backend=backend, source=source),
            ntop=1, **kwargs)
        return data


def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def ResNet18Body(net, from_layer,  use_pool5=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''

    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
            num_output=64, kernel_size=7, pad=3, stride=2,
            conv_prefix=conv_prefix, conv_postfix=conv_postfix,
            bn_prefix=bn_prefix, bn_postfix=bn_postfix,
            scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody2Layer(net, 'pool1', '2a', out2a=64, out2b=64, stride=1, use_branch1=True, **bn_param)
    ResBody2Layer(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False, **bn_param)

    from_layer = 'res2b'

    ResBody2Layer(net, from_layer, '3a', out2a=128, out2b=128, stride=2, use_branch1=True, **bn_param)
    ResBody2Layer(net, 'res3a', '3b', out2a=128, out2b=128, stride=1, use_branch1=False, **bn_param)

    from_layer = 'res3b'

    ResBody2Layer(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True, **bn_param)
    ResBody2Layer(net, 'res4a', '4b', out2a=256, out2b=256, stride=1, use_branch1=False, **bn_param)

    from_layer = 'res4b'

    ResBody2Layer(net, from_layer, '5a', out2a=512, out2b=512, stride=2, use_branch1=True, **bn_param)
    ResBody2Layer(net, 'res5a', '5b', out2a=512, out2b=512, stride=1, use_branch1=False, **bn_param)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5b, pool=P.Pooling.AVE, global_pooling=True)

    return net


def ResNet50Body(net, from_layer,  use_pool5=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''



    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
            num_output=64, kernel_size=7, pad=3, stride=2,
            conv_prefix=conv_prefix, conv_postfix=conv_postfix,
            bn_prefix=bn_prefix, bn_postfix=bn_postfix,
            scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.update('conv1', {'bias_term': True})

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True, **bn_param)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)

    from_layer = 'res2c'

    ResBody(net, from_layer, '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res3a'
    for i in ['b', 'c', 'd']:
      block_name = '3{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)


    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res4a'
    for i in ['b', 'c', 'd', 'e', 'f']:
      block_name = '4{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation, **bn_param)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)


    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def ResNet101Body(net, from_layer,  use_pool5=True, use_dilation_conv5=False,
        lr_mult=1, decay_mult=1, **bn_param):

    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''



    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
            num_output=64, kernel_size=7, pad=3, stride=2,
            conv_prefix=conv_prefix, conv_postfix=conv_postfix,
            bn_prefix=bn_prefix, bn_postfix=bn_postfix,
            scale_prefix=scale_prefix, scale_postfix=scale_postfix,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)

    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)

    from_layer = 'res2c'

    ResBody(net, from_layer, '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)

    from_layer = 'res3a'
    for i in range(1, 4):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False,
              lr_mult=lr_mult, decay_mult=decay_mult,
              **bn_param)
      from_layer = 'res{}'.format(block_name)


    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)

    from_layer = 'res4a'
    for i in range(1, 23):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False,
              lr_mult=lr_mult, decay_mult=decay_mult,
              **bn_param)

      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)

    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation,
            lr_mult=lr_mult, decay_mult=decay_mult,
            **bn_param)


    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net

def PreResNet101Body(net, from_layer, use_pool5=True, use_dilation_conv5=False):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    PreResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    PreResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    PreResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    PreResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in range(1, 4):
      block_name = '3b{}'.format(i)
      PreResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    PreResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in range(1, 23):
      block_name = '4b{}'.format(i)
      PreResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    PreResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation)
    PreResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation)
    PreResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def ResNet152Body(net, from_layer, use_pool5=True, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True, **bn_param)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False, **bn_param)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res3a'
    for i in range(1, 8):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res4a'
    for i in range(1, 36):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False, **bn_param)
      from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
      stride = 1
      dilation = 2

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=stride, use_branch1=True, dilation=dilation, **bn_param)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False, dilation=dilation, **bn_param)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def InceptionV3Body(net, from_layer, output_pred=False, **bn_param):
  # scale is fixed to 1, thus we ignore it.
  use_scale = False

  out_layer = 'conv'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_1'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_2'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  out_layer = 'conv_3'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_4'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool_1'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  # inceptions with 1x1, 3x3, 5x5 convolutions
  for inception_id in range(0, 3):
    if inception_id == 0:
      out_layer = 'mixed'
      tower_2_conv_num_output = 32
    else:
      out_layer = 'mixed_{}'.format(inception_id)
      tower_2_conv_num_output = 64
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3(in sequence) convolutions
  out_layer = 'mixed_3'
  towers = []
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
      dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  # inceptions with 1x1, 7x1, 1x7 convolutions
  for inception_id in range(4, 8):
    if inception_id == 4:
      num_output = 128
    elif inception_id == 5 or inception_id == 6:
      num_output = 160
    elif inception_id == 7:
      num_output = 192
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ], **bn_param)
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  out_layer = 'mixed_8'
  towers = []
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}/tower_1'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
      dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
      dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  for inception_id in range(9, 11):
    num_output = 384
    num_output2 = 448
    if inception_id == 9:
      pool = P.Pooling.AVE
    else:
      pool = P.Pooling.MAX
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)

    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  if output_pred:
    net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
    net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
    net.softmax_prob = L.Softmax(net.softmax)

  return net

def CreateMultiBoxHead(net, from_layers=[],
        normalizations=[],
        use_scale=True, inter_layer_depth=[],
        subpixel_sizes=[], subpixel_dims=[],
        lr_mult=1, decay_mult=1, **bn_param):

    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"
    if subpixel_sizes :
        assert len(from_layers) == len(subpixel_sizes), "from_layers and subpixel_sizes should have same length"

    if subpixel_dims :
        assert len(from_layers) == len(subpixel_dims), "from_layers and subpixel_dims should have same length"

    num = len(from_layers)
    output_layers = []
    priorbox_layers = []
    loc_layers = []
    conf_layers = []

    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        base_inter_name = from_layer
        inter_name = base_inter_name
        if inter_layer_depth:
            if len(inter_layer_depth[i]) > 0:
                for sub_len in inter_layer_depth[i]:
                    inter_name = "{}_inter{}".format(inter_name, sub_len)
                    #ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult, num_output=sub_len, kernel_size=3, pad=1, stride=1, **bn_param)
                    ResBody(net, from_layer, inter_name, out2a=sub_len, out2b=sub_len, out2c=(sub_len*4), stride=1, use_branch1=True,
                            lr_mult=1, decay_mult=1,  **bn_param)

                    from_layer = "res{}".format(inter_name)

        if subpixel_sizes:
            if subpixel_sizes[i] > 1:
                subpixel_name = "{}_subpixel".format(from_layer)
                DeconvBNLayer(net, from_layer, subpixel_name, True, True, subpixel_dims[i], subpixel_sizes[i], 0, subpixel_sizes[i],
                            lr_mult=1, decay_mult=1,  **bn_param)

                from_layer = subpixel_name
                subpixel_size = subpixel_sizes[i]

        output_layers.append(from_layer)

    return output_layers


def CreateMultiBoxOutput(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], starts=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, kernel_sizes=[], pads=[],
        conf_postfix='', loc_postfix='', conf_inter=[], loc_inter=[],
        **bn_param):

    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"

    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    if starts:
        assert len(from_layers) == len(starts), "from_layers and starts should have same length"

    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

    num = len(from_layers)
    mbox_layers = []
    priorbox_layers = []
    loc_layers = []
    conf_layers = []


    for i in range(0, num):
        from_layer = from_layers[i]

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]
        if len(starts) > i:
            start = starts[i]


        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        loc_name = from_layer
        if loc_inter:
            if loc_inter[i]:
                loc_inter_name = "{}_inter".format(name)
                ConvBNLayer(net, from_layer, loc_inter_name, use_bn=True, use_relu=True,
                        lr_mult=lr_mult,num_output=loc_inter[i], kernel_size=3, pad=1, stride=1,
                        **bn_param)
                loc_name = loc_inter_name

        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, loc_name, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_sizes[i], pad=pads[i], stride=1,  **bn_param)
        loc_layers.append(name)

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        conf_name = from_layer
        if conf_inter:
            if conf_inter[i]:
                conf_inter_name = "{}_inter".format(name)
                ConvBNLayer(net, from_layer, conf_inter_name, use_bn=True, use_relu=True,
                        lr_mult=lr_mult,num_output=loc_inter[i], kernel_size=3, pad=1, stride=1,
                        **bn_param)
                conf_name = conf_inter_name


        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, conf_name, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_sizes[i], pad=pads[i], stride=1, **bn_param)
        conf_layers.append(name)

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if start:
            net.update(name, {'offset': start})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(name)

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_sizes[i], pad=pads[i], stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Loc layers
    out_loc_layers = []
    for name in loc_layers:
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        out_loc_layers.append(net[flatten_name])

    name = "mbox_loc"
    net[name] = L.Concat(*out_loc_layers, axis=1)
    mbox_layers.append(net[name])

    # Conf layers
    out_conf_layers = []
    for name in conf_layers:
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        out_conf_layers.append(net[flatten_name])

    name = "mbox_conf"
    net[name] = L.Concat(*out_conf_layers, axis=1)
    mbox_layers.append(net[name])


    # PriorBox layers
    out_priorbox_layers = []
    for name in priorbox_layers:
        out_priorbox_layers.append(net[name])

    name = "mbox_priorbox"
    net[name] = L.Concat(*out_priorbox_layers, axis=2)
    mbox_layers.append(net[name])

    # Concatenate priorbox, loc, and conf layers.
#    if use_objectness:
#        name = "mbox_objectness"
#        net[name] = L.Concat(*objectness_layers, axis=1)
#        mbox_layers.append(net[name])

    return mbox_layers

