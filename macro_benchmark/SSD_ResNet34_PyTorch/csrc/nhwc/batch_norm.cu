/******************************************************************************
*
* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*

 ******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Handle.h>
#include <THC/THCNumerics.cuh>

#include <torch/torch.h>
#include <torch/extension.h>

#include "THC/THC.h"

#include "Descriptors.h"

#include <cuda.h>

inline size_t round_up_to_multiple(size_t x, int multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

const float BN_MIN_EPSILON = 1e-4;

// TODO: Stop manually allocating CUDA memory; allocate an ATen byte
// tensor instead.
struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    data = THCudaMalloc(at::globalContext().lazyInitCUDA(), size);
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(at::globalContext().lazyInitCUDA(), data);
    }
  }

  size_t size;
  void* data;
};

// Return {y. save_mean, save_var, reserve}
std::vector<at::Tensor> nhwc_bn_fwd_train_cudnn_impl(
                       const at::Tensor& x_t,
                       const at::Tensor& z_t,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu,
                       const bool fuse_add) {

  // We assume later that both x and z are contiguous
  at::Tensor x = x_t.contiguous();

  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // Allocate output tensor
  at::Tensor z, y = at::empty_like(x); // ({N, H, W, C}, x.options());

  // Setup tensor descriptors
  at::native::nhwc::TensorDescriptor x_desc, y_desc, z_desc, bn_desc;
  x_desc.set(x);
  y_desc.set(y);
  if (fuse_add) {
    z = z_t.contiguous();
    z_desc.set(z);
  }

  auto bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  auto bn_op = fuse_relu ? ((fuse_add) ? CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION : CUDNN_BATCHNORM_OPS_BN_ACTIVATION) : CUDNN_BATCHNORM_OPS_BN;

  // get derived tensor descriptor
  cudnnDeriveBNTensorDescriptor(bn_desc.mut_desc(), x_desc.desc(), bn_mode);

  // create activation descriptor
  cudnnActivationDescriptor_t activ_desc;
  cudnnCreateActivationDescriptor(&activ_desc);
  cudnnSetActivationDescriptor(activ_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);

  // get the workspace size and allocate
  size_t ws_num_bytes;
  AT_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      at::native::getCudnnHandle(),
      bn_mode,
      bn_op,
      x_desc.desc(),
      x_desc.desc(),
      y_desc.desc(),
      bn_desc.desc(),
      (fuse_relu) ? activ_desc : nullptr,
      &ws_num_bytes));
  Workspace ws(ws_num_bytes);

  // get the reserved size and allocate as tensor
  size_t reserve_num_bytes;
  AT_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      at::native::getCudnnHandle(),
      bn_mode,
      bn_op,
      (fuse_relu) ? activ_desc : nullptr,
      x_desc.desc(),
      &reserve_num_bytes));

  at::Tensor reserve = torch::empty({static_cast<long>(reserve_num_bytes)}, torch::CUDA(at::kByte));

  // call cuDNN
  float one = 1.f, zero = 0.f;
  auto eps = std::max(epsilon, BN_MIN_EPSILON);
  at::Tensor save_mean = torch::empty({C}, scale.options());
  at::Tensor save_var = torch::empty({C}, scale.options());
  AT_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
      at::native::getCudnnHandle(),
      bn_mode,
      bn_op,
      &one,
      &zero,
      x_desc.desc(),
      x.data<at::Half>(),
      fuse_add ? z_desc.desc() : nullptr,       // z descriptor for BN-Add-Relu
      fuse_add ? z.data<at::Half>() : nullptr,  // z for BN-Add-ReLU
      y_desc.desc(),
      y.data<at::Half>(),
      bn_desc.desc(),
      scale.data<float>(),
      bias.data<float>(),
      momentum,
      running_mean.data<float>(),
      running_inv_var.data<float>(),
      eps,
      save_mean.data<float>(),
      save_var.data<float>(),
      (fuse_relu) ? activ_desc : nullptr,
      ws.data,
      ws_num_bytes,
      reserve.data<uint8_t>(),                        // reserve space ptr
      reserve.numel()));                              // reserve space bytes

  return {y, save_mean, save_var, reserve};
}

// Return {y. save_mean, save_var, reserve}
std::vector<at::Tensor> nhwc_bn_fwd_train_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu) {
  return nhwc_bn_fwd_train_cudnn_impl(
      x,
      at::Tensor(),
      scale,
      bias,
      running_mean,
      running_inv_var,
      momentum,
      epsilon,
      fuse_relu,
      false /* fuse add */);
}

std::vector<at::Tensor> nhwc_bn_add_fwd_train_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu) {
  return nhwc_bn_fwd_train_cudnn_impl(
      x,
      z,
      scale,
      bias,
      running_mean,
      running_inv_var,
      momentum,
      epsilon,
      fuse_relu,
      true /* fuse add */);
}

at::Tensor nhwc_bn_fwd_eval_cudnn_impl(
                       const at::Tensor& x_t,
                       const at::Tensor& z_t,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu,
                       const bool fuse_add) {

  // We assume later that both x and z are contiguous
  at::Tensor x = x_t.contiguous();

  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // Allocate output tensor
  at::Tensor z, y = torch::empty({N, H, W, C}, x.options());

  // this does BN and optional fused Relu

  // Setup tensor descriptors
  at::native::nhwc::TensorDescriptor x_desc, y_desc, z_desc, bn_desc;
  x_desc.set(x);
  y_desc.set(y);
  if (fuse_add) {
    z = z_t.contiguous();
    z_desc.set(z);
  }

  auto bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

  // get derived tensor descriptor
  AT_CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc.mut_desc(), x_desc.desc(), bn_mode));

  // call cuDNN
  float one = 1.f, zero = 0.f;
  auto eps = std::max(epsilon, BN_MIN_EPSILON);
  AT_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      at::native::getCudnnHandle(),
      CUDNN_BATCHNORM_SPATIAL,
      &one,
      &zero,
      x_desc.desc(),
      x.data<at::Half>(),
      y_desc.desc(),
      y.data<at::Half>(),
      bn_desc.desc(),
      scale.data<float>(),
      bias.data<float>(),
      running_mean.data<float>(),
      running_inv_var.data<float>(),
      eps));

  if (fuse_add) {
    y += z;
  }

  if (fuse_relu) {
    y = y.relu_();
  }

  return y;
}

at::Tensor nhwc_bn_fwd_eval_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu) {
  return nhwc_bn_fwd_eval_cudnn_impl(
      x,
      at::Tensor(),
      scale,
      bias,
      running_mean,
      running_inv_var,
      momentum,
      epsilon,
      fuse_relu,
      false);
}

at::Tensor nhwc_bn_add_fwd_eval_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu) {
  return nhwc_bn_fwd_eval_cudnn_impl(
      x,
      z,
      scale,
      bias,
      running_mean,
      running_inv_var,
      momentum,
      epsilon,
      fuse_relu,
      true);
}

std::vector<at::Tensor> nhwc_bn_bwd_cudnn_impl(
                       const at::Tensor& x_t,
                       const at::Tensor& y_t,
                       const at::Tensor& dy_t,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& save_mean,
                       const at::Tensor& save_var,
                       const at::Tensor& reserve,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu,
                       const bool fuse_add) {
  // We assume that x, y, dy are all contiguous
  at::Tensor x = x_t.contiguous();
  at::Tensor y = y_t.contiguous();
  at::Tensor dy = dy_t.contiguous();

  // shape
  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // outputs
  at::Tensor x_grad, scale_grad, bias_grad, z_grad;

  // Allocate outputs
  x_grad = at::empty_like(x);
  scale_grad = at::empty_like(scale);
  bias_grad = at::empty_like(bias);

  // Setup tensor descriptors
  at::native::nhwc::TensorDescriptor x_desc, y_desc, dx_desc, dy_desc, dz_desc, bn_desc;
  x_desc.set(x);
  y_desc.set(y);
  dx_desc.set(x_grad);
  dy_desc.set(dy);

  if (fuse_add) {
    z_grad = at::empty_like(x);
    dz_desc.set(z_grad);
  } else {
    z_grad = torch::empty({}, torch::CUDA(at::kFloat));
  }

  auto bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  auto bn_op = fuse_relu ? ((fuse_add) ? CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION : CUDNN_BATCHNORM_OPS_BN_ACTIVATION) : CUDNN_BATCHNORM_OPS_BN;

  // get derived tensor descriptor
  AT_CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc.mut_desc(), x_desc.desc(), bn_mode));

  // create activation descriptor
  cudnnActivationDescriptor_t activ_desc;
  cudnnCreateActivationDescriptor(&activ_desc);
  AT_CUDNN_CHECK(cudnnSetActivationDescriptor(activ_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

  // get the workspace size and allocate
  size_t ws_num_bytes;
  AT_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      at::native::getCudnnHandle(),
      bn_mode,
      bn_op,
      x_desc.desc(),
      y_desc.desc(),
      dy_desc.desc(),
      fuse_add ? dz_desc.desc() : nullptr,  // dz_desc
      dx_desc.desc(),
      bn_desc.desc(),
      (fuse_relu) ? activ_desc : nullptr,
      &ws_num_bytes));
  Workspace ws(ws_num_bytes);

  // get the reserved size and allocate as tensor

  // call cuDNN
  float one = 1.f, zero = 0.f;
  auto eps = std::max(epsilon, BN_MIN_EPSILON);
  AT_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
      at::native::getCudnnHandle(),
      bn_mode,
      bn_op,
      &one,
      &zero,
      &one,
      &zero,
      x_desc.desc(),
      x.data<at::Half>(),
      y_desc.desc(),
      y.data<at::Half>(),
      dy_desc.desc(),
      dy.data<at::Half>(),
      fuse_add ? dz_desc.desc() : nullptr,          // dz_desc
      fuse_add ? z_grad.data<at::Half>() : nullptr, // dz_data
      dx_desc.desc(),
      x_grad.data<at::Half>(),
      bn_desc.desc(),
      scale.data<float>(),
      bias.data<float>(),
      scale_grad.data<float>(),
      bias_grad.data<float>(),
      eps,
      save_mean.data<float>(),
      save_var.data<float>(),
      (fuse_relu) ? activ_desc : nullptr,
      ws.data,
      ws_num_bytes,
      (fuse_add) ? reserve.data<uint8_t>() : nullptr,        // reserve space ptr
      (fuse_add) ? reserve.numel() : 0));              // reserve space bytes

  return std::vector<at::Tensor>{x_grad, z_grad, scale_grad, bias_grad};
}

std::vector<at::Tensor> nhwc_bn_bwd_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& y,
                       const at::Tensor& dy,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& save_mean,
                       const at::Tensor& save_var,
                       const at::Tensor& reserve,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu) {
  return nhwc_bn_bwd_cudnn_impl(
      x,
      y,
      dy,
      scale,
      bias,
      running_mean,
      running_inv_var,
      save_mean,
      save_var,
      reserve,
      momentum,
      epsilon,
      fuse_relu,
      false);
}

std::vector<at::Tensor> nhwc_bn_add_bwd_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& y,
                       const at::Tensor& dy,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& save_mean,
                       const at::Tensor& save_var,
                       const at::Tensor& reserve,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu) {
  return nhwc_bn_bwd_cudnn_impl(
      x,
      y,
      dy,
      scale,
      bias,
      running_mean,
      running_inv_var,
      save_mean,
      save_var,
      reserve,
      momentum,
      epsilon,
      fuse_relu,
      true);
}

