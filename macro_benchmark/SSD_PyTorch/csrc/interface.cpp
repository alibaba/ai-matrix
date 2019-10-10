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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ArrayRef.h>
#include <ATen/ScalarType.h>
#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Generator.h"


namespace py = pybind11;

namespace at { namespace native { namespace nhwc {
// NHWC conv
// fprop (X, W) -> Y
at::Tensor cudnn_convolution_nhwc(
    const at::Tensor& input_t, const at::Tensor& weight_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic);
// fprop (X, W, b) -> Y
at::Tensor cudnn_convolution_with_bias_nhwc(
    const at::Tensor& input_t, const at::Tensor& weight_t, const at::Tensor& bias_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic);
// bprop (X, dY, W) -> (dX, dW)
std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask);
// bprop (X, dY, W) -> (dX, dW, db)
std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_convolution_backward_with_bias_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask);

}}}

// NHWC Batch norm
std::vector<at::Tensor> nhwc_bn_fwd_train_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu);

std::vector<at::Tensor> nhwc_bn_add_fwd_train_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu);

at::Tensor nhwc_bn_fwd_eval_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu);

at::Tensor nhwc_bn_add_fwd_eval_cudnn(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu);

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
                       const bool fuse_relu);

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
                       const bool fuse_relu);

// NHWC MaxPool
at::Tensor max_pool_nhwc_fwd(
                       const at::Tensor& x,
                       const int kernel,
                       const int stride,
                       const int padding,
                       const int dilation);

at::Tensor max_pool_nhwc_bwd(const at::Tensor& x,
                             const at::Tensor& y,
                             const at::Tensor& grad_y,
                             const int kernel,
                             const int stride,
                             const int padding,
                             const int dilation);

// Box encoder
std::vector<at::Tensor> box_encoder(const int N_img,
                                    const at::Tensor& bbox_input,
                                    const at::Tensor& bbox_offsets,
                                    const at::Tensor& labels_input,
                                    const at::Tensor& dbox,
                                    const float criteria = 0.5);

at::Tensor calc_ious(const int N_img,
                     const at::Tensor& boxes1,
                     const at::Tensor& boxes1_offsets,
                     const at::Tensor& boxes2);

// Box decoder
std::vector<at::Tensor> nms(const int N, // number of images
                            const int num_classes,
                            const at::Tensor score_offsets,
                            const at::Tensor sorted_scores,
                            const at::Tensor scored_scores_idx,
                            const at::Tensor bboxes,
                            const float criteria,
                            const int max_num);

std::vector<at::Tensor> random_horiz_flip(
                             at::Tensor& img,
                             at::Tensor& bboxes,
                             const at::Tensor& bbox_offsets,
                             const float p,
                             const bool nhwc);

// Fused color jitter application
// ctm [4,4], img [H, W, C]
py::array_t<float> apply_transform(int H, int W, int C, py::array_t<float> img, py::array_t<float> ctm) {
  auto img_buf = img.request();
  auto ctm_buf = ctm.request();

  // printf("H: %d, W: %d, C: %d\n", H, W, C);
  py::array_t<float> result{img_buf.size};
  auto res_buf = result.request();

  float *img_ptr = (float *)img_buf.ptr;
  float *ctm_ptr = (float *)ctm_buf.ptr;
  float *res_ptr = (float *)res_buf.ptr;

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      float *ptr = &img_ptr[h * W * C + w * C];
      float *out_ptr = &res_ptr[h * W * C + w * C];
      // manually unroll over C
      out_ptr[0] = ctm_ptr[0] * ptr[0] + ctm_ptr[1] * ptr[1] + ctm_ptr[2] * ptr[2] + ctm_ptr[3];
      out_ptr[1] = ctm_ptr[4] * ptr[0] + ctm_ptr[5] * ptr[1] + ctm_ptr[6] * ptr[2] + ctm_ptr[7];
      out_ptr[2] = ctm_ptr[8] * ptr[0] + ctm_ptr[9] * ptr[1] + ctm_ptr[10] * ptr[2] + ctm_ptr[11];
    }
  }

  result.resize({H, W, C});

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // convolution stuff
  m.def("cudnn_convolution_nhwc", &at::native::nhwc::cudnn_convolution_nhwc, "cudnn_convolution_nhwc");
  m.def("cudnn_convolution_with_bias_nhwc", &at::native::nhwc::cudnn_convolution_with_bias_nhwc, "cudnn_convolution_with_bias_nhwc");
  m.def("cudnn_convolution_backward_nhwc", &at::native::nhwc::cudnn_convolution_backward_nhwc, "cudnn_convolution_backward_nhwc");
  m.def("cudnn_convolution_backward_with_bias_nhwc", &at::native::nhwc::cudnn_convolution_backward_with_bias_nhwc, "cudnn_convolution_backward_with_bias_nhwc");
  // BN
  // // Forward
  m.def("bn_fwd_nhwc_cudnn", &nhwc_bn_fwd_train_cudnn, "bn_fwd_nhwc_cudnn");
  m.def("bn_add_fwd_nhwc_cudnn", &nhwc_bn_add_fwd_train_cudnn, "bn_add_fwd_nhwc_cudnn");
  // Eval
  m.def("bn_fwd_eval_nhwc_cudnn", &nhwc_bn_fwd_eval_cudnn, "bn_fwd_eval_nhwc_cudnn");
  m.def("bn_add_fwd_eval_nhwc_cudnn", &nhwc_bn_add_fwd_eval_cudnn, "bn_add_fwd_eval_nhwc_cudnn");
  // Bwd
  m.def("bn_bwd_nhwc_cudnn", &nhwc_bn_bwd_cudnn, "bn_bwd_nhwc_cudnn");
  m.def("bn_add_bwd_nhwc_cudnn", &nhwc_bn_add_bwd_cudnn, "bn_add_bwd_nhwc_cudnn");
  // MaxPool
  m.def("max_pool_fwd_nhwc", &max_pool_nhwc_fwd, "max_pool_fwd_nhwc");
  m.def("max_pool_bwd_nhwc", &max_pool_nhwc_bwd, "max_pool_bwd_nhwc");
  // batched box encoder
  m.def("box_encoder", &box_encoder, "box_encoder");
  m.def("calc_ious", &calc_ious, "calc_ious");
  // box decoder
  m.def("nms", &nms, "nms");
  m.def("random_horiz_flip", &random_horiz_flip, "random_horiz_flip");
  // Apply fused color jitter
  m.def("apply_transform", &apply_transform, "apply_transform");
}
