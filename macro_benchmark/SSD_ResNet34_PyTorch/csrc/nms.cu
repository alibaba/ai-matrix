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
#include <THC/THCNumerics.cuh>
#include <THC/THC.h>

#include <cuda.h>

#include <torch/torch.h>
#include <torch/extension.h>

namespace {

__device__
float calc_single_iou(const float4 b1, const float4 b2) {
  // (lt), (rb)
  float l = max(b1.x, b2.x);
  float t = max(b1.y, b2.y);
  float r = min(b1.z, b2.z);
  float b = min(b1.w, b2.w);

  float first = (r - l);
  first = (first < 0) ? 0 : first;
  float second = (b - t);
  second = (second < 0) ? 0 : second;

  float intersection = first * second;

  float area1 = (b1.w - b1.y) * (b1.z - b1.x);
  float area2 = (b2.w - b2.y) * (b2.z - b2.x);

  return intersection / (area1 + area2 - intersection);
}

// Choose whether or not to delete a box
// return 1 to delete, 0 to keep
__device__
uint8_t masked_iou(const float4 box1,
                const float4 box2,
                const uint8_t box2_deleted,
                const float criteria) {
  // if box2 isn't already deleted, calculate IoU
  if (box2_deleted == 1) return 1;

  float iou = calc_single_iou(box1, box2);

  // if iou < criteria, keep otherwise delete
  return (iou < criteria) ? 0 : 1;
}

// Based on what has been deleted, get the first non-deleted index
// and the count of non-deleted values
__device__
void get_current_num_idx(const uint8_t *deleted,
                         const int num_to_consider,
                         int *first_non_deleted,
                         int *remaining) {
  // dumb.
  // TODO: Not dumb, actually parallel
  int first = INT_MAX;
  int count = 0;
  for (int i = 0; i < num_to_consider; ++i) {
    // if element is deleted, ignore
    if (deleted[i] == 0) {
      first = (i < first) ? i : first;
      count++;
    }
  }
  *first_non_deleted = first;
  *remaining = count;
}


__global__
void nms_kernel(const int N,
                const int num_classes,
                const int *score_offsets,
                const float *scores,
                const long *score_idx,
                const float4 *bboxes,
                const float criteria,      // IoU threshold
                const int max_num,         // maximum number of candidate boxes to use
                uint8_t *deleted,          // assume initialised to false for all values
                long *num_candidates_out,   // number of outputs for this class
                float *score_out,          // output scores
                float4 *bboxes_out,        // output bboxes
                long *labels_out) {        // output labels
  // launch one block per class for now
  // Ignore class 0 (background) by starting at 1
  const int cls = blockIdx.x + 1;

  // offsets into scores and their indices
  const int offset_start = score_offsets[cls];
  const int offset_end   = score_offsets[cls+1];
  const int num_scores = offset_end - offset_start;
  // alias into local scores, indices and deleted buffers
  const float *local_scores = &scores[offset_start];
  const long *local_indices = &score_idx[offset_start];
  uint8_t *local_deleted = &deleted[offset_start];

  // aliases into output buffers
  float *local_score_out = &score_out[offset_start];
  float4 *local_bbox_out = &bboxes_out[offset_start];
  long *local_labels_out = &labels_out[offset_start];

  // Nothing to do here - early exit
  if (num_scores == 0) {
    if (threadIdx.x == 0) {
      num_candidates_out[cls] = 0;
    }
    return;
  }

  // how many scores we care about
  int num_to_consider = min(num_scores, max_num);
  int current_num = num_to_consider;

  // always start by looking at the first (highest) score
  int first_score_idx = 0;

  // store _global_ bbox candidate indices in shmem
  __shared__ int local_candidates[200];
  // also store _local_ indices for scores
  __shared__ int local_score_indices[200];
  // only thread 0 tracks how many candidates there are - need
  // to distribute that via shmem
  __shared__ int shared_num_candidates;

  // index into shmem buffer for storing candidates
  int current_candidate_idx = 0;

  // initialise all shmem values to sentinels for sanity
  for (int i = threadIdx.x; i < 200; i += blockDim.x) {
    local_candidates[i] = -1;
    local_score_indices[i] = -1;
  }
  // Shouldn't be necessary, make sure that no entries are
  // coming in deleted from poor initialisation.
  for (int i = threadIdx.x; i < num_scores; i += blockDim.x) {
    local_deleted[i] = 0;
  }
  __syncthreads();

  // While there's more scores/boxes to process
  while (current_num > 0) {

    // get the candidate index & bbox
    // first_score_idx is _local_ into the aliased index-storing buffer
    // candidate_idx is _global_ into the bbox buffer
    const long candidate_idx = local_indices[first_score_idx];
    const float4 candidate_bbox = bboxes[candidate_idx];
    // Now we've looked at this candidate, remove it from consideration
    local_deleted[first_score_idx] = 1;

    // calculate the IoUs of candidate vs. remaining boxes & manipulate delete array
    // standard block-stride loop over boxes
    for (int i = threadIdx.x; i < num_to_consider; i += blockDim.x) {
      // Know we've already looked at all entries before the candidate, so we can ignore them
      // TODO: handle this loop more efficiently w.r.t. skipped entries
      if (i > first_score_idx) {
        long test_idx = local_indices[i];
        float4 test_bbox = bboxes[test_idx];
        // Note if we need to delete this box
        local_deleted[i] = masked_iou(candidate_bbox, test_bbox, local_deleted[i], criteria);
      }
    }

    // make sure all IoU / deletion calcs are done
    // NOTE: shouldn't be necessary, candidate writing isn't dependent on the results
    // of IoU calcs, and sync point _after_ that writing should cover.
    // __syncthreads();

    // write the candidate idx into shmem and increment storage pointer
    if (threadIdx.x == 0) {
      // idx into global bbox array
      local_candidates[current_candidate_idx] = candidate_idx;
      // idx into local scores
      local_score_indices[current_candidate_idx] = first_score_idx;
      // increment storage location
      current_candidate_idx++;
    }

    __syncthreads();

    // Now, get the number of remaining boxes and the first non-deleted idx
    get_current_num_idx(local_deleted, num_to_consider, &first_score_idx, &current_num);

    __syncthreads();
  }

  // Note: Only thread 0 has the correct number of candidates (as that's the thread
  // that actually handles candidate tracking). Need to bcast the correct value to
  // everyone for multi-threaded output writing, so do that here via shmem.
  if (threadIdx.x == 0) {
    shared_num_candidates = current_candidate_idx;
  }
  __syncthreads();

  // at this point we should have all candidate indices for this class
  // use them to write out scores, bboxes and labels
  for (int i = threadIdx.x; i < shared_num_candidates; i += blockDim.x) {
    local_score_out[i] = local_scores[local_score_indices[i]];
    local_bbox_out[i] = bboxes[local_candidates[i]]; // bboxes[local_indices[i]];
    local_labels_out[i] = cls;
  }

  // write the final number of candidates from this class to a buffer
  if (threadIdx.x == 0) {
    num_candidates_out[cls] = current_candidate_idx;
  }
}

__global__
void squash_outputs(const int N, // number of sets of outputs
                    const long *num_candidates, // number of candidates per entry
                    const int *output_offsets, // offsets into outputs
                    const float *output_scores,
                    const float4 *output_boxes,
                    const long* output_labels,
                    const long* squashed_offsets,
                    float *squashed_scores,
                    float4 *squashed_boxes,
                    long *squashed_labels) {
  // block per output
  const int cls = blockIdx.x + 1;

  const int num_to_write = num_candidates[cls];
  const long read_offset = output_offsets[cls];
  const long write_offset = squashed_offsets[cls];

  for (int i = threadIdx.x; i < num_to_write; i += blockDim.x) {
    // Read
    auto score = output_scores[read_offset + i];
    auto bbox = output_boxes[read_offset + i];
    auto label = output_labels[read_offset + i];

    // Write
    squashed_scores[write_offset + i] = score;
    squashed_boxes[write_offset + i] = bbox;
    squashed_labels[write_offset + i] = label;
  }
}

};  // anonymous namespace

std::vector<at::Tensor> nms(const int N, // number of images
                            const int num_classes,
                            const at::Tensor score_offsets,
                            const at::Tensor sorted_scores,
                            const at::Tensor sorted_scores_idx,
                            const at::Tensor bboxes,
                            const float criteria,
                            const int max_num) {
  // Run all classes in different blocks, ignore background class 0
  const int num_blocks = num_classes - 1;
  const int total_scores = score_offsets[score_offsets.numel()-1].item<int>();

  // track which elements have been deleted in each iteration
  at::Tensor deleted = torch::zeros({total_scores}, torch::CUDA(at::kByte));
  // track how many outputs we have for each class
  at::Tensor num_candidates_out = torch::zeros({num_classes}, torch::CUDA(at::kLong));
  // outputs
  at::Tensor score_out = torch::empty({total_scores}, torch::CUDA(at::kFloat));
  at::Tensor label_out = torch::empty({total_scores}, torch::CUDA(at::kLong));
  at::Tensor bbox_out = torch::empty({total_scores, 4}, torch::CUDA(at::kFloat));

  // Run the kernel
  const int THREADS_PER_BLOCK = 64;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  nms_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(N,
                                                           num_classes,
                                                           score_offsets.data<int>(),
                                                           sorted_scores.data<float>(),
                                                           sorted_scores_idx.data<long>(),
                                                           (float4*)bboxes.data<float>(),
                                                           criteria,
                                                           max_num,
                                                           deleted.data<uint8_t>(),
                                                           num_candidates_out.data<long>(),
                                                           score_out.data<float>(),
                                                           (float4*)bbox_out.data<float>(),
                                                           label_out.data<long>());
  THCudaCheck(cudaGetLastError());

  // Now need to squash the output so it's contiguous.
  // get prefix sum of num_candidates_out
  // Note: Still need lengths
  auto output_offsets = num_candidates_out.cumsum(0);
  auto total_outputs = output_offsets[output_offsets.numel()-1].item<long>();
  output_offsets = output_offsets - num_candidates_out;

  // allocate final outputs
  at::Tensor squashed_scores = torch::empty({total_outputs}, torch::CUDA(at::kFloat));
  at::Tensor squashed_bboxes = torch::empty({total_outputs, 4}, torch::CUDA(at::kFloat));
  at::Tensor squashed_labels = torch::empty({total_outputs}, torch::CUDA(at::kLong));

  // Copy non-squashed outputs -> squashed.
  squash_outputs<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(N,
                                                               num_candidates_out.data<long>(),
                                                               score_offsets.data<int>(),
                                                               score_out.data<float>(),
                                                               (float4*)bbox_out.data<float>(),
                                                               label_out.data<long>(),
                                                               output_offsets.contiguous().data<long>(),
                                                               squashed_scores.data<float>(),
                                                               (float4*)squashed_bboxes.data<float>(),
                                                               squashed_labels.data<long>());
  THCudaCheck(cudaGetLastError());

  return {squashed_bboxes, squashed_scores, squashed_labels};
}
















