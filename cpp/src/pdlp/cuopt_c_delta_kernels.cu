/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Device-side helpers for the cuOpt delta C API. Hosts the kernels and
// thrust calls that cannot live in cuopt_c_delta.cpp (which is plain C++,
// not CUDA-compiled).
//
// Two responsibilities here:
//   1. set_objective_coefficients_device: thrust::scatter helper invoked by
//      apply_set_objective_op in cuopt_c_delta.cpp at resolve time. The
//      scatter kernel needs nvcc, so it lives in this TU.
//   2. compact_warm_start_vector_device: in-place device-side compaction of
//      the persisted PDLP warm-start vectors via thrust::stable_partition,
//      replacing the per-vector D2H/H2D round-trips in the original delete
//      path.

#include <cuopt/linear_programming/cuopt_c_delta.h>

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <pdlp/cuopt_c_internal.hpp>

#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>

namespace cuopt::linear_programming {

// Stage `indices` and `values` onto the device and scatter them into
// op->get_objective_coefficients() in a single kernel. Caller must already
// have validated that every index is in [0, n_vars). Synchronises `stream`
// before returning (so the host-side payload buffers can be freed).
void set_objective_coefficients_device(
  optimization_problem_t<cuopt_int_t, cuopt_float_t>& op,
  const cuopt_int_t* indices,
  const cuopt_float_t* values,
  cuopt_int_t num_indices,
  rmm::cuda_stream_view stream)
{
  if (num_indices == 0) { return; }

  rmm::device_uvector<cuopt_int_t> d_indices(static_cast<std::size_t>(num_indices), stream);
  rmm::device_uvector<cuopt_float_t> d_values(static_cast<std::size_t>(num_indices), stream);
  raft::copy(d_indices.data(), indices, static_cast<std::size_t>(num_indices), stream);
  raft::copy(d_values.data(), values, static_cast<std::size_t>(num_indices), stream);

  auto& d_obj = op.get_objective_coefficients();
  auto policy = rmm::exec_policy(stream);
  thrust::scatter(
    policy, d_values.begin(), d_values.end(), d_indices.begin(), d_obj.begin());

  stream.synchronize();
}

// Compact `v` device-direct: keep entries i where mask[i] == 0; preserves
// relative order. Caller must guarantee old_size == v.size() before the
// call.
//
// In-place via thrust::stable_partition with a counting-iterator stencil:
// surviving entries (mask[i] == 0) are moved to the front of v, dropped
// entries to the tail. We then resize() v down to new_size, releasing the
// tail without copying. This avoids the cudaMalloc + cudaFree pair that
// the previous out-of-place thrust::copy_if path paid per call (5 vectors
// per cuOptDeleteColumns / cuOptDeleteRows on PDLP, so up to 5 alloc+free
// per warm-start compaction). RMM pool helps but stable_partition still
// wins on cudaMalloc-backed allocators.
void compact_warm_start_vector_device(rmm::device_uvector<cuopt_float_t>& v,
                                      const rmm::device_uvector<cuopt_int_t>& d_mask,
                                      cuopt_int_t old_size,
                                      cuopt_int_t new_size,
                                      rmm::cuda_stream_view stream)
{
  if (old_size == 0) { return; }
  if (new_size == old_size) { return; }  // No-op shortcut.

  auto policy        = rmm::exec_policy(stream);
  const auto* mask_p = d_mask.data();
  thrust::stable_partition(policy,
                           v.begin(),
                           v.begin() + static_cast<std::ptrdiff_t>(old_size),
                           thrust::counting_iterator<cuopt_int_t>(0),
                           [mask_p] __device__(cuopt_int_t i) { return mask_p[i] == 0; });
  v.resize(static_cast<std::size_t>(new_size), stream);
}

}  // namespace cuopt::linear_programming
