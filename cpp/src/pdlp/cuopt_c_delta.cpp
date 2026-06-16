/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/cuopt_c_delta.h>

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_solution.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <pdlp/cuopt_c_internal.hpp>
#include <pdlp/dual_simplex_warm_state.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

/*
 * cuOpt delta C API implementation.
 *
 * Lazy-rebuild architecture:
 *
 *   - Each mutator (cuOptAddColumns / cuOptAddRows / cuOptDeleteColumns /
 *     cuOptDeleteRows / cuOptSetObjectiveCoefficients) does NO GPU work.
 *     It validates inputs against logical (post-pending) sizes, deep-copies
 *     them into a host-side payload, appends to view->pending.log, and
 *     returns.
 *   - cuOptResolve drains the buffer via apply_pending_mutations() right
 *     before delegating to solve_lp. Each apply_*_op helper holds the
 *     existing eager-path code (device-direct AddRows path, host-roundtrip
 *     AddColumns matrix path, thrust::copy_if compaction, thrust::scatter
 *     for SetObjective) plus the per-op PDLP warm-start pad/compact.
 *   - First resolve with no mutations: apply_pending_mutations is a no-op.
 *
 * Coalescing of pending operations (e.g. AddColumn+DeleteColumn → drop) is
 * intentionally out of scope here — straight replay in arrival order.
 *
 * Public-API contract: cuOptGetNum* and other getters reflect the
 * last-resolved state of op_problem, not pending mutations. Between a mutator
 * and the next cuOptResolve, op_problem is "stale". Documented in
 * cuopt_c_delta.h.
 *
 * Solve-pipeline routing inside cuOptResolve (uniform across methods):
 *
 * - apply_pending_mutations drains the staged log into the device-resident
 *   op_problem (CSR), then cuOptResolve issues a single solve_lp(*op,
 *   settings) for every method (barrier / PDLP / dual-simplex). There is no
 *   per-method dispatch and no cached host user_problem: solve_lp ->
 *   solve_lp_with_method routes barrier through run_barrier internally.
 * - The third-party presolve passed to solve_lp is forced to presolver_t::None
 *   on every resolve for every method, so solve_lp never (re-)presolves. The
 *   delta path therefore always solves the original (outer-unpresolved) problem.
 *   An EXPLICIT PSLP or Papilo presolver request is rejected with
 *   CUOPT_INVALID_ARGUMENT; Default / None proceed silently.
 *   Skipping solve_lp's presolve also skips the sort_csr that runs with it,
 *   which relies on the delta mutators producing sorted CSR (see cuOptResolve).
 */

using cuopt::linear_programming::add_columns_payload_t;
using cuopt::linear_programming::add_rows_payload_t;
using cuopt::linear_programming::delete_columns_payload_t;
using cuopt::linear_programming::delete_rows_payload_t;
using cuopt::linear_programming::get_settings_handle;
using cuopt::linear_programming::optimization_problem_solution_t;
using cuopt::linear_programming::optimization_problem_t;
using cuopt::linear_programming::pdlp_warm_start_data_t;
using cuopt::linear_programming::pending_mutation_t;
using cuopt::linear_programming::problem_and_stream_view_t;
using cuopt::linear_programming::set_objective_payload_t;
using cuopt::linear_programming::solution_and_stream_view_t;
using cuopt::linear_programming::var_t;

namespace {

// ----------------------------------------------------------------------------
// View / payload helpers
// ----------------------------------------------------------------------------

problem_and_stream_view_t* get_view(cuOptOptimizationProblem problem)
{
  return static_cast<problem_and_stream_view_t*>(problem);
}

// Seed view->pending.logical_n_{vars,constraints} from op_problem the first
// time we hand the buffer back to a mutator. After the first mutation the
// logical sizes are maintained incrementally; after each apply they are
// reset to op_problem state. Idempotent for the common "no pending log"
// case so we can call it unconditionally at mutator entry.
void ensure_pending_initialized(problem_and_stream_view_t& view)
{
  if (view.pending.dirty) { return; }
  if (view.get_gpu_problem() == nullptr) { return; }
  view.pending.logical_n_vars        = view.get_gpu_problem()->get_n_variables();
  view.pending.logical_n_constraints = view.get_gpu_problem()->get_n_constraints();
}

// ----------------------------------------------------------------------------
// Below: shared helpers used by the apply_*_op functions.  All of these still
// touch the GPU.  None of them are called from the public mutators.
// ----------------------------------------------------------------------------

rmm::cuda_stream_view get_stream(problem_and_stream_view_t& view) { return *view.stream_view_ptr; }

// Convert row_types + rhs (the non-ranged representation produced by
// cuOptCreateProblem) to constraint_lower_bounds / constraint_upper_bounds and
// drop the row_types+rhs fields via optimization_problem_t::clear_row_types /
// clear_constraint_bounds (resize-to-empty, leaving n_constraints_ intact). The
// delta API advertises ranged bounds through cuOptAddRows, so we normalise on
// first mutation and keep a single representation alive thereafter.
void ensure_ranged_representation(optimization_problem_t<cuopt_int_t, cuopt_float_t>& op,
                                  rmm::cuda_stream_view stream)
{
  const auto n                 = op.get_n_constraints();
  const bool ranged_already_ok = op.get_constraint_lower_bounds().size() ==
                                   static_cast<std::size_t>(n) &&
                                 op.get_constraint_upper_bounds().size() == static_cast<std::size_t>(n);
  const bool row_types_present   = !op.get_row_types().is_empty();
  const bool cstr_bounds_present = !op.get_constraint_bounds().is_empty();

  if (ranged_already_ok && !row_types_present && !cstr_bounds_present) {
    // Ranged already and no stale non-ranged arrays to drop.
    return;
  }

  std::vector<cuopt_float_t> lower(static_cast<std::size_t>(n));
  std::vector<cuopt_float_t> upper(static_cast<std::size_t>(n));
  const cuopt_float_t inf = std::numeric_limits<cuopt_float_t>::infinity();

  if (n > 0) {
    if (ranged_already_ok) {
      lower = cuopt::host_copy(op.get_constraint_lower_bounds(), stream);
      upper = cuopt::host_copy(op.get_constraint_upper_bounds(), stream);
      stream.synchronize();
    } else {
      auto row_types_h = cuopt::host_copy(op.get_row_types(), stream);
      auto rhs_h       = cuopt::host_copy(op.get_constraint_bounds(), stream);
      stream.synchronize();
      for (cuopt_int_t i = 0; i < n; ++i) {
        const char s = row_types_h.empty() ? 'E' : row_types_h[static_cast<std::size_t>(i)];
        const auto b = rhs_h.empty() ? cuopt_float_t{0} : rhs_h[static_cast<std::size_t>(i)];
        switch (s) {
          case 'L':
            lower[i] = -inf;
            upper[i] = b;
            break;
          case 'G':
            lower[i] = b;
            upper[i] = inf;
            break;
          case 'E':
            lower[i] = b;
            upper[i] = b;
            break;
          default:
            // Mirror upstream's problem_checking_t: unknown row sense is a
            // caller error, not silently reinterpretable as equality.
            cuopt_expects(false,
                          cuopt::error_type_t::ValidationError,
                          "Unknown row sense '%c' in ensure_ranged_representation; expected one "
                          "of 'E', 'L', 'G'.",
                          s);
        }
      }
    }
  }

  // Drop the redundant non-ranged representation if present (resize-to-empty;
  // leaves n_constraints_ untouched), then push the ranged bounds.
  if (row_types_present) { op.clear_row_types(); }
  if (cstr_bounds_present) { op.clear_constraint_bounds(); }
  if (n > 0) {
    op.set_constraint_lower_bounds(lower.data(), n);
    op.set_constraint_upper_bounds(upper.data(), n);
  } else {
    cuopt_float_t dummy = cuopt_float_t{0};
    op.set_constraint_lower_bounds(&dummy, static_cast<cuopt_int_t>(0));
    op.set_constraint_upper_bounds(&dummy, static_cast<cuopt_int_t>(0));
  }
}

// Many delta operations are easier to express on host-side mirrors of the
// problem, then re-upload wholesale. This helper captures the shared "pull
// everything back" step.
struct host_mirror_t {
  cuopt_int_t n_vars{0};
  cuopt_int_t n_constraints{0};
  // CSR matrix.
  std::vector<cuopt_float_t> a_values;
  std::vector<cuopt_int_t> a_indices;
  std::vector<cuopt_int_t> a_offsets;
  // Constraint bounds (ranged).
  std::vector<cuopt_float_t> c_lower;
  std::vector<cuopt_float_t> c_upper;
  // Variable side.
  std::vector<cuopt_float_t> obj;
  std::vector<cuopt_float_t> v_lower;
  std::vector<cuopt_float_t> v_upper;
  std::vector<var_t> v_types;
};

host_mirror_t pull_to_host(optimization_problem_t<cuopt_int_t, cuopt_float_t>& op,
                           rmm::cuda_stream_view stream)
{
  ensure_ranged_representation(op, stream);

  host_mirror_t m;
  m.n_vars        = op.get_n_variables();
  m.n_constraints = op.get_n_constraints();
  m.a_values      = cuopt::host_copy(op.get_constraint_matrix_values(), stream);
  m.a_indices     = cuopt::host_copy(op.get_constraint_matrix_indices(), stream);
  m.a_offsets     = cuopt::host_copy(op.get_constraint_matrix_offsets(), stream);
  m.c_lower       = cuopt::host_copy(op.get_constraint_lower_bounds(), stream);
  m.c_upper       = cuopt::host_copy(op.get_constraint_upper_bounds(), stream);
  m.obj           = cuopt::host_copy(op.get_objective_coefficients(), stream);
  m.v_lower       = cuopt::host_copy(op.get_variable_lower_bounds(), stream);
  m.v_upper       = cuopt::host_copy(op.get_variable_upper_bounds(), stream);
  m.v_types       = cuopt::host_copy(op.get_variable_types(), stream);
  stream.synchronize();

  // After the first constraint has been set we expect A_offsets to always be
  // size n_constraints + 1. Before anything is set, set_csr_constraint_matrix
  // may not have been called yet.
  if (m.a_offsets.empty()) {
    m.a_offsets.assign(static_cast<std::size_t>(m.n_constraints + 1), 0);
  }
  return m;
}

// Pad a single device_uvector by `num_appended` zeros at the tail. resize()
// preserves the existing prefix and leaves the new tail uninitialized; we
// zero it directly on device with cudaMemsetAsync (no host alloc, no copy).
void pad_device_vector_with_zeros(rmm::device_uvector<cuopt_float_t>& v,
                                  std::size_t num_appended,
                                  rmm::cuda_stream_view stream)
{
  if (num_appended == 0) { return; }
  const std::size_t old_size = v.size();
  v.resize(old_size + num_appended, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    v.data() + old_size, 0, num_appended * sizeof(cuopt_float_t), stream.value()));
}

// Upload the host mask buffer to device for downstream thrust::copy_if. The
// mask convention is the cuOptDelete{Columns,Rows} one: 0 == keep, 1 == drop.
// The same device buffer is reused for every warm-start vector compaction
// driven by the same delete call, so the H2D copy happens once per delete
// (not once per vector).
rmm::device_uvector<cuopt_int_t> upload_mask(const cuopt_int_t* mask,
                                             cuopt_int_t size,
                                             rmm::cuda_stream_view stream)
{
  rmm::device_uvector<cuopt_int_t> d_mask(static_cast<std::size_t>(size), stream);
  if (size > 0) {
    raft::copy(d_mask.data(), mask, static_cast<std::size_t>(size), stream);
  }
  return d_mask;
}

// Build a 0/1 mask of length `size` from a sparse list of indices to drop.
// Returns std::nullopt on invalid input (unsorted, duplicate, or out-of-range)
// so the caller can return CUOPT_INVALID_ARGUMENT. The input contract for
// cuOptDelete{Columns,Rows} is sorted strictly ascending unique indices in
// [0, size) — same shape both functions.
std::optional<std::vector<cuopt_int_t>> build_mask_from_sparse_indices(
  cuopt_int_t num_indices, const cuopt_int_t* indices, cuopt_int_t size)
{
  std::vector<cuopt_int_t> mask(static_cast<std::size_t>(size), cuopt_int_t{0});
  cuopt_int_t prev = -1;
  for (cuopt_int_t i = 0; i < num_indices; ++i) {
    const cuopt_int_t idx = indices[i];
    if (idx <= prev || idx < 0 || idx >= size) { return std::nullopt; }
    mask[static_cast<std::size_t>(idx)] = 1;
    prev                                = idx;
  }
  return mask;
}

// Zero-pad the primal-side warm-start vectors (all sized n_vars) by
// `num_new_columns` trailing zeros. Used on AddColumns apply. No-op if the
// warm start is not yet populated -- first PDLP resolve will seed it.
void pad_warm_start_for_columns(problem_and_stream_view_t& view, cuopt_int_t num_new_columns)
{
  if (!view.pdlp_warm_start.has_value() || num_new_columns <= 0) { return; }
  auto& ws          = *view.pdlp_warm_start;
  const auto stream = *view.stream_view_ptr;
  const auto n_add  = static_cast<std::size_t>(num_new_columns);
  pad_device_vector_with_zeros(ws.current_primal_solution_, n_add, stream);
  pad_device_vector_with_zeros(ws.initial_primal_average_, n_add, stream);
  pad_device_vector_with_zeros(ws.current_ATY_, n_add, stream);
  pad_device_vector_with_zeros(ws.sum_primal_solutions_, n_add, stream);
  pad_device_vector_with_zeros(ws.last_restart_duality_gap_primal_solution_, n_add, stream);
  stream.synchronize();
}

// Zero-pad the dual-side warm-start vectors (all sized n_constraints) by
// `num_new_rows` trailing zeros. Used on AddRows apply.
//
// Also invalidates ws.current_ATY_. A^T y depends on every row of A, so any
// change to the row count means existing entries of A^T y are also stale, not
// just the tail. Zeroing is the cheapest safe reset: PDLP will recompute from
// the (potentially still-useful) primal iterate, which is better than
// propagating a silently-wrong gradient into the first PDHG step. Same reason
// we clear it on row-deletion below.
void invalidate_current_aty(problem_and_stream_view_t& view)
{
  if (!view.pdlp_warm_start.has_value()) { return; }
  auto& ws          = *view.pdlp_warm_start;
  const auto stream = *view.stream_view_ptr;
  if (ws.current_ATY_.size() == 0) { return; }
  RAFT_CUDA_TRY(cudaMemsetAsync(
    ws.current_ATY_.data(), 0, ws.current_ATY_.size() * sizeof(cuopt_float_t), stream.value()));
  stream.synchronize();
}

void pad_warm_start_for_rows(problem_and_stream_view_t& view, cuopt_int_t num_new_rows)
{
  if (!view.pdlp_warm_start.has_value() || num_new_rows <= 0) { return; }
  auto& ws          = *view.pdlp_warm_start;
  const auto stream = *view.stream_view_ptr;
  const auto n_add  = static_cast<std::size_t>(num_new_rows);
  pad_device_vector_with_zeros(ws.current_dual_solution_, n_add, stream);
  pad_device_vector_with_zeros(ws.initial_dual_average_, n_add, stream);
  pad_device_vector_with_zeros(ws.sum_dual_solutions_, n_add, stream);
  pad_device_vector_with_zeros(ws.last_restart_duality_gap_dual_solution_, n_add, stream);
  stream.synchronize();
  // current_ATY_ invalidation is performed once per drain by
  // apply_pending_mutations after the loop, not per row mutation.
}

// Compact the primal-side warm-start vectors per `d_mask` (length
// old_n_vars). The mask must already be on device — callers upload it once
// per delete call so the H2D copy is paid once instead of per vector.
void compact_warm_start_columns(problem_and_stream_view_t& view,
                                const rmm::device_uvector<cuopt_int_t>& d_mask,
                                cuopt_int_t old_n_vars,
                                cuopt_int_t new_n_vars)
{
  if (!view.pdlp_warm_start.has_value() || old_n_vars == 0) { return; }
  auto& ws          = *view.pdlp_warm_start;
  const auto stream = *view.stream_view_ptr;
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.current_primal_solution_, d_mask, old_n_vars, new_n_vars, stream);
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.initial_primal_average_, d_mask, old_n_vars, new_n_vars, stream);
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.current_ATY_, d_mask, old_n_vars, new_n_vars, stream);
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.sum_primal_solutions_, d_mask, old_n_vars, new_n_vars, stream);
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.last_restart_duality_gap_primal_solution_, d_mask, old_n_vars, new_n_vars, stream);
  stream.synchronize();
}

// Compact the dual-side warm-start vectors per `d_mask` (length old_n_rows).
// current_ATY_ invalidation is hoisted to apply_pending_mutations: see the
// post-loop call there.
void compact_warm_start_rows(problem_and_stream_view_t& view,
                             const rmm::device_uvector<cuopt_int_t>& d_mask,
                             cuopt_int_t old_n_rows,
                             cuopt_int_t new_n_rows)
{
  if (!view.pdlp_warm_start.has_value() || old_n_rows == 0) { return; }
  auto& ws          = *view.pdlp_warm_start;
  const auto stream = *view.stream_view_ptr;
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.current_dual_solution_, d_mask, old_n_rows, new_n_rows, stream);
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.initial_dual_average_, d_mask, old_n_rows, new_n_rows, stream);
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.sum_dual_solutions_, d_mask, old_n_rows, new_n_rows, stream);
  cuopt::linear_programming::compact_warm_start_vector_device(
    ws.last_restart_duality_gap_dual_solution_, d_mask, old_n_rows, new_n_rows, stream);
  stream.synchronize();
}

void push_from_host(optimization_problem_t<cuopt_int_t, cuopt_float_t>& op, const host_mirror_t& m)
{
  // set_* helpers each call resize + raft::copy and also update n_vars_ /
  // n_constraints_ from their size arguments.
  op.set_csr_constraint_matrix(m.a_values.data(),
                               static_cast<cuopt_int_t>(m.a_values.size()),
                               m.a_indices.data(),
                               static_cast<cuopt_int_t>(m.a_indices.size()),
                               m.a_offsets.data(),
                               static_cast<cuopt_int_t>(m.a_offsets.size()));
  op.set_constraint_lower_bounds(m.c_lower.data(), m.n_constraints);
  op.set_constraint_upper_bounds(m.c_upper.data(), m.n_constraints);
  op.set_objective_coefficients(m.obj.data(), m.n_vars);
  op.set_variable_lower_bounds(m.v_lower.data(), m.n_vars);
  op.set_variable_upper_bounds(m.v_upper.data(), m.n_vars);
  op.set_variable_types(m.v_types.data(), m.n_vars);
}

// ----------------------------------------------------------------------------
// apply_*_op helpers — the real work of each mutator. Each consumes a payload
// captured by the public C function. Throws cuopt::logic_error on invalid
// device-time input, std::exception on lower-level errors. Caller maps to
// CUOPT_INVALID_ARGUMENT / CUOPT_RUNTIME_ERROR.
// ----------------------------------------------------------------------------

void apply_add_columns_op(problem_and_stream_view_t& view, const add_columns_payload_t& p)
{
  auto* op    = view.get_gpu_problem();
  auto stream = get_stream(view);

  // Adding columns to a CSR matrix is fundamentally a rewrite: every
  // existing row potentially gains new entries (sparse fill across rows),
  // and A_offsets must be re-prefix-summed. There is no architectural
  // shortcut without a CSC mirror or a deferred-pending-appends buffer.
  //
  // Non-matrix fields (obj, v_lower, v_upper, v_types) ARE pure tail
  // appends and are handled device-direct below.
  ensure_ranged_representation(*op, stream);

  const cuopt_int_t n_rows    = op->get_n_constraints();
  const cuopt_int_t old_n_var = op->get_n_variables();
  const cuopt_int_t num_columns = p.num_columns;
  const cuopt_int_t new_n_var = old_n_var + num_columns;
  const cuopt_int_t new_nnz =
    p.column_starts.empty() ? 0 : p.column_starts.back();

  // -- Matrix path: pull A_* to host, rewrite, push back. --
  auto a_values_h  = cuopt::host_copy(op->get_constraint_matrix_values(), stream);
  auto a_indices_h = cuopt::host_copy(op->get_constraint_matrix_indices(), stream);
  auto a_offsets_h = cuopt::host_copy(op->get_constraint_matrix_offsets(), stream);
  stream.synchronize();

  if (a_offsets_h.empty()) {
    a_offsets_h.assign(static_cast<std::size_t>(n_rows + 1), 0);
  }

  // Per-row count of new entries. Cap row-index validation at apply time
  // against the ACTUAL device n_rows since the apply order may have grown
  // the row count via earlier AddRows ops in this same drain.
  std::vector<cuopt_int_t> new_per_row(static_cast<std::size_t>(n_rows), 0);
  for (cuopt_int_t k = 0; k < new_nnz; ++k) {
    const cuopt_int_t r = p.row_indices[static_cast<std::size_t>(k)];
    if (r < 0 || r >= n_rows) {
      throw cuopt::logic_error(std::string("cuOptAddColumns: row index out of range at apply time"), cuopt::error_type_t::ValidationError);
    }
    new_per_row[static_cast<std::size_t>(r)]++;
  }

  const std::size_t old_nnz = a_values_h.size();
  std::vector<cuopt_float_t> new_values;
  std::vector<cuopt_int_t> new_indices;
  new_values.reserve(old_nnz + static_cast<std::size_t>(new_nnz));
  new_indices.reserve(old_nnz + static_cast<std::size_t>(new_nnz));
  std::vector<cuopt_int_t> new_offsets(static_cast<std::size_t>(n_rows + 1), 0);

  // Group new (column-index, value) pairs by row using a flat scratch buffer
  // backed by a single allocation, instead of vector<vector<pair>>(n_rows).
  // For BUS-2632 hundreds-of-thousands-of-rows instances the per-row vector
  // ctors dominate even when only a handful of new columns is added in this
  // call.
  //
  // Layout: bucket_starts[r] (size n_rows + 1) is the prefix-sum of
  // new_per_row, so the appended entries for row r occupy
  // [bucket_starts[r], bucket_starts[r+1]) in bucket_cols / bucket_vals.
  // We scatter columns once with a per-row write cursor (write_cursor[r]
  // counts entries already placed for row r). After scatter, write_cursor
  // is fully consumed so we don't need to reset it.
  std::vector<cuopt_int_t> bucket_starts(static_cast<std::size_t>(n_rows + 1), 0);
  for (cuopt_int_t r = 0; r < n_rows; ++r) {
    bucket_starts[static_cast<std::size_t>(r + 1)] =
      bucket_starts[static_cast<std::size_t>(r)] + new_per_row[static_cast<std::size_t>(r)];
  }
  std::vector<cuopt_int_t> bucket_cols(static_cast<std::size_t>(new_nnz));
  std::vector<cuopt_float_t> bucket_vals(static_cast<std::size_t>(new_nnz));
  std::vector<cuopt_int_t> write_cursor(static_cast<std::size_t>(n_rows), 0);
  for (cuopt_int_t j = 0; j < num_columns; ++j) {
    const cuopt_int_t col_idx = old_n_var + j;
    const cuopt_int_t start   = p.column_starts[static_cast<std::size_t>(j)];
    const cuopt_int_t end     = p.column_starts[static_cast<std::size_t>(j + 1)];
    for (cuopt_int_t k = start; k < end; ++k) {
      const cuopt_int_t r = p.row_indices[static_cast<std::size_t>(k)];
      const auto pos = static_cast<std::size_t>(
        bucket_starts[static_cast<std::size_t>(r)] + write_cursor[static_cast<std::size_t>(r)]);
      bucket_cols[pos] = col_idx;
      bucket_vals[pos] = p.values[static_cast<std::size_t>(k)];
      write_cursor[static_cast<std::size_t>(r)]++;
    }
  }

  for (cuopt_int_t r = 0; r < n_rows; ++r) {
    new_offsets[static_cast<std::size_t>(r)] = static_cast<cuopt_int_t>(new_values.size());

    const cuopt_int_t row_start = a_offsets_h[static_cast<std::size_t>(r)];
    const cuopt_int_t row_end   = a_offsets_h[static_cast<std::size_t>(r + 1)];
    for (cuopt_int_t k = row_start; k < row_end; ++k) {
      new_values.push_back(a_values_h[static_cast<std::size_t>(k)]);
      new_indices.push_back(a_indices_h[static_cast<std::size_t>(k)]);
    }
    // Appended entries already have strictly ascending column indices
    // because columns are iterated in order j=0..num_columns-1 and col_idx
    // = old_n_var + j, so no sort is required.
    const cuopt_int_t bs = bucket_starts[static_cast<std::size_t>(r)];
    const cuopt_int_t be = bucket_starts[static_cast<std::size_t>(r + 1)];
    for (cuopt_int_t k = bs; k < be; ++k) {
      new_values.push_back(bucket_vals[static_cast<std::size_t>(k)]);
      new_indices.push_back(bucket_cols[static_cast<std::size_t>(k)]);
    }
  }
  new_offsets[static_cast<std::size_t>(n_rows)] = static_cast<cuopt_int_t>(new_values.size());

  op->set_csr_constraint_matrix(new_values.data(),
                                static_cast<cuopt_int_t>(new_values.size()),
                                new_indices.data(),
                                static_cast<cuopt_int_t>(new_indices.size()),
                                new_offsets.data(),
                                static_cast<cuopt_int_t>(new_offsets.size()));

  // -- Non-matrix path: device-direct resize + raft::copy on the tail. --
  auto& d_obj     = op->get_objective_coefficients();
  auto& d_v_lower = op->get_variable_lower_bounds();
  auto& d_v_upper = op->get_variable_upper_bounds();

  auto append_floats = [&](rmm::device_uvector<cuopt_float_t>& v,
                           const cuopt_float_t* src,
                           std::size_t n_add) {
    const std::size_t old_size = v.size();
    v.resize(old_size + n_add, stream);
    raft::copy(v.data() + old_size, src, n_add, stream);
  };
  const std::size_t n_add = static_cast<std::size_t>(num_columns);
  append_floats(d_obj, p.objective_coefficients.data(), n_add);
  append_floats(d_v_lower, p.variable_lower_bounds.data(), n_add);
  append_floats(d_v_upper, p.variable_upper_bounds.data(), n_add);

  // Variable types: build a typed buffer host-side (one byte per var) then
  // async-copy it into the resized device tail. Mirror cuOptCreateProblem
  // semantics: CONTINUOUS only on exact match, INTEGER otherwise. Empty
  // payload.variable_types means caller passed NULL — default to CONTINUOUS.
  std::vector<var_t> v_types_new(n_add);
  for (cuopt_int_t j = 0; j < num_columns; ++j) {
    const char vt = p.variable_types.empty()
                      ? CUOPT_CONTINUOUS
                      : p.variable_types[static_cast<std::size_t>(j)];
    v_types_new[static_cast<std::size_t>(j)] =
      vt == CUOPT_CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;
  }
  // get_variable_types() exposes only a const reference; we need the
  // mutable handle. The underlying device_uvector is owned by op, so
  // const_cast'ing the reference is well-defined.
  auto& d_v_types = const_cast<rmm::device_uvector<var_t>&>(op->get_variable_types());
  {
    const std::size_t old_size = d_v_types.size();
    d_v_types.resize(old_size + n_add, stream);
    raft::copy(d_v_types.data() + old_size, v_types_new.data(), n_add, stream);
  }

  op->set_n_variables(new_n_var);

  // NOTE: problem_category_ is intentionally not re-derived here. Appending an
  // INTEGER column to an LP-category problem leaves the category LP (asymmetric
  // with the delete path, which recomputes it). This is benign for the delta
  // API's only solve targets: the barrier / PDLP paths solve the continuous
  // relaxation and ignore integrality, and mcfcg only ever appends continuous
  // columns. Revisit if a MIP consumer ever drives delta resolves.

  // Extend persisted PDLP warm-start primal iterates by zeros for the new
  // columns. No-op if no warm start exists yet (first resolve will seed it).
  pad_warm_start_for_columns(view, num_columns);

  // Single sync at end: callers expect host buffers (and v_types_new
  // here) to be reusable on return.
  stream.synchronize();
}

void apply_add_rows_op(problem_and_stream_view_t& view, const add_rows_payload_t& p)
{
  auto* op    = view.get_gpu_problem();
  auto stream = get_stream(view);

  // Adding rows is a pure tail-append on the CSR matrix and constraint-bound
  // arrays. Existing rows do not need to be rewritten:
  //   - A_values / A_indices: append `new_nnz` entries from the user buffers.
  //   - A_offsets: append `num_rows` entries, each = old_nnz_total +
  //     row_starts[r+1]. The first old_n_rows + 1 entries (including the
  //     terminator at A_offsets[old_n_rows] == old_nnz_total) are unchanged.
  //   - constraint_lower_bounds / constraint_upper_bounds: append num_rows.
  //
  // We resize each device_uvector in place (preserves prefix, leaves tail
  // uninitialized) and async H2D-copy directly into the new tail. One
  // stream.synchronize() at the end fences against the host buffers
  // returning to the caller (raft::copy is async on `stream`).
  //
  // The translation from row_types+rhs to ranged bounds only matters when
  // the problem was created via cuOptCreateProblem; we keep the existing
  // ensure_ranged_representation helper for that one-time conversion.
  ensure_ranged_representation(*op, stream);

  const cuopt_int_t old_n_rows = op->get_n_constraints();
  const cuopt_int_t old_n_vars = op->get_n_variables();
  const cuopt_int_t num_rows   = p.num_rows;
  const cuopt_int_t new_n_rows = old_n_rows + num_rows;
  const cuopt_int_t new_nnz =
    p.row_starts.empty() ? 0 : p.row_starts.back();

  // Validate column indices against the current (post-prior-apply) device
  // n_vars. Required at apply time because previous AddColumns ops in this
  // same drain may have grown n_vars beyond what the AddRows mutator saw.
  for (cuopt_int_t k = 0; k < new_nnz; ++k) {
    const cuopt_int_t c = p.column_indices[static_cast<std::size_t>(k)];
    if (c < 0 || c >= old_n_vars) {
      throw cuopt::logic_error(std::string("cuOptAddRows: column index out of range at apply time"), cuopt::error_type_t::ValidationError);
    }
  }

  auto& a_values  = op->get_constraint_matrix_values();
  auto& a_indices = op->get_constraint_matrix_indices();
  auto& a_offsets = op->get_constraint_matrix_offsets();
  auto& c_lower   = op->get_constraint_lower_bounds();
  auto& c_upper   = op->get_constraint_upper_bounds();

  const std::size_t old_nnz_total = a_values.size();

  // A_offsets pre-mutation should be either of size old_n_rows + 1 with
  // terminator old_nnz_total, or empty (no constraint-matrix rows have
  // been set yet, e.g. synthetic problem). For the empty case we seed the
  // leading 0 entry; existing rows need no other patching. cudaMemsetAsync
  // avoids any host-source-buffer lifetime concern (vs raft::copy(&zero)).
  if (a_offsets.size() == 0) {
    a_offsets.resize(1, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(a_offsets.data(), 0, sizeof(cuopt_int_t), stream.value()));
  }

  // Append new_nnz entries to A_values / A_indices.
  if (new_nnz > 0) {
    const std::size_t old_av_size = a_values.size();
    a_values.resize(old_av_size + static_cast<std::size_t>(new_nnz), stream);
    raft::copy(
      a_values.data() + old_av_size, p.values.data(), static_cast<std::size_t>(new_nnz), stream);

    const std::size_t old_ai_size = a_indices.size();
    a_indices.resize(old_ai_size + static_cast<std::size_t>(new_nnz), stream);
    raft::copy(a_indices.data() + old_ai_size,
               p.column_indices.data(),
               static_cast<std::size_t>(new_nnz),
               stream);
  }

  // Append num_rows shifted row offsets. Build a small host buffer; a
  // tiny on-device shift kernel would also work, but for typical CG
  // iteration sizes num_rows is small and the H2D copy is the dominant
  // term either way. Keep `shifted` alive until the final stream sync
  // so cudaMemcpyAsync can read it safely (raft::copy on pageable host
  // memory stages synchronously, but we don't rely on that).
  std::vector<cuopt_int_t> shifted(static_cast<std::size_t>(num_rows));
  for (cuopt_int_t r = 0; r < num_rows; ++r) {
    shifted[static_cast<std::size_t>(r)] =
      static_cast<cuopt_int_t>(old_nnz_total) + p.row_starts[static_cast<std::size_t>(r + 1)];
  }
  {
    const std::size_t old_off_size = a_offsets.size();
    a_offsets.resize(old_off_size + static_cast<std::size_t>(num_rows), stream);
    raft::copy(a_offsets.data() + old_off_size,
               shifted.data(),
               static_cast<std::size_t>(num_rows),
               stream);
  }

  // Append num_rows constraint bounds.
  {
    const std::size_t old_cl_size = c_lower.size();
    c_lower.resize(old_cl_size + static_cast<std::size_t>(num_rows), stream);
    raft::copy(c_lower.data() + old_cl_size,
               p.constraint_lower_bounds.data(),
               static_cast<std::size_t>(num_rows),
               stream);

    const std::size_t old_cu_size = c_upper.size();
    c_upper.resize(old_cu_size + static_cast<std::size_t>(num_rows), stream);
    raft::copy(c_upper.data() + old_cu_size,
               p.constraint_upper_bounds.data(),
               static_cast<std::size_t>(num_rows),
               stream);
  }

  op->set_n_constraints(new_n_rows);

  // Pad the warm start before the sync so its own H2D zero-fills overlap.
  pad_warm_start_for_rows(view, num_rows);
  stream.synchronize();
}

void apply_delete_columns_op(problem_and_stream_view_t& view, const delete_columns_payload_t& p)
{
  auto* op    = view.get_gpu_problem();
  auto stream = get_stream(view);

  auto m = pull_to_host(*op, stream);

  const cuopt_int_t old_n = m.n_vars;
  if (old_n == 0) { return; }
  const cuopt_int_t num_indices = static_cast<cuopt_int_t>(p.indices.size());
  if (num_indices == 0) { return; }

  auto mask_opt = build_mask_from_sparse_indices(num_indices, p.indices.data(), old_n);
  if (!mask_opt.has_value()) {
    throw cuopt::logic_error(std::string("cuOptDeleteColumns: invalid indices at apply time"), cuopt::error_type_t::ValidationError);
  }
  const auto& mask = *mask_opt;

  // Remap: new_col[j] = index of column j in compacted output, or -1 if dropped.
  std::vector<cuopt_int_t> remap(static_cast<std::size_t>(old_n));
  cuopt_int_t kept = 0;
  for (cuopt_int_t j = 0; j < old_n; ++j) {
    if (mask[j] == 0) {
      remap[static_cast<std::size_t>(j)] = kept++;
    } else {
      remap[static_cast<std::size_t>(j)] = -1;
    }
  }

  // Compact variable-side arrays in place using the remap.
  std::vector<cuopt_float_t> new_obj(static_cast<std::size_t>(kept));
  std::vector<cuopt_float_t> new_vl(static_cast<std::size_t>(kept));
  std::vector<cuopt_float_t> new_vu(static_cast<std::size_t>(kept));
  std::vector<var_t> new_vt(static_cast<std::size_t>(kept));
  for (cuopt_int_t j = 0; j < old_n; ++j) {
    const auto nj = remap[static_cast<std::size_t>(j)];
    if (nj < 0) { continue; }
    new_obj[static_cast<std::size_t>(nj)] = m.obj[static_cast<std::size_t>(j)];
    new_vl[static_cast<std::size_t>(nj)]  = m.v_lower[static_cast<std::size_t>(j)];
    new_vu[static_cast<std::size_t>(nj)]  = m.v_upper[static_cast<std::size_t>(j)];
    new_vt[static_cast<std::size_t>(nj)]  = m.v_types[static_cast<std::size_t>(j)];
  }
  m.obj     = std::move(new_obj);
  m.v_lower = std::move(new_vl);
  m.v_upper = std::move(new_vu);
  m.v_types = std::move(new_vt);
  m.n_vars  = kept;

  // Filter CSR matrix entries whose column survives; rewrite offsets.
  std::vector<cuopt_float_t> new_vals;
  std::vector<cuopt_int_t> new_inds;
  new_vals.reserve(m.a_values.size());
  new_inds.reserve(m.a_indices.size());
  std::vector<cuopt_int_t> new_offsets(static_cast<std::size_t>(m.n_constraints + 1), 0);
  for (cuopt_int_t r = 0; r < m.n_constraints; ++r) {
    new_offsets[static_cast<std::size_t>(r)] = static_cast<cuopt_int_t>(new_vals.size());
    const cuopt_int_t start = m.a_offsets[static_cast<std::size_t>(r)];
    const cuopt_int_t end   = m.a_offsets[static_cast<std::size_t>(r + 1)];
    for (cuopt_int_t k = start; k < end; ++k) {
      const cuopt_int_t old_col = m.a_indices[static_cast<std::size_t>(k)];
      const cuopt_int_t new_col = remap[static_cast<std::size_t>(old_col)];
      if (new_col < 0) { continue; }
      new_vals.push_back(m.a_values[static_cast<std::size_t>(k)]);
      new_inds.push_back(new_col);
    }
  }
  new_offsets[static_cast<std::size_t>(m.n_constraints)] =
    static_cast<cuopt_int_t>(new_vals.size());
  m.a_values  = std::move(new_vals);
  m.a_indices = std::move(new_inds);
  m.a_offsets = std::move(new_offsets);

  push_from_host(*op, m);

  // Compact persisted PDLP warm-start primal iterates in sync with the
  // column compaction we just applied. Upload the mask to device once and
  // share it across all warm-start vectors.
  if (view.pdlp_warm_start.has_value()) {
    auto d_mask = upload_mask(mask.data(), old_n, *view.stream_view_ptr);
    compact_warm_start_columns(view, d_mask, old_n, kept);
  }
}

void apply_delete_rows_op(problem_and_stream_view_t& view, const delete_rows_payload_t& p)
{
  auto* op    = view.get_gpu_problem();
  auto stream = get_stream(view);

  auto m = pull_to_host(*op, stream);

  const cuopt_int_t old_m = m.n_constraints;
  if (old_m == 0) { return; }
  const cuopt_int_t num_indices = static_cast<cuopt_int_t>(p.indices.size());
  if (num_indices == 0) { return; }

  auto mask_opt = build_mask_from_sparse_indices(num_indices, p.indices.data(), old_m);
  if (!mask_opt.has_value()) {
    throw cuopt::logic_error(std::string("cuOptDeleteRows: invalid indices at apply time"), cuopt::error_type_t::ValidationError);
  }
  const auto& mask       = *mask_opt;
  const cuopt_int_t kept = old_m - num_indices;

  std::vector<cuopt_float_t> new_cl(static_cast<std::size_t>(kept));
  std::vector<cuopt_float_t> new_cu(static_cast<std::size_t>(kept));
  std::vector<cuopt_float_t> new_vals;
  std::vector<cuopt_int_t> new_inds;
  new_vals.reserve(m.a_values.size());
  new_inds.reserve(m.a_indices.size());
  std::vector<cuopt_int_t> new_offsets(static_cast<std::size_t>(kept + 1), 0);

  cuopt_int_t out_r = 0;
  for (cuopt_int_t r = 0; r < old_m; ++r) {
    if (mask[r] != 0) { continue; }
    new_cl[static_cast<std::size_t>(out_r)] = m.c_lower[static_cast<std::size_t>(r)];
    new_cu[static_cast<std::size_t>(out_r)] = m.c_upper[static_cast<std::size_t>(r)];
    new_offsets[static_cast<std::size_t>(out_r)] =
      static_cast<cuopt_int_t>(new_vals.size());
    const cuopt_int_t start = m.a_offsets[static_cast<std::size_t>(r)];
    const cuopt_int_t end   = m.a_offsets[static_cast<std::size_t>(r + 1)];
    for (cuopt_int_t k = start; k < end; ++k) {
      new_vals.push_back(m.a_values[static_cast<std::size_t>(k)]);
      new_inds.push_back(m.a_indices[static_cast<std::size_t>(k)]);
    }
    ++out_r;
  }
  new_offsets[static_cast<std::size_t>(kept)] = static_cast<cuopt_int_t>(new_vals.size());

  m.c_lower       = std::move(new_cl);
  m.c_upper       = std::move(new_cu);
  m.a_values      = std::move(new_vals);
  m.a_indices     = std::move(new_inds);
  m.a_offsets     = std::move(new_offsets);
  m.n_constraints = kept;

  push_from_host(*op, m);

  // Compact persisted PDLP warm-start dual iterates in sync with the row
  // compaction we just applied. Upload the mask to device once and share
  // it across all warm-start vectors.
  if (view.pdlp_warm_start.has_value()) {
    auto d_mask = upload_mask(mask.data(), old_m, *view.stream_view_ptr);
    compact_warm_start_rows(view, d_mask, old_m, kept);
  }
}

void apply_set_objective_op(problem_and_stream_view_t& view, const set_objective_payload_t& p)
{
  auto* op    = view.get_gpu_problem();
  auto stream = *view.stream_view_ptr;

  const cuopt_int_t n           = op->get_n_variables();
  const cuopt_int_t num_indices = static_cast<cuopt_int_t>(p.indices.size());
  for (cuopt_int_t i = 0; i < num_indices; ++i) {
    const cuopt_int_t idx = p.indices[static_cast<std::size_t>(i)];
    if (idx < 0 || idx >= n) {
      throw cuopt::logic_error(std::string("cuOptSetObjectiveCoefficients: index out of range at apply time"), cuopt::error_type_t::ValidationError);
    }
  }

  // The thrust::scatter that consumes (indices, values) lives in
  // cuopt_c_delta_kernels.cu so it can be nvcc-compiled. Synchronises
  // `stream` before returning.
  cuopt::linear_programming::set_objective_coefficients_device(
    *op, p.indices.data(), p.values.data(), num_indices, stream);
}

// Concatenate a contiguous run of AddColumns payloads in [first, last) into a
// single super-payload semantically equivalent to applying them sequentially.
//
// Why coalesce: each apply_add_columns_op pays a full pull_to_host of the
// constraint matrix + host bucket-rebuild + push_back. Under mcfcg's CG loop
// the master adds 1-2 columns per iteration via several AddColumns calls
// interleaved with capacity-row additions, and back-to-back AddColumns
// (the common case before the first AddRows of the iter) would otherwise pay
// K full host round-trips of A. After coalescing, K consecutive AddColumns
// pay exactly one round-trip.
//
// Concatenation rules:
//   * num_columns / objective_coefficients / variable_lower_bounds /
//     variable_upper_bounds: simple append.
//   * column_starts: keep the first payload's column_starts as-is, then for
//     each subsequent payload p_i append (p_i.column_starts[k] + base_nnz)
//     for k = 1..p_i.num_columns, where base_nnz is the running merged
//     values.size() before appending p_i. The trailing terminator is
//     therefore the merged total nnz.
//   * row_indices / values: concatenate.
//   * variable_types: if any source payload supplied an explicit types
//     vector, the merged result must materialise types for ALL columns
//     (defaulting absent payload entries to CUOPT_CONTINUOUS). If every
//     source had empty types (caller passed nullptr everywhere), leave
//     the merged vector empty so the apply path still defaults to
//     CONTINUOUS — preserves the no-allocation case.
//
// Index validation: every source payload was already validated at staging
// time against the logical row count (which itself accounts for prior pending
// AddRows / DeleteRows). The merged payload inherits that validation by
// construction — we don't need to re-check.
add_columns_payload_t coalesce_add_columns_run(
  std::vector<pending_mutation_t>::const_iterator first,
  std::vector<pending_mutation_t>::const_iterator last)
{
  // Pre-pass: total sizes for reserve, plus detect whether any payload has
  // explicit variable types.
  cuopt_int_t total_cols = 0;
  std::size_t total_nnz  = 0;
  bool any_types_set     = false;
  for (auto it = first; it != last; ++it) {
    const auto& p = std::get<add_columns_payload_t>(it->payload);
    total_cols += p.num_columns;
    total_nnz +=
      p.column_starts.empty() ? 0 : static_cast<std::size_t>(p.column_starts.back());
    if (!p.variable_types.empty()) { any_types_set = true; }
  }

  add_columns_payload_t merged;
  merged.num_columns = total_cols;
  merged.objective_coefficients.reserve(static_cast<std::size_t>(total_cols));
  merged.variable_lower_bounds.reserve(static_cast<std::size_t>(total_cols));
  merged.variable_upper_bounds.reserve(static_cast<std::size_t>(total_cols));
  merged.column_starts.reserve(static_cast<std::size_t>(total_cols) + 1);
  merged.row_indices.reserve(total_nnz);
  merged.values.reserve(total_nnz);
  if (any_types_set) {
    merged.variable_types.reserve(static_cast<std::size_t>(total_cols));
  }
  merged.column_starts.push_back(0);  // canonical leading 0

  cuopt_int_t base_nnz = 0;  // running sum = merged.values.size() so far.
  for (auto it = first; it != last; ++it) {
    const auto& p = std::get<add_columns_payload_t>(it->payload);
    // Floats: simple append.
    merged.objective_coefficients.insert(
      merged.objective_coefficients.end(),
      p.objective_coefficients.begin(),
      p.objective_coefficients.end());
    merged.variable_lower_bounds.insert(
      merged.variable_lower_bounds.end(),
      p.variable_lower_bounds.begin(),
      p.variable_lower_bounds.end());
    merged.variable_upper_bounds.insert(
      merged.variable_upper_bounds.end(),
      p.variable_upper_bounds.begin(),
      p.variable_upper_bounds.end());
    // CSC matrix: shift-and-append column_starts (skip the leading 0 of each
    // sub-payload — we already pushed the global 0), then concatenate
    // row_indices + values.
    for (cuopt_int_t k = 1; k <= p.num_columns; ++k) {
      merged.column_starts.push_back(p.column_starts[static_cast<std::size_t>(k)] + base_nnz);
    }
    merged.row_indices.insert(merged.row_indices.end(), p.row_indices.begin(), p.row_indices.end());
    merged.values.insert(merged.values.end(), p.values.begin(), p.values.end());
    base_nnz = static_cast<cuopt_int_t>(merged.values.size());

    // variable_types: when any source has explicit types, all columns must
    // get a byte. Default to CUOPT_CONTINUOUS for source payloads that
    // passed nullptr.
    if (any_types_set) {
      if (p.variable_types.empty()) {
        merged.variable_types.insert(
          merged.variable_types.end(), static_cast<std::size_t>(p.num_columns), CUOPT_CONTINUOUS);
      } else {
        merged.variable_types.insert(
          merged.variable_types.end(), p.variable_types.begin(), p.variable_types.end());
      }
    }
  }
  return merged;
}

// Drain view->pending.log against view->op_problem. Each entry is replayed
// in arrival order using the apply_*_op helpers above. Consecutive
// AddColumns runs are coalesced into a single apply call to avoid per-call
// pull_to_host overhead — see coalesce_add_columns_run above. After the
// drain, the log is cleared, dirty=false, and logical sizes are reseeded
// from the (now-current) op_problem.
void apply_pending_mutations(problem_and_stream_view_t& view)
{
  if (!view.pending.dirty) { return; }

  // Tracks whether the drain touched any row of A (add or delete). Used after
  // the loop to trigger the one-shot current_ATY_ invalidation below.
  bool row_mutation = false;
  const auto& log   = view.pending.log;
  try {
  for (auto it = log.begin(); it != log.end();) {
    switch (it->kind) {
      case pending_mutation_t::kind_t::AddColumns: {
        // Greedy run-detection: extend `run_end` while the next entry is
        // also an AddColumns. Single-entry runs degenerate to the
        // direct-apply path (no extra copies) by reusing the source
        // payload's reference.
        auto run_end = it + 1;
        while (run_end != log.end() &&
               run_end->kind == pending_mutation_t::kind_t::AddColumns) {
          ++run_end;
        }
        if (run_end == it + 1) {
          const auto& p = std::get<add_columns_payload_t>(it->payload);
          apply_add_columns_op(view, p);
        } else {
          add_columns_payload_t merged = coalesce_add_columns_run(it, run_end);
          apply_add_columns_op(view, merged);
        }
        it = run_end;
        break;
      }
      case pending_mutation_t::kind_t::AddRows: {
        const auto& p = std::get<add_rows_payload_t>(it->payload);
        apply_add_rows_op(view, p);
        row_mutation = true;
        ++it;
        break;
      }
      case pending_mutation_t::kind_t::DeleteColumns: {
        const auto& p = std::get<delete_columns_payload_t>(it->payload);
        apply_delete_columns_op(view, p);
        ++it;
        break;
      }
      case pending_mutation_t::kind_t::DeleteRows: {
        const auto& p = std::get<delete_rows_payload_t>(it->payload);
        apply_delete_rows_op(view, p);
        row_mutation = true;
        ++it;
        break;
      }
      case pending_mutation_t::kind_t::SetObjective: {
        const auto& p = std::get<set_objective_payload_t>(it->payload);
        apply_set_objective_op(view, p);
        ++it;
        break;
      }
    }
  }
  } catch (...) {
    // A throw mid-drain (OOM on a large rebuild, a lower-level CUDA error)
    // leaves op_problem partially mutated with some ops applied and the log
    // still intact. Re-draining that log on a later resolve would double-apply
    // the already-applied prefix and silently corrupt the handle. Poison it
    // instead: every delta entry point and cuOptResolve will refuse to operate
    // on this handle until the caller destroys and recreates the problem. The
    // partially-mutated op_problem is unrecoverable in place.
    view.pending.poisoned = true;
    throw;
  }

  // current_ATY_ depends on every row of A (it caches A^T y). Any row
  // mutation in the drain — add or delete — invalidates it. Hoisted out of
  // the per-mutation helpers so multiple row mutations in a single drain pay
  // exactly one zero-fill + sync, not one per call.
  if (row_mutation) { invalidate_current_aty(view); }

  view.pending.log.clear();
  view.pending.dirty                 = false;
  view.pending.logical_n_vars        = view.get_gpu_problem()->get_n_variables();
  view.pending.logical_n_constraints = view.get_gpu_problem()->get_n_constraints();
}

}  // namespace

extern "C" {

cuopt_int_t cuOptAddColumns(cuOptOptimizationProblem problem,
                            cuopt_int_t num_columns,
                            const cuopt_float_t* objective_coefficients,
                            const cuopt_float_t* variable_lower_bounds,
                            const cuopt_float_t* variable_upper_bounds,
                            const cuopt_int_t* column_starts,
                            const cuopt_int_t* row_indices,
                            const cuopt_float_t* values,
                            const char* variable_types)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_columns < 0) { return CUOPT_INVALID_ARGUMENT; }
  if (num_columns == 0) { return CUOPT_SUCCESS; }
  if (objective_coefficients == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (variable_lower_bounds == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (variable_upper_bounds == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (column_starts == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  const cuopt_int_t new_nnz = column_starts[num_columns];
  if (new_nnz < 0) { return CUOPT_INVALID_ARGUMENT; }
  if (new_nnz > 0 && (row_indices == nullptr || values == nullptr)) {
    return CUOPT_INVALID_ARGUMENT;
  }
  if (column_starts[0] != 0) { return CUOPT_INVALID_ARGUMENT; }
  for (cuopt_int_t j = 0; j < num_columns; ++j) {
    const cuopt_int_t start = column_starts[j];
    const cuopt_int_t end   = column_starts[j + 1];
    if (end < start || end > new_nnz) { return CUOPT_INVALID_ARGUMENT; }
  }

  try {
    auto* view = get_view(problem);
    if (view->get_gpu_problem() == nullptr) {
      CUOPT_LOG_INFO("cuOpt delta API requires a GPU memory backend");
      return CUOPT_INVALID_ARGUMENT;
    }
    if (view->pending.poisoned) {
      CUOPT_LOG_ERROR(
        "cuOpt delta handle is unusable after a failed cuOptResolve drain; recreate the problem");
      return CUOPT_INVALID_ARGUMENT;
    }
    ensure_pending_initialized(*view);

    // Row-index range checks against the current logical row count, which
    // already accounts for any earlier pending AddRows / DeleteRows. Catching
    // bad indices at staging time keeps the apply path infallible w.r.t.
    // user input (so a partial replay can never poison the handle).
    const cuopt_int_t logical_m = view->pending.logical_n_constraints;
    for (cuopt_int_t k = 0; k < new_nnz; ++k) {
      const cuopt_int_t r = row_indices[k];
      if (r < 0 || r >= logical_m) { return CUOPT_INVALID_ARGUMENT; }
    }

    add_columns_payload_t p;
    p.num_columns           = num_columns;
    p.objective_coefficients.assign(objective_coefficients,
                                    objective_coefficients + num_columns);
    p.variable_lower_bounds.assign(variable_lower_bounds, variable_lower_bounds + num_columns);
    p.variable_upper_bounds.assign(variable_upper_bounds, variable_upper_bounds + num_columns);
    p.column_starts.assign(column_starts, column_starts + num_columns + 1);
    if (new_nnz > 0) {
      p.row_indices.assign(row_indices, row_indices + new_nnz);
      p.values.assign(values, values + new_nnz);
    }
    if (variable_types != nullptr) {
      p.variable_types.assign(variable_types, variable_types + num_columns);
    }  // else leave empty -> apply will default to CUOPT_CONTINUOUS.

    view->pending.log.push_back(
      pending_mutation_t{pending_mutation_t::kind_t::AddColumns, std::move(p)});
    view->pending.logical_n_vars += num_columns;
    view->pending.dirty = true;
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("cuOptAddColumns: %s", e.what());
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptAddRows(cuOptOptimizationProblem problem,
                         cuopt_int_t num_rows,
                         const cuopt_float_t* constraint_lower_bounds,
                         const cuopt_float_t* constraint_upper_bounds,
                         const cuopt_int_t* row_starts,
                         const cuopt_int_t* column_indices,
                         const cuopt_float_t* values)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_rows < 0) { return CUOPT_INVALID_ARGUMENT; }
  if (num_rows == 0) { return CUOPT_SUCCESS; }
  if (constraint_lower_bounds == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (constraint_upper_bounds == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (row_starts == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  const cuopt_int_t new_nnz = row_starts[num_rows];
  if (new_nnz < 0) { return CUOPT_INVALID_ARGUMENT; }
  if (new_nnz > 0 && (column_indices == nullptr || values == nullptr)) {
    return CUOPT_INVALID_ARGUMENT;
  }
  if (row_starts[0] != 0) { return CUOPT_INVALID_ARGUMENT; }
  for (cuopt_int_t r = 0; r < num_rows; ++r) {
    const cuopt_int_t rs_prev = row_starts[r];
    const cuopt_int_t rs_next = row_starts[r + 1];
    if (rs_next < rs_prev || rs_next > new_nnz) { return CUOPT_INVALID_ARGUMENT; }
  }

  try {
    auto* view = get_view(problem);
    if (view->get_gpu_problem() == nullptr) {
      CUOPT_LOG_INFO("cuOpt delta API requires a GPU memory backend");
      return CUOPT_INVALID_ARGUMENT;
    }
    if (view->pending.poisoned) {
      CUOPT_LOG_ERROR(
        "cuOpt delta handle is unusable after a failed cuOptResolve drain; recreate the problem");
      return CUOPT_INVALID_ARGUMENT;
    }
    ensure_pending_initialized(*view);

    // Column-index range checks against the current logical column count.
    // Mirror cuOptAddColumns: catch bad indices at staging time so the apply
    // path never throws on user input.
    const cuopt_int_t logical_n = view->pending.logical_n_vars;
    for (cuopt_int_t k = 0; k < new_nnz; ++k) {
      const cuopt_int_t c = column_indices[k];
      if (c < 0 || c >= logical_n) { return CUOPT_INVALID_ARGUMENT; }
    }

    add_rows_payload_t p;
    p.num_rows = num_rows;
    p.constraint_lower_bounds.assign(constraint_lower_bounds,
                                     constraint_lower_bounds + num_rows);
    p.constraint_upper_bounds.assign(constraint_upper_bounds,
                                     constraint_upper_bounds + num_rows);
    p.row_starts.assign(row_starts, row_starts + num_rows + 1);
    if (new_nnz > 0) {
      p.column_indices.assign(column_indices, column_indices + new_nnz);
      p.values.assign(values, values + new_nnz);
    }

    view->pending.log.push_back(
      pending_mutation_t{pending_mutation_t::kind_t::AddRows, std::move(p)});
    view->pending.logical_n_constraints += num_rows;
    view->pending.dirty = true;
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("cuOptAddRows: %s", e.what());
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptDeleteColumns(cuOptOptimizationProblem problem,
                               cuopt_int_t num_indices,
                               const cuopt_int_t* indices)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices < 0) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices > 0 && indices == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices == 0) { return CUOPT_SUCCESS; }

  try {
    auto* view = get_view(problem);
    if (view->get_gpu_problem() == nullptr) {
      CUOPT_LOG_INFO("cuOpt delta API requires a GPU memory backend");
      return CUOPT_INVALID_ARGUMENT;
    }
    if (view->pending.poisoned) {
      CUOPT_LOG_ERROR(
        "cuOpt delta handle is unusable after a failed cuOptResolve drain; recreate the problem");
      return CUOPT_INVALID_ARGUMENT;
    }
    ensure_pending_initialized(*view);

    // Validate sorted-strictly-ascending against current logical n_vars.
    // num_indices > 0 against an empty problem is a caller bug — index 0 is
    // out of range — so reject rather than silently succeeding.
    const cuopt_int_t logical_n = view->pending.logical_n_vars;
    if (logical_n == 0) { return CUOPT_INVALID_ARGUMENT; }
    cuopt_int_t prev = -1;
    for (cuopt_int_t i = 0; i < num_indices; ++i) {
      const cuopt_int_t idx = indices[i];
      if (idx <= prev || idx < 0 || idx >= logical_n) { return CUOPT_INVALID_ARGUMENT; }
      prev = idx;
    }

    delete_columns_payload_t p;
    p.indices.assign(indices, indices + num_indices);

    view->pending.log.push_back(
      pending_mutation_t{pending_mutation_t::kind_t::DeleteColumns, std::move(p)});
    view->pending.logical_n_vars -= num_indices;
    view->pending.dirty = true;
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("cuOptDeleteColumns: %s", e.what());
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptDeleteRows(cuOptOptimizationProblem problem,
                            cuopt_int_t num_indices,
                            const cuopt_int_t* indices)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices < 0) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices > 0 && indices == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices == 0) { return CUOPT_SUCCESS; }

  try {
    auto* view = get_view(problem);
    if (view->get_gpu_problem() == nullptr) {
      CUOPT_LOG_INFO("cuOpt delta API requires a GPU memory backend");
      return CUOPT_INVALID_ARGUMENT;
    }
    if (view->pending.poisoned) {
      CUOPT_LOG_ERROR(
        "cuOpt delta handle is unusable after a failed cuOptResolve drain; recreate the problem");
      return CUOPT_INVALID_ARGUMENT;
    }
    ensure_pending_initialized(*view);

    // num_indices > 0 against an empty problem is a caller bug — index 0 is
    // out of range — so reject rather than silently succeeding.
    const cuopt_int_t logical_m = view->pending.logical_n_constraints;
    if (logical_m == 0) { return CUOPT_INVALID_ARGUMENT; }
    cuopt_int_t prev = -1;
    for (cuopt_int_t i = 0; i < num_indices; ++i) {
      const cuopt_int_t idx = indices[i];
      if (idx <= prev || idx < 0 || idx >= logical_m) { return CUOPT_INVALID_ARGUMENT; }
      prev = idx;
    }

    delete_rows_payload_t p;
    p.indices.assign(indices, indices + num_indices);

    view->pending.log.push_back(
      pending_mutation_t{pending_mutation_t::kind_t::DeleteRows, std::move(p)});
    view->pending.logical_n_constraints -= num_indices;
    view->pending.dirty = true;
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("cuOptDeleteRows: %s", e.what());
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetObjectiveCoefficients(cuOptOptimizationProblem problem,
                                          cuopt_int_t num_indices,
                                          const cuopt_int_t* indices,
                                          const cuopt_float_t* values)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices < 0) { return CUOPT_INVALID_ARGUMENT; }
  if (num_indices == 0) { return CUOPT_SUCCESS; }
  if (indices == nullptr || values == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  try {
    auto* view = get_view(problem);
    if (view->get_gpu_problem() == nullptr) {
      CUOPT_LOG_INFO("cuOpt delta API requires a GPU memory backend");
      return CUOPT_INVALID_ARGUMENT;
    }
    if (view->pending.poisoned) {
      CUOPT_LOG_ERROR(
        "cuOpt delta handle is unusable after a failed cuOptResolve drain; recreate the problem");
      return CUOPT_INVALID_ARGUMENT;
    }
    ensure_pending_initialized(*view);

    const cuopt_int_t logical_n = view->pending.logical_n_vars;
    for (cuopt_int_t i = 0; i < num_indices; ++i) {
      const cuopt_int_t idx = indices[i];
      if (idx < 0 || idx >= logical_n) { return CUOPT_INVALID_ARGUMENT; }
    }

    set_objective_payload_t p;
    p.indices.assign(indices, indices + num_indices);
    p.values.assign(values, values + num_indices);

    view->pending.log.push_back(
      pending_mutation_t{pending_mutation_t::kind_t::SetObjective, std::move(p)});
    view->pending.dirty = true;
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("cuOptSetObjectiveCoefficients: %s", e.what());
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptResolve(cuOptOptimizationProblem problem,
                         cuOptSolverSettings settings,
                         cuOptSolution* previous_solution_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (settings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (previous_solution_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  try {
    auto* view = static_cast<problem_and_stream_view_t*>(problem);
    auto* op   = view->get_gpu_problem();
    if (op == nullptr) {
      CUOPT_LOG_INFO("cuOptResolve: delta API requires a GPU memory backend");
      return CUOPT_INVALID_ARGUMENT;
    }
    if (view->pending.poisoned) {
      CUOPT_LOG_ERROR(
        "cuOptResolve: delta handle is unusable after a failed drain; recreate the problem");
      return CUOPT_INVALID_ARGUMENT;
    }
    auto* handle        = get_settings_handle(settings);
    auto& pdlp_settings = handle->settings->get_pdlp_settings();

    const bool is_dual_simplex =
      (pdlp_settings.method == cuopt::linear_programming::method_t::DualSimplex);

    // The delta path requires presolve OFF. An explicit PSLP or Papilo presolver
    // is rejected up front, BEFORE draining the pending mutations, so a rejected
    // resolve is a clean no-op: the staged delta is left pending (not committed)
    // and the handle stays usable. Default / None proceed and are forced to None
    // before the solve. The delta path cannot honor a general outer presolve: the
    // PDLP warm-start seed lives in the on-device problem's space, dual simplex
    // warm-starts from the persisted basis of the unpresolved LP, and barrier
    // builds its user_problem directly from the on-device problem.
    if (pdlp_settings.presolver == cuopt::linear_programming::presolver_t::PSLP ||
        pdlp_settings.presolver == cuopt::linear_programming::presolver_t::Papilo) {
      CUOPT_LOG_ERROR(
        "cuOptResolve: the delta path requires presolve off; an explicit PSLP or Papilo presolver "
        "is not supported. Leave CUOPT_PRESOLVE at its default or set it off.");
      return CUOPT_INVALID_ARGUMENT;
    }

    // Drain the pending-mutation buffer right at the top, before any
    // settings/warm-start plumbing so the rest of cuOptResolve sees the
    // post-mutation problem dimensions. The fast no-op path (no mutations
    // staged) does nothing here and falls through to solve_lp on the
    // as-built / as-last-resolved problem.
    apply_pending_mutations(*view);

    const bool is_pdlp =
      (pdlp_settings.method == cuopt::linear_programming::method_t::PDLP);

    // PDLP warm start across resolves: seed ONLY the previous iterate via the
    // initial_primal/dual_solution path -- do NOT inject the full
    // pdlp_warm_start_data state.
    //
    // The full warm-start state carries, in addition to the iterate, the
    // running averages, current A^T y, and -- critically -- the convergence /
    // restart bookkeeping (last_restart / last_candidate KKT scores, the
    // last-restart duality-gap vectors, iteration counts, step size, primal
    // weight). All of that lives in the Ruiz / Pock-Chambolle *scaled* space of
    // the matrix that produced it. A delta mutation (column/row append is the
    // common CG case) changes the matrix and therefore the scaling, so that
    // state is stale: injecting the restart KKT scores / duality gap makes PDLP
    // believe it is already near-optimal and terminate at a stale, now-wrong
    // point -- returning incorrect solutions, not merely slow ones. Resetting
    // the scalar bookkeeping is not enough; the stale average / ATY / duality
    // gap *vectors* also corrupt the solve (verified empirically in mcfcg: every
    // variant of full-state injection returns objectives off the from-scratch
    // optimum).
    //
    // Seeding just the iterate through set_initial_primal/dual_solution leaves
    // averages, ATY, restart state and convergence tracking fresh, so PDLP
    // re-assesses convergence honestly on the new problem and converges to the
    // same optimum as a from-scratch solve, while still starting from a point
    // close to the answer (large speedup in CG, where each resolve appends only
    // a few columns). The current_primal/dual vectors are kept dimensionally in
    // sync with the problem by the delta pad/compact machinery.
    //
    // NOTE: the iterate is in the previous matrix's scaled space, and
    // update_primal_dual_solutions copies it into the new scaled working
    // solution without rescaling. The residual scaling mismatch only affects
    // how good the starting point is (speed), not correctness, because
    // termination is re-evaluated against the new problem. Consumers needing
    // CG-grade dual precision should solve PDLP to a tight tolerance (mcfcg
    // uses 1e-6); a too-loose tolerance lets warm-started PDLP stop with
    // imprecise duals that stall CG pricing.
    // Presolve rule on the delta path: the third-party presolve is forced OFF.
    // A caller that left presolve at Default or None proceeds silently (forced
    // to None below) and the resolve solves the original on-device problem; an
    // explicit PSLP / Papilo presolver was already rejected up front with
    // CUOPT_INVALID_ARGUMENT.
    //
    // Why force solve_lp's presolver OFF: barrier never ran the outer presolve
    // (it builds the user_problem directly from the problem and runs the
    // barrier), and PDLP's warm-start seed (below) lives in the on-device
    // problem's space. Solving with solve_lp's presolver off keeps behaviour
    // stable across mutate->resolve and converges to the same optimum.
    //
    // Note on sorted-CSR: solve_lp only runs sort_csr as part of the presolve
    // block, so skipping presolve also skips sort_csr. This relies on the delta
    // mutators producing sorted CSR output (cuOptAddColumns appends with strictly
    // ascending column indices; cuOptAddRows passes the user slice through
    // unchanged, so callers must supply sorted row data). An explicit PSLP /
    // Papilo presolver was already rejected up front (before the mutation drain);
    // Default / None reach here and are forced to None on the settings copy below.

    // Solve against a LOCAL COPY of the caller's PDLP settings so the public
    // cuOptSolverSettings handle is never mutated: we force presolve off and
    // (for PDLP) seed the persisted warm-start iterate on the copy only. This is
    // the same lightweight by-value duplication solve.cu uses for its batch
    // path; the caller's settings are observably unchanged on every exit, so
    // there is no snapshot/restore dance and no exception-safety window.
    auto solve_settings = pdlp_settings;

    // Delta presolve-off rule, forced on the copy. An explicit PSLP / Papilo
    // presolver was already rejected up front; Default / None reach here.
    solve_settings.presolver = cuopt::linear_programming::presolver_t::None;

    // PDLP warm start across resolves: seed ONLY the previous iterate (not the
    // full warm-start state -- see the note above) and only when the caller
    // supplied no initial solution of their own, so we never clobber user state.
    const bool seed_initial = is_pdlp && view->pdlp_warm_start.has_value() &&
                              !pdlp_settings.has_initial_primal_solution() &&
                              !pdlp_settings.has_initial_dual_solution();
    if (seed_initial) {
      auto stream = get_stream(*view);
      auto& ws    = *view->pdlp_warm_start;
      solve_settings.set_initial_primal_solution(
        ws.current_primal_solution_.data(),
        static_cast<cuopt_int_t>(ws.current_primal_solution_.size()),
        stream);
      solve_settings.set_initial_dual_solution(
        ws.current_dual_solution_.data(),
        static_cast<cuopt_int_t>(ws.current_dual_solution_.size()),
        stream);
    }

    // One uniform solve for every method. solve_lp -> solve_lp_with_method
    // routes method_t::Barrier -> run_barrier (which builds the user_problem
    // from the current op_problem and sets inner_presolve_optimizations=false),
    // PDLP, and dual-simplex through the same entry point. With presolver=None
    // on the copy, barrier through solve_lp reproduces what the old dedicated
    // persistent-barrier entry point did, minus the host user_problem cache
    // (removed: #26 measured it perf-neutral on mcfcg's small master LP).
    std::optional<optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>> lp_solution_opt;
    if (is_dual_simplex) {
      // #22: dual-simplex with presolve off -> persist the converted LP + basis
      // across resolves and warm-start from the previous optimum on a tail-only
      // structural extension; otherwise cold-rebuild. Objective-equivalent to a
      // from-scratch solve.
      if (!view->dual_simplex_warm_state) {
        view->dual_simplex_warm_state = std::make_unique<
          cuopt::linear_programming::dual_simplex_warm_state_t<cuopt_int_t, cuopt_float_t>>();
      }
      lp_solution_opt.emplace(
        cuopt::linear_programming::solve_lp_dual_simplex_warm<cuopt_int_t, cuopt_float_t>(
          *op, solve_settings, *view->dual_simplex_warm_state));
    } else {
      // barrier / PDLP, solved against the on-device (unpresolved) problem.
      lp_solution_opt.emplace(
        cuopt::linear_programming::solve_lp<cuopt_int_t, cuopt_float_t>(*op, solve_settings));
    }
    auto& lp_solution = *lp_solution_opt;

    // Preserve the contract: on non-success return, *previous_solution_ptr is
    // left unchanged. Infeasible / unbounded / iteration-limit terminations
    // still count as successful solves here (error_type == Success) so the
    // caller can inspect the termination status via cuOptGetTerminationStatus.
    const auto error_code =
      static_cast<cuopt_int_t>(lp_solution.get_error_status().get_error_type());
    if (error_code != CUOPT_SUCCESS) { return error_code; }

    // Capture warm-start iterates out of the fresh solution for the next
    // resolve. PDLP always fills these in via get_filled_warmed_start_data,
    // regardless of termination status. Barrier leaves them unset.
    if (is_pdlp) {
      view->pdlp_warm_start.emplace();
      *view->pdlp_warm_start = std::move(lp_solution.get_pdlp_warm_start_data());
    }

    // Recycle the output solution handle per the header contract. The header
    // also promises: on non-success return, *previous_solution_ptr is left
    // unchanged. So we allocate the replacement first — both objects — and
    // only destroy the old handle once both allocations have succeeded.
    // Otherwise an OOM between destroy and the second `new` would leave the
    // caller holding a nulled pointer while their previous solution is gone.
    // Wrap the by-value optimization_problem_solution_t in a gpu_lp_solution_t
    // (the lp_solution_interface_t implementation the post-v26.02 C API stores
    // behind the opaque cuOptSolution handle). Allocate both the wrapper and
    // the view before destroying the old handle, so an OOM mid-swap never
    // leaves the caller holding a freed pointer.
    auto new_lp_solution =
      std::make_unique<cuopt::linear_programming::gpu_lp_solution_t<cuopt_int_t, cuopt_float_t>>(
        std::move(lp_solution));
    auto new_solution_view =
      std::make_unique<solution_and_stream_view_t>(false, view->memory_backend);
    new_solution_view->lp_solution_interface_ptr = new_lp_solution.release();

    if (*previous_solution_ptr != nullptr) {
      cuOptDestroySolution(previous_solution_ptr);
    }
    *previous_solution_ptr = static_cast<cuOptSolution>(new_solution_view.release());

    return CUOPT_SUCCESS;
  } catch (const cuopt::logic_error& e) {
    CUOPT_LOG_ERROR("cuOptResolve: %s", e.what());
    return CUOPT_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("cuOptResolve: %s", e.what());
    return CUOPT_RUNTIME_ERROR;
  }
}

}  // extern "C"
