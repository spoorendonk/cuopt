/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/cuopt_c.h>
#include <cuopt/linear_programming/backend_selection.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_solution_interface.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <pdlp/dual_simplex_warm_state.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <variant>
#include <vector>

// Forward declaration; only used by pointer inside solver_settings_handle_t.
namespace cuopt::internals {
class base_solution_callback_t;
}

namespace cuopt::linear_programming {

// Compact `v` device-direct, dropping entries i where mask[i] != 0 (matches
// the cuOptDelete{Columns,Rows} mask convention). Caller must guarantee
// d_mask.size() == old_size, and `new_size` must equal the count of
// mask[i] == 0 entries. Defined in cuopt_c_delta_kernels.cu so the kernel
// stays in a CUDA-compiled translation unit.
void compact_warm_start_vector_device(rmm::device_uvector<cuopt_float_t>& v,
                                      const rmm::device_uvector<cuopt_int_t>& d_mask,
                                      cuopt_int_t old_size,
                                      cuopt_int_t new_size,
                                      rmm::cuda_stream_view stream);

// Scatter (indices, values) into op's objective-coefficient device vector
// in a single kernel. Defined in cuopt_c_delta_kernels.cu. The caller must
// already have validated every index is in [0, op.get_n_variables()).
void set_objective_coefficients_device(
  optimization_problem_t<cuopt_int_t, cuopt_float_t>& op,
  const cuopt_int_t* indices,
  const cuopt_float_t* values,
  cuopt_int_t num_indices,
  rmm::cuda_stream_view stream);

// ----------------------------------------------------------------------------
// Pending-mutation buffer.
//
// Each delta-API mutator (cuOptAddColumns / cuOptAddRows /
// cuOptDelete{Columns,Rows} / cuOptSetObjectiveCoefficients) stages its inputs
// into a host-side payload and appends a pending_mutation_t to the log.
// cuOptResolve drains the log via apply_pending_mutations() before delegating
// to solve_lp, replaying mutations in arrival order against the persistent
// optimization_problem_t. No coalescing — straight replay.
//
// The payload structs intentionally own deep copies of all inputs (caller
// pointers don't survive past the C call). They live behind the opaque view
// pointer and are not part of the public C API surface.
// ----------------------------------------------------------------------------

struct add_columns_payload_t {
  cuopt_int_t num_columns{0};
  std::vector<cuopt_float_t> objective_coefficients;
  std::vector<cuopt_float_t> variable_lower_bounds;
  std::vector<cuopt_float_t> variable_upper_bounds;
  std::vector<cuopt_int_t> column_starts;     // size num_columns + 1
  std::vector<cuopt_int_t> row_indices;       // size column_starts.back()
  std::vector<cuopt_float_t> values;          // size column_starts.back()
  std::vector<char> variable_types;           // size num_columns; empty if NULL was passed
};

struct add_rows_payload_t {
  cuopt_int_t num_rows{0};
  std::vector<cuopt_float_t> constraint_lower_bounds;
  std::vector<cuopt_float_t> constraint_upper_bounds;
  std::vector<cuopt_int_t> row_starts;        // size num_rows + 1
  std::vector<cuopt_int_t> column_indices;    // size row_starts.back()
  std::vector<cuopt_float_t> values;          // size row_starts.back()
};

struct delete_columns_payload_t {
  std::vector<cuopt_int_t> indices;  // sorted strictly ascending
};

struct delete_rows_payload_t {
  std::vector<cuopt_int_t> indices;  // sorted strictly ascending
};

struct set_objective_payload_t {
  std::vector<cuopt_int_t> indices;
  std::vector<cuopt_float_t> values;
};

using pending_payload_t = std::variant<add_columns_payload_t,
                                       add_rows_payload_t,
                                       delete_columns_payload_t,
                                       delete_rows_payload_t,
                                       set_objective_payload_t>;

struct pending_mutation_t {
  enum class kind_t {
    AddColumns,
    AddRows,
    DeleteColumns,
    DeleteRows,
    SetObjective,
  };
  kind_t kind;
  pending_payload_t payload;
};

struct pending_mutations_t {
  std::vector<pending_mutation_t> log;
  // Logical post-pending sizes — incrementally maintained by the mutators so
  // they can validate index inputs (e.g. SetObjective indices < logical
  // n_vars, DeleteColumns indices < logical n_vars) without round-tripping
  // through the (potentially stale) device problem. Reset by
  // apply_pending_mutations after the log is drained.
  cuopt_int_t logical_n_vars{0};
  cuopt_int_t logical_n_constraints{0};
  bool dirty{false};  // mirror of !log.empty(); kept for explicit fast-path
                      // checks in cuOptResolve.
  // Set true if a drain (apply_pending_mutations) throws partway through,
  // leaving op_problem partially mutated. The handle is then unrecoverable in
  // place: every delta entry point and cuOptResolve bails with
  // CUOPT_INVALID_ARGUMENT so a stale log can never be re-drained and
  // double-applied. The caller must destroy and recreate the problem.
  bool poisoned{false};
};

struct problem_and_stream_view_t {
  problem_and_stream_view_t(memory_backend_t mem_backend)
    : memory_backend(mem_backend), stream_view_ptr(nullptr), handle_ptr(nullptr)
  {
    if (mem_backend == memory_backend_t::GPU) {
      // Use RAII locals so partial allocations are cleaned up if a later new throws
      std::unique_ptr<rmm::cuda_stream_view> sv(
        new rmm::cuda_stream_view(rmm::cuda_stream_per_thread));
      std::unique_ptr<raft::handle_t> h(new raft::handle_t(*sv));
      std::unique_ptr<optimization_problem_t<cuopt_int_t, cuopt_float_t>> gp(
        new optimization_problem_t<cuopt_int_t, cuopt_float_t>(h.get()));
      stream_view_ptr = sv.release();
      handle_ptr      = h.release();
      gpu_problem     = gp.release();
      cpu_problem     = nullptr;
    } else {
      cpu_problem = new cpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>();
      gpu_problem = nullptr;
    }
  }

  // Non-copyable
  problem_and_stream_view_t(const problem_and_stream_view_t&)            = delete;
  problem_and_stream_view_t& operator=(const problem_and_stream_view_t&) = delete;

  // Movable
  problem_and_stream_view_t(problem_and_stream_view_t&& other) noexcept
    : memory_backend(other.memory_backend),
      gpu_problem(other.gpu_problem),
      cpu_problem(other.cpu_problem),
      stream_view_ptr(other.stream_view_ptr),
      handle_ptr(other.handle_ptr),
      // delta-API persistent state — moved so a relocated handle keeps its
      // pending buffer / warm-start intact.
      pdlp_warm_start(std::move(other.pdlp_warm_start)),
      dual_simplex_warm_state(std::move(other.dual_simplex_warm_state)),
      pending(std::move(other.pending))
  {
    other.gpu_problem     = nullptr;
    other.cpu_problem     = nullptr;
    other.stream_view_ptr = nullptr;
    other.handle_ptr      = nullptr;
  }

  problem_and_stream_view_t& operator=(problem_and_stream_view_t&& other) noexcept
  {
    if (this != &other) {
      if (gpu_problem) delete gpu_problem;
      if (cpu_problem) delete cpu_problem;
      if (handle_ptr) delete handle_ptr;
      if (stream_view_ptr) delete stream_view_ptr;

      memory_backend  = other.memory_backend;
      gpu_problem     = other.gpu_problem;
      cpu_problem     = other.cpu_problem;
      stream_view_ptr = other.stream_view_ptr;
      handle_ptr      = other.handle_ptr;

      other.gpu_problem     = nullptr;
      other.cpu_problem     = nullptr;
      other.stream_view_ptr = nullptr;
      other.handle_ptr      = nullptr;

      // delta-API persistent state
      pdlp_warm_start         = std::move(other.pdlp_warm_start);
      dual_simplex_warm_state = std::move(other.dual_simplex_warm_state);
      pending                 = std::move(other.pending);
    }
    return *this;
  }

  ~problem_and_stream_view_t()
  {
    if (gpu_problem) delete gpu_problem;
    if (cpu_problem) delete cpu_problem;
    if (handle_ptr) delete handle_ptr;
    if (stream_view_ptr) delete stream_view_ptr;
  }

  raft::handle_t* get_handle_ptr() { return handle_ptr; }

  optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>* get_problem()
  {
    return memory_backend == memory_backend_t::GPU
             ? static_cast<optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 gpu_problem)
             : static_cast<optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 cpu_problem);
  }

  optimization_problem_t<cuopt_int_t, cuopt_float_t>* get_gpu_problem()
  {
    if (memory_backend == memory_backend_t::GPU) {
      return gpu_problem;
    } else {
      return nullptr;
    }
  }

  memory_backend_t memory_backend;
  optimization_problem_t<cuopt_int_t, cuopt_float_t>* gpu_problem;
  cpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>* cpu_problem;
  rmm::cuda_stream_view*
    stream_view_ptr;           // nullptr for CPU memory backend to avoid CUDA initialization
  raft::handle_t* handle_ptr;  // nullptr for CPU memory backend to avoid CUDA initialization

  // ----- delta-API persistent state (GPU backend only) -----
  // These are default-empty on a CPU-backed handle and never touched there;
  // the delta-API mutators / cuOptResolve bail with CUOPT_INVALID_ARGUMENT
  // before reaching them when get_gpu_problem() == nullptr.

  // PDLP warm-start iterates persisted across cuOptResolve calls. Populated
  // only after the first PDLP solve in cuOptResolve; mutated in-shape by
  // cuOptAddColumns / cuOptAddRows / cuOptDeleteColumns / cuOptDeleteRows so
  // the iterate stays aligned with the problem dimensions. Barrier resolves
  // leave this alone.
  std::optional<pdlp_warm_start_data_t<cuopt_int_t, cuopt_float_t>> pdlp_warm_start;

  // Dual-simplex warm-state layer (issue #22). Persists the converted LP +
  // optimal basis (basis_update / basic_list / nonbasic_list / vstatus /
  // edge_norms) across resolves so a dual-simplex resolve can warm-start from
  // the previous optimum instead of cold-solving. Populated lazily on the first
  // dual-simplex cuOptResolve; left null for barrier / PDLP. Held by unique_ptr
  // so the (large) basis state stays off the view's move/copy hot path and out
  // of the common barrier/PDLP code path. See dual_simplex_warm_state.hpp.
  std::unique_ptr<dual_simplex_warm_state_t<cuopt_int_t, cuopt_float_t>>
    dual_simplex_warm_state;

  // Lazy-rebuild buffer. Mutators stage inputs here; cuOptResolve drains.
  // logical_n_vars / logical_n_constraints are seeded from op_problem on
  // first mutator entry (see cuopt_c_delta.cpp: ensure_pending_initialized).
  pending_mutations_t pending;
};

struct solution_and_stream_view_t {
  solution_and_stream_view_t(bool solution_for_mip, memory_backend_t mem_backend)
    : is_mip(solution_for_mip),
      mip_solution_interface_ptr(nullptr),
      lp_solution_interface_ptr(nullptr),
      memory_backend(mem_backend)
  {
  }

  // Non-copyable
  solution_and_stream_view_t(const solution_and_stream_view_t&)            = delete;
  solution_and_stream_view_t& operator=(const solution_and_stream_view_t&) = delete;

  // Movable
  solution_and_stream_view_t(solution_and_stream_view_t&& other) noexcept
    : is_mip(other.is_mip),
      mip_solution_interface_ptr(other.mip_solution_interface_ptr),
      lp_solution_interface_ptr(other.lp_solution_interface_ptr),
      memory_backend(other.memory_backend)
  {
    other.mip_solution_interface_ptr = nullptr;
    other.lp_solution_interface_ptr  = nullptr;
  }

  solution_and_stream_view_t& operator=(solution_and_stream_view_t&& other) noexcept
  {
    if (this != &other) {
      if (mip_solution_interface_ptr) delete mip_solution_interface_ptr;
      if (lp_solution_interface_ptr) delete lp_solution_interface_ptr;

      is_mip                     = other.is_mip;
      mip_solution_interface_ptr = other.mip_solution_interface_ptr;
      lp_solution_interface_ptr  = other.lp_solution_interface_ptr;
      memory_backend             = other.memory_backend;

      other.mip_solution_interface_ptr = nullptr;
      other.lp_solution_interface_ptr  = nullptr;
    }
    return *this;
  }

  ~solution_and_stream_view_t()
  {
    if (mip_solution_interface_ptr) delete mip_solution_interface_ptr;
    if (lp_solution_interface_ptr) delete lp_solution_interface_ptr;
  }

  /**
   * @brief Get the solution as base interface pointer
   * @return Base interface pointer for polymorphic access to common methods
   * @note Allows uniform access to get_solution_host(), get_error_status(), get_solve_time()
   */
  optimization_problem_solution_interface_t<cuopt_int_t, cuopt_float_t>* get_solution()
  {
    return is_mip
             ? static_cast<optimization_problem_solution_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 mip_solution_interface_ptr)
             : static_cast<optimization_problem_solution_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 lp_solution_interface_ptr);
  }

  bool is_mip;
  mip_solution_interface_t<cuopt_int_t, cuopt_float_t>* mip_solution_interface_ptr;
  lp_solution_interface_t<cuopt_int_t, cuopt_float_t>* lp_solution_interface_ptr;
  memory_backend_t memory_backend;  // Track if GPU or CPU memory for data access
};

// Owns solver settings and C callback wrappers for C API lifetime. Lifted out
// of cuopt_c.cpp into this header so both cuopt_c.cpp and cuopt_c_delta.cpp can
// reach through the opaque cuOptSolverSettings handle.
struct solver_settings_handle_t {
  solver_settings_handle_t() : settings(new solver_settings_t<cuopt_int_t, cuopt_float_t>()) {}
  ~solver_settings_handle_t() { delete settings; }
  solver_settings_t<cuopt_int_t, cuopt_float_t>* settings;
  std::vector<std::unique_ptr<cuopt::internals::base_solution_callback_t>> callbacks;
};

inline solver_settings_handle_t* get_settings_handle(cuOptSolverSettings settings)
{
  return static_cast<solver_settings_handle_t*>(settings);
}

}  // namespace cuopt::linear_programming
