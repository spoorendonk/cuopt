/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cuopt::linear_programming {

// ----------------------------------------------------------------------------
// Dual-simplex warm-state layer (mcfcg issue #22).
//
// The uniform delta resolve path (#29) converts the device-resident
// optimization_problem_t to a host dual_simplex::user_problem_t and cold-solves
// it via run_dual_simplex on every cuOptResolve. Cold-solving discards the
// optimal basis between resolves, which is exactly the state a warm-started
// dual simplex needs to re-optimize a column-generation / cutting-plane master
// in a handful of pivots.
//
// dual_simplex_warm_state_t persists the converted LP and its optimal basis
// across resolves. On a structural EXTENSION of the previous problem (the
// column-generation and cut cases — new columns appended at the tail, new rows
// appended at the tail, all prior rows/columns preserved bit-for-bit) the
// persisted basis is transplanted onto the freshly converted LP:
//   * appended columns enter nonbasic at their lower bound (cheap — the basis
//     matrix B is unchanged; only nonbasic_list / vstatus / edge_norms grow);
//   * appended rows enter as cuts via dual_simplex::add_cuts, which appends the
//     row + its logical slack and updates the LU factor (basis_update) in
//     place, leaving the slack basic.
// The transplanted basis then seeds dual_phase2_with_advanced_basis, which
// re-optimizes from the warm basis. When the new problem is NOT a structural
// extension (first solve, deleted rows/columns, or a coefficient change on an
// existing entry) the layer falls back to a cold
// solve_linear_program_with_advanced_basis and re-captures the basis.
//
// CORRECTNESS / COORDINATE NOTE. The persisted basis lives in the coordinate
// space of `lp` (the converted, slack-augmented lp_problem_t). For the basis to
// be transplantable and for add_cuts (which operates on `lp`, not on a
// presolved/scaled copy) to be valid, the conversion must NOT run an
// outer presolve or column scaling that would re-index or rescale columns. The
// warm-solve settings therefore force scale_columns=false,
// inner_presolve_optimizations=false, eliminate_singletons=false,
// barrier_presolve=false. With those flags, for a clean LP (no empty rows)
// dual_simplex::presolve and dual_simplex::scaling are the identity, so the
// basis returned by solve_linear_program_with_advanced_basis is already in `lp`
// coordinates — this is the same configuration the MIP branch-and-bound cut
// loop relies on when it interleaves add_cuts with the persisted root basis
// (see branch_and_bound.cpp). This mirrors the now-uniform presolve-off rule of
// the delta path (#29) and is the path #28 later relaxes with a restricted
// forward-mappable presolve.
//
// The struct is opaque to cuopt_c_internal.hpp: the view holds a
// std::unique_ptr<dual_simplex_warm_state_t> and only forward-declares it, so
// the dual_simplex headers do not leak into the C API translation unit.
// ----------------------------------------------------------------------------
template <typename i_t, typename f_t>
struct dual_simplex_warm_state_t {
  // True once a basis has been captured and is consistent with `lp`.
  bool has_basis{false};

  // The converted (slack-augmented) LP whose coordinate space the basis lives
  // in. Rebuilt from user_problem on a cold solve; extended in place on a warm
  // resolve.
  std::optional<dual_simplex::lp_problem_t<i_t, f_t>> lp;

  // Slack columns appended by convert_user_problem / add_cuts. Tracked so a
  // warm re-solve and subsequent add_cuts stay consistent.
  std::vector<i_t> new_slacks;

  // The persisted optimal basis, all in `lp` coordinates.
  std::optional<dual_simplex::basis_update_mpf_t<i_t, f_t>> basis_update;
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;
  std::vector<dual_simplex::variable_status_t> vstatus;
  std::vector<f_t> edge_norms;

  // Snapshot of the user-problem shape that produced the current `lp`. Used to
  // decide whether the next resolve is a tail-only structural extension (warm
  // transplant) or requires a cold rebuild.
  i_t user_num_rows{0};
  i_t user_num_cols{0};
  // Per-column / per-row signatures of the previous user_problem, so we can
  // verify the prefix is preserved (no coefficient/bound/objective edit on an
  // existing entry) before transplanting the basis.
  std::vector<f_t> prev_user_obj;        // size user_num_cols
  std::vector<f_t> prev_user_lower;      // size user_num_cols
  std::vector<f_t> prev_user_upper;      // size user_num_cols
  std::vector<i_t> prev_user_col_nnz;    // size user_num_cols
  std::vector<f_t> prev_user_rhs;        // size user_num_rows
  std::vector<char> prev_user_row_sense; // size user_num_rows
  i_t prev_user_num_range_rows{0};
};

// Resolve the dual-simplex method on the (already mutation-drained) device
// problem, persisting / reusing the optimal basis across calls via `state`.
// Behaviourally equivalent to a cold run_dual_simplex on the accumulated
// problem (objective parity); only the starting basis differs.
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_dual_simplex_warm(
  optimization_problem_t<i_t, f_t>& op_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  dual_simplex_warm_state_t<i_t, f_t>& state);

}  // namespace cuopt::linear_programming
