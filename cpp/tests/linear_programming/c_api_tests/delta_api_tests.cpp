/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// gtest coverage of the cuOpt delta C API. The tests exercise the public
// header (cuopt_c_delta.h) only — we link against libcuopt via the
// existing ConfigureTest() machinery and don't reach into internals.
//
// All delta-API entry points touch the GPU (cuOptCreateRangedProblem
// itself uploads CSR / bounds via raft::copy on the underlying RAFT
// handle), so every test in this file needs a working CUDA device. A
// runtime gate based on cudaGetDeviceCount() is applied per-test so the
// suite GTEST_SKIPs cleanly on a CPU-only build agent rather than
// failing.
//
// Lazy-rebuild contract: mutators stage operations on the host and apply
// them in arrival order at cuOptResolve. Public getters (cuOptGetNum* /
// cuOptGetObjectiveCoefficients / cuOptGetConstraintMatrix) reflect the
// last-resolved state of the problem, NOT pending mutations. Tests that
// want to inspect post-mutation shape must resolve first.

#include <cuopt/linear_programming/cuopt_c.h>
#include <cuopt/linear_programming/cuopt_c_delta.h>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace {

// True if the host has at least one CUDA-visible device. Tests skip when
// false rather than failing — CI agents without a GPU still need to be
// able to compile and link these tests.
bool gpu_available()
{
  int count          = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    // Drain the sticky error so subsequent tests don't observe it.
    (void)cudaGetLastError();
    return false;
  }
  return count > 0;
}

// Build a tiny ranged LP:
//   min   x + y
//   s.t.  x + y >= 1   (lower bound 1, upper bound +inf)
//         0 <= x, y <= 10
// Returns CUOPT_SUCCESS and sets *problem_ptr on success. Caller owns the
// handle and must cuOptDestroyProblem it.
cuopt_int_t make_tiny_problem(cuOptOptimizationProblem* problem_ptr)
{
  static const cuopt_float_t obj[]      = {1.0, 1.0};
  static const cuopt_int_t row_off[]    = {0, 2};
  static const cuopt_int_t row_idx[]    = {0, 1};
  static const cuopt_float_t row_val[]  = {1.0, 1.0};
  static const cuopt_float_t cstr_lo[]  = {1.0};
  static const cuopt_float_t cstr_hi[]  = {CUOPT_INFINITY};
  static const cuopt_float_t var_lo[]   = {0.0, 0.0};
  static const cuopt_float_t var_hi[]   = {10.0, 10.0};
  static const char var_types[]         = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
  return cuOptCreateRangedProblem(/*num_constraints=*/1,
                                  /*num_variables=*/2,
                                  /*objective_sense=*/CUOPT_MINIMIZE,
                                  /*objective_offset=*/0.0,
                                  obj,
                                  row_off,
                                  row_idx,
                                  row_val,
                                  cstr_lo,
                                  cstr_hi,
                                  var_lo,
                                  var_hi,
                                  var_types,
                                  problem_ptr);
}

// Configure a settings handle to use a deterministic small-problem solver
// (dual simplex) — used in tests that call cuOptResolve only to flush the
// pending mutation buffer, not to test convergence behaviour. PDLP would
// also work but takes more iterations on these tiny LPs.
cuopt_int_t make_dual_simplex_settings(cuOptSolverSettings* settings_ptr)
{
  cuopt_int_t rc = cuOptCreateSolverSettings(settings_ptr);
  if (rc != CUOPT_SUCCESS) { return rc; }
  return cuOptSetIntegerParameter(*settings_ptr, CUOPT_METHOD, CUOPT_METHOD_DUAL_SIMPLEX);
}

}  // namespace

// ----------------------------------------------------------------------------
// (a) delta_api.smoke: each mutator runs end-to-end against a 1-row, 2-var LP.
// In the lazy-rebuild architecture, dimensions exposed via the public getters
// only reflect post-mutation state once cuOptResolve has drained the pending
// buffer, so this test resolves between probes.
// ----------------------------------------------------------------------------
TEST(delta_api, smoke)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);
  ASSERT_NE(problem, nullptr);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  cuopt_int_t n_vars = -1, n_cons = -1;
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 2);
  EXPECT_EQ(n_cons, 1);

  // Stage every mutator without solving. None of these should touch the GPU
  // and none should appear in the public getters until we call cuOptResolve.
  // cuOptAddColumns: append one column with one nz at row 0.
  {
    const cuopt_float_t obj[]   = {3.0};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {5.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t row_idx[] = {0};
    const cuopt_float_t val[]   = {2.0};
    const char vt[]             = {CUOPT_CONTINUOUS};
    EXPECT_EQ(cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, vt),
              CUOPT_SUCCESS);
  }
  // Pre-resolve, the public getter still reports the original n_vars=2.
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 2);

  // cuOptAddRows: append one row touching all three columns (the new column
  // is logically index 2, even though it isn't on the device yet).
  {
    const cuopt_float_t c_lo[]    = {0.0};
    const cuopt_float_t c_hi[]    = {7.0};
    const cuopt_int_t row_off[]   = {0, 3};
    const cuopt_int_t col_idx[]   = {0, 1, 2};
    const cuopt_float_t val[]     = {1.0, 1.0, 1.0};
    EXPECT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, val), CUOPT_SUCCESS);
  }

  // cuOptSetObjectiveCoefficients: scalar use (n=1) bumps variable 0.
  {
    const cuopt_int_t idx        = 0;
    const cuopt_float_t coeff    = 7.0;
    EXPECT_EQ(cuOptSetObjectiveCoefficients(problem, 1, &idx, &coeff), CUOPT_SUCCESS);
  }

  // cuOptSetObjectiveCoefficients: batched scatter to vars 1 and 2.
  {
    const cuopt_int_t indices[]   = {1, 2};
    const cuopt_float_t values[]  = {8.0, 9.0};
    EXPECT_EQ(cuOptSetObjectiveCoefficients(problem, 2, indices, values), CUOPT_SUCCESS);
  }

  // Drain the buffer via resolve and verify dimensions reflect the
  // post-mutation state.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 3);
  EXPECT_EQ(n_cons, 2);

  // cuOptDeleteColumns: drop variable 1.
  {
    const cuopt_int_t indices[] = {1};
    EXPECT_EQ(cuOptDeleteColumns(problem, 1, indices), CUOPT_SUCCESS);
  }

  // cuOptDeleteRows: drop the original row 0.
  {
    const cuopt_int_t indices[] = {0};
    EXPECT_EQ(cuOptDeleteRows(problem, 1, indices), CUOPT_SUCCESS);
  }

  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 2);
  EXPECT_EQ(n_cons, 1);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
  EXPECT_EQ(problem, nullptr);
}

// ----------------------------------------------------------------------------
// (b) delta_api.invariants: after a sequence of mutations and a resolve, the
// problem's public-facing matrix shape, objective, and bounds match what
// constructing the equivalent problem from scratch would produce. We compare
// via the public getters only — no internals.
// ----------------------------------------------------------------------------
TEST(delta_api, invariants)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  // Add a column [row0:2.0] with cost 3.0 (var index 2).
  {
    const cuopt_float_t obj[]   = {3.0};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {5.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t row_idx[] = {0};
    const cuopt_float_t val[]   = {2.0};
    const char vt[]             = {CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, vt),
              CUOPT_SUCCESS);
  }
  // Add a row [c0:1, c1:1, c2:1] with bounds [0, 7].
  {
    const cuopt_float_t c_lo[]  = {0.0};
    const cuopt_float_t c_hi[]  = {7.0};
    const cuopt_int_t row_off[] = {0, 3};
    const cuopt_int_t col_idx[] = {0, 1, 2};
    const cuopt_float_t val[]   = {1.0, 1.0, 1.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, val), CUOPT_SUCCESS);
  }
  // Bump objectives: x->1.5 (scalar use, n=1), y->2.5 + z->3.5 (batched).
  {
    const cuopt_int_t idx     = 0;
    const cuopt_float_t coeff = 1.5;
    ASSERT_EQ(cuOptSetObjectiveCoefficients(problem, 1, &idx, &coeff), CUOPT_SUCCESS);
  }
  {
    const cuopt_int_t idx[]   = {1, 2};
    const cuopt_float_t val[] = {2.5, 3.5};
    ASSERT_EQ(cuOptSetObjectiveCoefficients(problem, 2, idx, val), CUOPT_SUCCESS);
  }

  // Drain the pending buffer onto the device.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  // Verify dimensions.
  cuopt_int_t n_vars = -1, n_cons = -1, nnz = -1;
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumNonZeros(problem, &nnz), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 3);
  EXPECT_EQ(n_cons, 2);
  // Row 0 had 2 nz, plus the new column adds 1 to row 0; row 1 added has 3 nz.
  EXPECT_EQ(nnz, 6);

  // Verify objective via the public getter.
  std::vector<cuopt_float_t> obj_out(static_cast<std::size_t>(n_vars), 0.0);
  ASSERT_EQ(cuOptGetObjectiveCoefficients(problem, obj_out.data()), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(obj_out[0], 1.5);
  EXPECT_DOUBLE_EQ(obj_out[1], 2.5);
  EXPECT_DOUBLE_EQ(obj_out[2], 3.5);

  // Verify constraint matrix shape via cuOptGetConstraintMatrix.
  std::vector<cuopt_int_t> row_off(static_cast<std::size_t>(n_cons + 1), 0);
  std::vector<cuopt_int_t> col_idx(static_cast<std::size_t>(nnz), 0);
  std::vector<cuopt_float_t> val(static_cast<std::size_t>(nnz), 0.0);
  ASSERT_EQ(cuOptGetConstraintMatrix(problem, row_off.data(), col_idx.data(), val.data()),
            CUOPT_SUCCESS);
  EXPECT_EQ(row_off[0], 0);
  EXPECT_EQ(row_off[1], 3);
  EXPECT_EQ(row_off[2], 6);
  // Row 0 contains entries for cols 0,1,2 (in some sort order). Sum of
  // values is 4.0 (1 + 1 + 2). Row 1 contains cols 0,1,2 with values 1+1+1
  // = 3.0. We assert sums rather than order-sensitive layout because
  // cuOptAddColumns is not promised to preserve original column-index
  // sorting in a row when sort_csr() is skipped (delta path).
  cuopt_float_t row0_sum = 0.0, row1_sum = 0.0;
  for (cuopt_int_t k = row_off[0]; k < row_off[1]; ++k) { row0_sum += val[k]; }
  for (cuopt_int_t k = row_off[1]; k < row_off[2]; ++k) { row1_sum += val[k]; }
  EXPECT_DOUBLE_EQ(row0_sum, 4.0);
  EXPECT_DOUBLE_EQ(row1_sum, 3.0);

  // Verify ranged constraint bounds.
  std::vector<cuopt_float_t> c_lo(static_cast<std::size_t>(n_cons), 0.0);
  std::vector<cuopt_float_t> c_hi(static_cast<std::size_t>(n_cons), 0.0);
  ASSERT_EQ(cuOptGetConstraintLowerBounds(problem, c_lo.data()), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetConstraintUpperBounds(problem, c_hi.data()), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(c_lo[0], 1.0);
  EXPECT_TRUE(std::isinf(c_hi[0]) && c_hi[0] > 0);
  EXPECT_DOUBLE_EQ(c_lo[1], 0.0);
  EXPECT_DOUBLE_EQ(c_hi[1], 7.0);

  // Verify variable bounds.
  std::vector<cuopt_float_t> v_lo(static_cast<std::size_t>(n_vars), 0.0);
  std::vector<cuopt_float_t> v_hi(static_cast<std::size_t>(n_vars), 0.0);
  ASSERT_EQ(cuOptGetVariableLowerBounds(problem, v_lo.data()), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetVariableUpperBounds(problem, v_hi.data()), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(v_lo[0], 0.0);
  EXPECT_DOUBLE_EQ(v_lo[1], 0.0);
  EXPECT_DOUBLE_EQ(v_lo[2], 0.0);
  EXPECT_DOUBLE_EQ(v_hi[0], 10.0);
  EXPECT_DOUBLE_EQ(v_hi[1], 10.0);
  EXPECT_DOUBLE_EQ(v_hi[2], 5.0);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (c) delta_api.invariants_after_delete: post-resolve dimensions and matrix
// nnz reflect the dropped rows/columns. Exercises the device-side warm-start
// compaction path indirectly (warm-start is empty before the first resolve,
// so this is dimension-only — a fuller test lives below in
// delta_api.pdlp_warmstart).
// ----------------------------------------------------------------------------
TEST(delta_api, invariants_after_delete)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  // Stage growth to 2 rows, 4 vars then resolve so subsequent deletes can
  // be validated against logical 4-var / 2-row sizes.
  for (int j = 0; j < 2; ++j) {
    const cuopt_float_t obj[]   = {1.0};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {1.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t row_idx[] = {0};
    const cuopt_float_t val[]   = {1.0};
    const char vt[]             = {CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, vt),
              CUOPT_SUCCESS);
  }
  {
    const cuopt_float_t c_lo[]  = {0.0};
    const cuopt_float_t c_hi[]  = {10.0};
    const cuopt_int_t row_off[] = {0, 4};
    const cuopt_int_t col_idx[] = {0, 1, 2, 3};
    const cuopt_float_t val[]   = {1.0, 1.0, 1.0, 1.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, val), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  cuopt_int_t n_vars = -1, n_cons = -1;
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  ASSERT_EQ(n_vars, 4);
  ASSERT_EQ(n_cons, 2);

  // Drop variables 1 and 3, drop the original row.
  {
    const cuopt_int_t indices[] = {1, 3};
    ASSERT_EQ(cuOptDeleteColumns(problem, 2, indices), CUOPT_SUCCESS);
  }
  {
    const cuopt_int_t indices[] = {0};
    ASSERT_EQ(cuOptDeleteRows(problem, 1, indices), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 2);
  EXPECT_EQ(n_cons, 1);

  cuopt_int_t nnz = -1;
  ASSERT_EQ(cuOptGetNumNonZeros(problem, &nnz), CUOPT_SUCCESS);
  // Survivors are var 0 and var 2 in row 1 (the appended row, mask kept).
  // Original row 0 dropped; surviving row's column-1/column-3 entries
  // dropped with their columns. Two survivors, both touching the row.
  EXPECT_EQ(nnz, 2);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (d) delta_api.resolve_smoke: minimal cuOptCreate -> mutate -> cuOptResolve
// path. Requires a working solver (PDLP / Dual Simplex). Currently CUDA on
// this build agent is wedged so this skips at runtime; on a working agent
// it should solve the tiny LP and report optimal.
// ----------------------------------------------------------------------------
TEST(delta_api, resolve_smoke)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  // Add a column with cost 0.5 (cheaper than x or y at 1.0). Optimal under
  // the new x+y+z >= 1 / sum-of-objs minimization should drive the new var
  // upward, since it has the smallest cost coefficient.
  {
    const cuopt_float_t obj[]   = {0.5};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {1.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t row_idx[] = {0};
    const cuopt_float_t val[]   = {1.0};
    const char vt[]             = {CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, vt),
              CUOPT_SUCCESS);
  }

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_DUAL_SIMPLEX),
            CUOPT_SUCCESS);

  cuOptSolution solution = nullptr;
  const cuopt_int_t rc   = cuOptResolve(problem, settings, &solution);
  EXPECT_EQ(rc, CUOPT_SUCCESS);

  if (rc == CUOPT_SUCCESS) {
    cuopt_int_t status = -1;
    ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
    EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

    cuopt_float_t obj_val = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
    EXPECT_FALSE(std::isnan(obj_val));
    // (We just check the solve returned something parsable; the real
    // objective semantics are covered by the upstream c_api tests.)
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (e) delta_api.pdlp_warmstart: PDLP-only test that resolves twice with a
// column added between, and verifies the warm-start primal vector dimension
// matches the post-add variable count. Probes the apply-time
// pad_warm_start_for_columns / cuOptResolve persist-and-reinject path
// without reaching into the internal handle.
// ----------------------------------------------------------------------------
TEST(delta_api, pdlp_warmstart)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP),
            CUOPT_SUCCESS);
  // Pin presolve to DEFAULT (-1, distinct from the OFF/None the delta path forces
  // internally) so we can assert below that cuOptResolve never mutates the
  // caller's settings handle — it solves against a private copy.
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_DEFAULT),
            CUOPT_SUCCESS);

  cuOptSolution solution = nullptr;
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_NE(solution, nullptr);

  // Add a column.
  {
    const cuopt_float_t obj[]   = {2.0};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {1.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t row_idx[] = {0};
    const cuopt_float_t val[]   = {1.0};
    const char vt[]             = {CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, vt),
              CUOPT_SUCCESS);
  }

  // Second resolve. Should drain the pending AddColumns, pad the warm-start
  // primal iterate by one zero, then re-solve. Post-resolve n_vars should be
  // 3. This resolve also exercises the warm-start seed path, which mutates a
  // PRIVATE copy of the settings, never the caller's handle.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  // The caller's settings handle must be observably unchanged across resolves:
  // the delta path forces presolve off and seeds the warm-start iterate on a
  // local copy, not on `settings`.
  cuopt_int_t presolve_after = -999, method_after = -999;
  ASSERT_EQ(cuOptGetIntegerParameter(settings, CUOPT_PRESOLVE, &presolve_after), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetIntegerParameter(settings, CUOPT_METHOD, &method_after), CUOPT_SUCCESS);
  EXPECT_EQ(presolve_after, CUOPT_PRESOLVE_DEFAULT);
  EXPECT_EQ(method_after, CUOPT_METHOD_PDLP);

  cuopt_int_t n_vars = -1;
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 3);

  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  // PDLP under second resolve with warm start should still hit optimal
  // for this trivial LP; iteration / time-limit terminations would be a
  // regression in the warm-start handling.
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  // Pull primal solution to verify dimension matches new n_vars.
  std::vector<cuopt_float_t> primal(static_cast<std::size_t>(n_vars), 0.0);
  ASSERT_EQ(cuOptGetPrimalSolution(solution, primal.data()), CUOPT_SUCCESS);
  // No specific value check — the LP is degenerate (multiple optima); we
  // only assert the call succeeded with the expected vector size, which
  // confirms the post-add dimension propagated through the resolve.

  cuOptDestroySolution(&solution);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (f) delta_api.lazy_replay: stage 5+ mutations of mixed types, then a single
// resolve, and assert the final problem matches what an eager replay would
// produce. The mix exercises every mutator, including index validation
// against logical (post-pending) sizes.
// ----------------------------------------------------------------------------
TEST(delta_api, lazy_replay)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  // -- Stage 7 mutations without resolving. --
  // (1) AddColumns: append var 2 with row-0 coefficient 2.0, cost 3.0.
  {
    const cuopt_float_t obj[]   = {3.0};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {5.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t row_idx[] = {0};
    const cuopt_float_t val[]   = {2.0};
    ASSERT_EQ(
      cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, nullptr),
      CUOPT_SUCCESS);
  }
  // (2) AddRows: append row 1 referencing var 0, var 1, var 2 (where var 2
  //     only exists in the pending log — validation must pass against the
  //     logical n_vars, not the device n_vars).
  {
    const cuopt_float_t c_lo[]  = {0.0};
    const cuopt_float_t c_hi[]  = {7.0};
    const cuopt_int_t row_off[] = {0, 3};
    const cuopt_int_t col_idx[] = {0, 1, 2};
    const cuopt_float_t val[]   = {1.0, 1.0, 1.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, val), CUOPT_SUCCESS);
  }
  // (3) AddColumns: append var 3 with cost 4.0 and row-0 + row-1 coefficients.
  //     Row index 1 only exists in the pending log so far.
  {
    const cuopt_float_t obj[]   = {4.0};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {3.0};
    const cuopt_int_t col_off[] = {0, 2};
    const cuopt_int_t row_idx[] = {0, 1};
    const cuopt_float_t val[]   = {1.5, 0.5};
    ASSERT_EQ(
      cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, nullptr),
      CUOPT_SUCCESS);
  }
  // (4) SetObjective: bump var 0 to 0.7.
  {
    const cuopt_int_t idx     = 0;
    const cuopt_float_t coeff = 0.7;
    ASSERT_EQ(cuOptSetObjectiveCoefficients(problem, 1, &idx, &coeff), CUOPT_SUCCESS);
  }
  // (5) DeleteColumns: drop var 1.
  {
    const cuopt_int_t indices[] = {1};
    ASSERT_EQ(cuOptDeleteColumns(problem, 1, indices), CUOPT_SUCCESS);
  }
  // (6) SetObjective: batched bump on the post-delete indices. After (5), the
  //     logical layout is var0 (orig 0, now 0), var2 (orig 2, now 1),
  //     var3 (orig 3, now 2). Bump var-now-1 -> 9.0, var-now-2 -> 10.0.
  {
    const cuopt_int_t idx[] = {1, 2};
    const cuopt_float_t v[] = {9.0, 10.0};
    ASSERT_EQ(cuOptSetObjectiveCoefficients(problem, 2, idx, v), CUOPT_SUCCESS);
  }
  // (7) DeleteRows: drop the original row 0 (now logical-row 0 again, since
  //     no row deletion has happened in this run yet).
  {
    const cuopt_int_t indices[] = {0};
    ASSERT_EQ(cuOptDeleteRows(problem, 1, indices), CUOPT_SUCCESS);
  }

  // Public getters before resolve still report the original 1-row, 2-var
  // shape — the staged mutations have not touched op_problem yet.
  cuopt_int_t n_vars = -1, n_cons = -1;
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 2);
  EXPECT_EQ(n_cons, 1);

  // Drain.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  // Expected post-replay state:
  //   vars (in order): var0 (cost 0.7), original var 2 (cost 9.0),
  //                    original var 3 (cost 10.0)  -- i.e. 3 vars total
  //   rows: just the appended one (original row 0 dropped)
  //   matrix entry for the surviving row: the appended row referenced
  //     pre-delete cols 0,1,2 with values 1,1,1. After deleting col 1,
  //     two entries remain: surviving-col-0 (was col 0) value 1.0, and
  //     surviving-col-1 (was col 2) value 1.0. Then AddColumns step (3)
  //     appended a coefficient 0.5 for var 3 in row 1, so
  //     surviving-col-2 (was var 3) value 0.5.
  //
  //   row 0 is the row dropped in step (7); the surviving row was the
  //   appended row, so its bounds are [0, 7] from step (2).
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 3);
  EXPECT_EQ(n_cons, 1);

  std::vector<cuopt_float_t> obj_out(static_cast<std::size_t>(n_vars), 0.0);
  ASSERT_EQ(cuOptGetObjectiveCoefficients(problem, obj_out.data()), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(obj_out[0], 0.7);
  EXPECT_DOUBLE_EQ(obj_out[1], 9.0);
  EXPECT_DOUBLE_EQ(obj_out[2], 10.0);

  std::vector<cuopt_float_t> v_lo(static_cast<std::size_t>(n_vars), 0.0);
  std::vector<cuopt_float_t> v_hi(static_cast<std::size_t>(n_vars), 0.0);
  ASSERT_EQ(cuOptGetVariableLowerBounds(problem, v_lo.data()), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetVariableUpperBounds(problem, v_hi.data()), CUOPT_SUCCESS);
  // var0 keeps its original [0, 10] bounds; new vars come from steps (1)/(3)
  // with [0, 5] / [0, 3].
  EXPECT_DOUBLE_EQ(v_lo[0], 0.0);
  EXPECT_DOUBLE_EQ(v_hi[0], 10.0);
  EXPECT_DOUBLE_EQ(v_lo[1], 0.0);
  EXPECT_DOUBLE_EQ(v_hi[1], 5.0);
  EXPECT_DOUBLE_EQ(v_lo[2], 0.0);
  EXPECT_DOUBLE_EQ(v_hi[2], 3.0);

  cuopt_int_t nnz = -1;
  ASSERT_EQ(cuOptGetNumNonZeros(problem, &nnz), CUOPT_SUCCESS);
  EXPECT_EQ(nnz, 3);

  std::vector<cuopt_int_t> row_off(static_cast<std::size_t>(n_cons + 1), 0);
  std::vector<cuopt_int_t> col_idx(static_cast<std::size_t>(nnz), 0);
  std::vector<cuopt_float_t> mat_val(static_cast<std::size_t>(nnz), 0.0);
  ASSERT_EQ(cuOptGetConstraintMatrix(problem, row_off.data(), col_idx.data(), mat_val.data()),
            CUOPT_SUCCESS);
  EXPECT_EQ(row_off[0], 0);
  EXPECT_EQ(row_off[1], 3);
  // Map (col -> value). Order in the row depends on the apply-replay
  // mechanics; check by index lookup.
  cuopt_float_t v_at[3] = {0.0, 0.0, 0.0};
  bool seen[3]          = {false, false, false};
  for (cuopt_int_t k = 0; k < nnz; ++k) {
    const cuopt_int_t c = col_idx[k];
    ASSERT_GE(c, 0);
    ASSERT_LT(c, 3);
    v_at[c] = mat_val[k];
    seen[c] = true;
  }
  EXPECT_TRUE(seen[0] && seen[1] && seen[2]);
  EXPECT_DOUBLE_EQ(v_at[0], 1.0);  // var0 keeps row coefficient 1.0
  EXPECT_DOUBLE_EQ(v_at[1], 1.0);  // var2 (now col 1) keeps row coefficient 1.0
  EXPECT_DOUBLE_EQ(v_at[2], 0.5);  // var3 (now col 2) was appended at value 0.5

  std::vector<cuopt_float_t> c_lo(static_cast<std::size_t>(n_cons), 0.0);
  std::vector<cuopt_float_t> c_hi(static_cast<std::size_t>(n_cons), 0.0);
  ASSERT_EQ(cuOptGetConstraintLowerBounds(problem, c_lo.data()), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetConstraintUpperBounds(problem, c_hi.data()), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(c_lo[0], 0.0);
  EXPECT_DOUBLE_EQ(c_hi[0], 7.0);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (g) delta_api.empty_resolve_fast_path: create + resolve + resolve again
// with no mutations between. The second resolve should hit the
// `pending.dirty == false` fast path inside cuOptResolve and avoid the
// apply-pending overhead. Verifies it produces a sensible result and is
// not slower than the first.
// ----------------------------------------------------------------------------
TEST(delta_api, empty_resolve_fast_path)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  // First resolve.
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  const auto t1 = std::chrono::steady_clock::now();

  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  cuopt_float_t obj1 = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj1), CUOPT_SUCCESS);

  // Second resolve, no mutations. apply_pending_mutations is a no-op.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  const auto t2 = std::chrono::steady_clock::now();
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  cuopt_float_t obj2 = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj2), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(obj1, obj2);

  // Sanity check: post-resolve dimensions unchanged from create.
  cuopt_int_t n_vars = -1, n_cons = -1;
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 2);
  EXPECT_EQ(n_cons, 1);

  // The second resolve runs without the first-iter presolve, so it should
  // not be substantially slower than the first. We don't claim measured
  // perf here — just guard against pathological regression (10x). On a
  // tiny LP both are well under 100 ms.
  const auto first_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  const auto second_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  // Guard with a generous bound; this is intentionally loose to avoid
  // flakiness on shared CI agents.
  EXPECT_LE(second_ms, 10 * (first_ms + 1));

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (h) delta_api.persistent_barrier_resolve: a barrier-method resolve followed
// by an objective-only mutation and a second resolve must both succeed and
// reach the same optimal objective. Correctness-only — the uniform resolve
// path rebuilds from op_problem every time, so there is no cache or timing to
// assert.
// ----------------------------------------------------------------------------
TEST(delta_api, persistent_barrier_resolve)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(
    cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER),
    CUOPT_SUCCESS);

  cuOptSolution solution = nullptr;

  // Resolve 1: first barrier solve.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  cuopt_float_t obj1 = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj1), CUOPT_SUCCESS);

  // Pure obj-only mutation: change c[0] from 1.0 to 2.0.
  const cuopt_int_t obj_idx[]    = {0};
  const cuopt_float_t obj_vals[] = {2.0};
  ASSERT_EQ(cuOptSetObjectiveCoefficients(problem, 1, obj_idx, obj_vals), CUOPT_SUCCESS);

  // Resolve 2.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  cuopt_float_t obj2 = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj2), CUOPT_SUCCESS);

  // Original LP is min(x + y) s.t. x + y >= 1; x,y in [0,10]. obj1 == 1.
  // After raising c[0] to 2 we still want x + y == 1 with x = 0, y = 1, so
  // obj2 == 1. Use ASSERT_NEAR rather than DOUBLE_EQ since barrier
  // tolerances differ slightly across resolves.
  EXPECT_NEAR(obj1, 1.0, 1e-6);
  EXPECT_NEAR(obj2, 1.0, 1e-6);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (i) delta_api.barrier_row_eviction: a barrier resolve sequence that mixes an
// objective-only mutation and a structural (row-add) mutation must keep
// producing correct optima. The test layout is:
//   resolve 1  → first barrier solve.
//   set obj    → objective-only mutation.
//   resolve 2.
//   add row    → structural mutation (row count grows).
//   resolve 3.
// All three resolves must succeed and the post-resolve dimensions must reflect
// the added row.
// ----------------------------------------------------------------------------
TEST(delta_api, barrier_row_eviction)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(
    cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER),
    CUOPT_SUCCESS);

  cuOptSolution solution = nullptr;

  // Resolve 1.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  // Obj-only mutation, then resolve 2.
  {
    const cuopt_int_t obj_idx[]    = {1};
    const cuopt_float_t obj_vals[] = {1.5};
    ASSERT_EQ(cuOptSetObjectiveCoefficients(problem, 1, obj_idx, obj_vals), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  // Add a redundant row: x + y <= 100 (cannot bind, original optimum is 1) as
  // a structural mutation before resolve 3.
  {
    const cuopt_float_t row_lower[] = {-CUOPT_INFINITY};
    const cuopt_float_t row_upper[] = {100.0};
    const cuopt_int_t row_starts[]  = {0, 2};
    const cuopt_int_t col_indices[] = {0, 1};
    const cuopt_float_t row_vals[]  = {1.0, 1.0};
    ASSERT_EQ(cuOptAddRows(problem,
                           /*num_rows=*/1,
                           row_lower,
                           row_upper,
                           row_starts,
                           col_indices,
                           row_vals),
              CUOPT_SUCCESS);
  }

  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  // Post-resolve dimensions reflect the added row.
  cuopt_int_t n_cons = -1;
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  EXPECT_EQ(n_cons, 2);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (i) delta_api.invalid_args: every mutator's reject paths.
// Each invalid call must return non-CUOPT_SUCCESS and leave the problem in a
// usable state so a clean resolve afterwards still works.
// ----------------------------------------------------------------------------
TEST(delta_api, invalid_args)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);  // 1 cons, 2 vars
  ASSERT_NE(problem, nullptr);
  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  // Negative num_indices on every relevant mutator.
  {
    const cuopt_int_t idx[]   = {0};
    const cuopt_float_t val[] = {1.0};
    EXPECT_EQ(cuOptSetObjectiveCoefficients(problem, -1, idx, val), CUOPT_INVALID_ARGUMENT);
    EXPECT_EQ(cuOptDeleteColumns(problem, -1, idx), CUOPT_INVALID_ARGUMENT);
    EXPECT_EQ(cuOptDeleteRows(problem, -1, idx), CUOPT_INVALID_ARGUMENT);
  }

  // Out-of-range index in cuOptSetObjectiveCoefficients (vars are 0..1).
  {
    const cuopt_int_t idx[]   = {2};
    const cuopt_float_t val[] = {3.0};
    EXPECT_EQ(cuOptSetObjectiveCoefficients(problem, 1, idx, val), CUOPT_INVALID_ARGUMENT);
  }

  // Unsorted indices in cuOptDeleteColumns.
  {
    const cuopt_int_t idx[] = {1, 0};
    EXPECT_EQ(cuOptDeleteColumns(problem, 2, idx), CUOPT_INVALID_ARGUMENT);
  }
  // Duplicate indices in cuOptDeleteColumns.
  {
    const cuopt_int_t idx[] = {0, 0};
    EXPECT_EQ(cuOptDeleteColumns(problem, 2, idx), CUOPT_INVALID_ARGUMENT);
  }
  // Out-of-range index in cuOptDeleteRows (rows are 0..0).
  {
    const cuopt_int_t idx[] = {1};
    EXPECT_EQ(cuOptDeleteRows(problem, 1, idx), CUOPT_INVALID_ARGUMENT);
  }

  // cuOptAddRows with a column index out of range (vars are 0..1, col 2 bad).
  {
    const cuopt_float_t c_lo[]  = {0.0};
    const cuopt_float_t c_hi[]  = {1.0};
    const cuopt_int_t row_off[] = {0, 1};
    const cuopt_int_t bad_col[] = {2};
    const cuopt_float_t val[]   = {1.0};
    EXPECT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, bad_col, val), CUOPT_INVALID_ARGUMENT);
  }

  // cuOptAddColumns with a row index out of range (rows are 0..0, row 5 bad).
  {
    const cuopt_float_t obj[]   = {1.0};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {1.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t bad_row[] = {5};
    const cuopt_float_t val[]   = {1.0};
    EXPECT_EQ(cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, bad_row, val, nullptr),
              CUOPT_INVALID_ARGUMENT);
  }

  // After all the rejections the handle should still be usable: a clean
  // resolve should converge against the original 1-row, 2-var LP. This
  // confirms the rejected mutations did NOT poison the pending log.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// delta_api.explicit_presolve_rejected: the delta path forces presolve off. An
// EXPLICIT PSLP or Papilo request must get a hard CUOPT_INVALID_ARGUMENT from
// cuOptResolve on every method, and the handle must stay usable: clearing the
// explicit presolver (back to off/default) and resolving again succeeds. Mirrors
// the invalid_args reusability pattern.
// ----------------------------------------------------------------------------
TEST(delta_api, explicit_presolve_rejected)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  // Use the barrier method so this also covers presolve rejection on a
  // non-dual-simplex method (dual_simplex_presolve_rejected covers dual-simplex).
  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  // Explicit PSLP presolve: rejected.
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_PSLP), CUOPT_SUCCESS);
  EXPECT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_INVALID_ARGUMENT);

  // Explicit Papilo presolve: also rejected.
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_PAPILO),
            CUOPT_SUCCESS);
  EXPECT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_INVALID_ARGUMENT);

  // The rejection must not have poisoned the handle. Turn presolve off and
  // resolve again: it should converge.
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_OFF), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (j) delta_api.add_columns_coalescing: stages two runs of consecutive
// cuOptAddColumns calls separated by a cuOptAddRows, then a final mutation,
// and asserts the post-resolve dimensions / matrix nnz / objective vector
// match what an eager replay would produce. The implementation coalesces
// each run into a single apply call inside apply_pending_mutations; this
// test guards against state corruption introduced by that coalescing.
//
// Specifically it covers:
//   * two non-trivial AddColumns runs (5 and 5), so the coalesce path is
//     exercised twice in one drain, with an AddRows in between to break
//     the run.
//   * mixed variable_types: some payloads pass nullptr (defaults to
//     CONTINUOUS), others pass an explicit byte. Coalescing must
//     materialise types for ALL columns when any source had them set, so
//     we assert the final per-column types via cuOptGetVariableTypes-
//     equivalent paths (variable bounds + getter for objective; the
//     matrix shape is the load-bearing invariant).
//   * the second run references row indices that only exist in the
//     pending log (added by the prior staged AddRows), so coalesce-time
//     concatenation must preserve correct row references.
// ----------------------------------------------------------------------------
TEST(delta_api, add_columns_coalescing)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);  // 1 cons (row 0), 2 vars (0,1)

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);
  cuOptSolution solution = nullptr;

  // First run: 5 consecutive AddColumns. Each adds 1 column with 1 nz at
  // row 0. Costs 1.0..5.0; bounds [0, 1]. Mix nullptr / non-null types.
  // Cumulative new nnz from this run: 5.
  for (int j = 0; j < 5; ++j) {
    const cuopt_float_t obj_coef = static_cast<cuopt_float_t>(j + 1);
    const cuopt_float_t v_lo[]   = {0.0};
    const cuopt_float_t v_hi[]   = {1.0};
    const cuopt_int_t col_off[]  = {0, 1};
    const cuopt_int_t row_idx[]  = {0};
    const cuopt_float_t val[]    = {static_cast<cuopt_float_t>(j + 1)};
    // Alternate nullptr vs explicit CONTINUOUS to stress the coalesce-time
    // type-materialisation logic.
    if (j % 2 == 0) {
      ASSERT_EQ(cuOptAddColumns(problem, 1, &obj_coef, v_lo, v_hi, col_off, row_idx, val, nullptr),
                CUOPT_SUCCESS);
    } else {
      const char vt[] = {CUOPT_CONTINUOUS};
      ASSERT_EQ(cuOptAddColumns(problem, 1, &obj_coef, v_lo, v_hi, col_off, row_idx, val, vt),
                CUOPT_SUCCESS);
    }
  }

  // Break the run with one AddRows: append row 1 referencing all 7 logical
  // vars (var 0..1 from create + 5 just staged). nnz contribution: 7.
  {
    const cuopt_float_t c_lo[]    = {0.0};
    const cuopt_float_t c_hi[]    = {100.0};
    const cuopt_int_t row_off[]   = {0, 7};
    const cuopt_int_t col_idx[]   = {0, 1, 2, 3, 4, 5, 6};
    const cuopt_float_t val[]     = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, val), CUOPT_SUCCESS);
  }

  // Second run: another 5 consecutive AddColumns. Each touches both row 0
  // and row 1 (the staged row); so each contributes 2 nnz, total 10.
  for (int j = 0; j < 5; ++j) {
    const cuopt_float_t obj_coef = static_cast<cuopt_float_t>(10 + j);
    const cuopt_float_t v_lo[]   = {0.0};
    const cuopt_float_t v_hi[]   = {2.0};
    const cuopt_int_t col_off[]  = {0, 2};
    const cuopt_int_t row_idx[]  = {0, 1};
    const cuopt_float_t val[]    = {0.5, 0.25};
    ASSERT_EQ(cuOptAddColumns(problem, 1, &obj_coef, v_lo, v_hi, col_off, row_idx, val, nullptr),
              CUOPT_SUCCESS);
  }

  // Drain.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  // Expected dimensions: 2 (orig) + 5 (first run) + 5 (second run) = 12 vars.
  // Constraints: 1 (orig) + 1 (added) = 2.
  // nnz: 2 (orig row 0 entries) + 5 (first run row-0 entries) + 7 (added row
  //   touching first 7 vars) + 5*2 (second run, each touching row 0 and row 1)
  //   = 2 + 5 + 7 + 10 = 24.
  cuopt_int_t n_vars = -1, n_cons = -1, nnz = -1;
  ASSERT_EQ(cuOptGetNumVariables(problem, &n_vars), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumConstraints(problem, &n_cons), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetNumNonZeros(problem, &nnz), CUOPT_SUCCESS);
  EXPECT_EQ(n_vars, 12);
  EXPECT_EQ(n_cons, 2);
  EXPECT_EQ(nnz, 24);

  // Objective: vars 0,1 unchanged (1.0, 1.0), vars 2..6 are 1..5,
  // vars 7..11 are 10..14.
  std::vector<cuopt_float_t> obj_out(static_cast<std::size_t>(n_vars), 0.0);
  ASSERT_EQ(cuOptGetObjectiveCoefficients(problem, obj_out.data()), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(obj_out[0], 1.0);
  EXPECT_DOUBLE_EQ(obj_out[1], 1.0);
  for (int j = 0; j < 5; ++j) {
    EXPECT_DOUBLE_EQ(obj_out[static_cast<std::size_t>(2 + j)],
                     static_cast<cuopt_float_t>(j + 1));
  }
  for (int j = 0; j < 5; ++j) {
    EXPECT_DOUBLE_EQ(obj_out[static_cast<std::size_t>(7 + j)],
                     static_cast<cuopt_float_t>(10 + j));
  }

  // Variable bounds: vars 0,1 keep [0, 10]; vars 2..6 are [0, 1]; vars 7..11
  // are [0, 2].
  std::vector<cuopt_float_t> v_lo(static_cast<std::size_t>(n_vars), 0.0);
  std::vector<cuopt_float_t> v_hi(static_cast<std::size_t>(n_vars), 0.0);
  ASSERT_EQ(cuOptGetVariableLowerBounds(problem, v_lo.data()), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetVariableUpperBounds(problem, v_hi.data()), CUOPT_SUCCESS);
  EXPECT_DOUBLE_EQ(v_hi[0], 10.0);
  EXPECT_DOUBLE_EQ(v_hi[1], 10.0);
  for (int j = 0; j < 5; ++j) { EXPECT_DOUBLE_EQ(v_hi[static_cast<std::size_t>(2 + j)], 1.0); }
  for (int j = 0; j < 5; ++j) { EXPECT_DOUBLE_EQ(v_hi[static_cast<std::size_t>(7 + j)], 2.0); }

  // Verify per-row nnz counts (row 0: 2 + 5 + 5 = 12; row 1: 7 + 5 = 12).
  std::vector<cuopt_int_t> row_off(static_cast<std::size_t>(n_cons + 1), 0);
  std::vector<cuopt_int_t> col_idx(static_cast<std::size_t>(nnz), 0);
  std::vector<cuopt_float_t> mat_val(static_cast<std::size_t>(nnz), 0.0);
  ASSERT_EQ(cuOptGetConstraintMatrix(problem, row_off.data(), col_idx.data(), mat_val.data()),
            CUOPT_SUCCESS);
  EXPECT_EQ(row_off[0], 0);
  EXPECT_EQ(row_off[1], 12);
  EXPECT_EQ(row_off[2], 24);

  // Sanity: solve should still terminate optimally on the resulting LP.
  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (l) delta_api.barrier_correctness_transforms: under barrier, the always-on
// inner-presolve correctness transforms (nonzero lower-bound shift,
// free-variable v-w split) must run regardless of the
// inner_presolve_optimizations gate on simplex_solver_settings_t. Both
// transforms are required for barrier to handle LPs that don't already have
// x >= 0 and finite bounds; if either gets accidentally skipped, the solver
// returns the wrong primal.
//
// Cold LP (3 vars):
//   min  1*x1 + (-1)*x2 + 1*x3
//   s.t. x1 + x2 + x3 = 5      (single equality row)
//        x1 in [2, 10]           (nonzero LB -> exercises LB shift)
//        x2 free                  (-inf, +inf -> exercises v-w split)
//        x3 in [0, 10]
// Closed form (substitute x2 = 5 - x1 - x3): obj = 2*x1 + 2*x3 - 5, minimised
// at x1=2, x3=0 -> obj = -1, x2 = 3.
//
// After adding x4 (cost -2, coef 1 in row, [0, 10]):
//   obj = 2*x1 + 2*x3 - x4 - 5, minimised at x1=2, x3=0, x4=10 -> obj = -11,
//   x2 = -7 (must be representable; only works if free-var split is sound).
// ----------------------------------------------------------------------------
TEST(delta_api, barrier_correctness_transforms)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  static const cuopt_float_t obj[]     = {1.0, -1.0, 1.0};
  static const cuopt_int_t row_off[]   = {0, 3};
  static const cuopt_int_t row_idx[]   = {0, 1, 2};
  static const cuopt_float_t row_val[] = {1.0, 1.0, 1.0};
  static const cuopt_float_t cstr_lo[] = {5.0};
  static const cuopt_float_t cstr_hi[] = {5.0};
  static const cuopt_float_t var_lo[]  = {2.0, -CUOPT_INFINITY, 0.0};
  static const cuopt_float_t var_hi[]  = {10.0, CUOPT_INFINITY, 10.0};
  static const char var_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(cuOptCreateRangedProblem(/*num_constraints=*/1,
                                     /*num_variables=*/3,
                                     CUOPT_MINIMIZE,
                                     0.0,
                                     obj,
                                     row_off,
                                     row_idx,
                                     row_val,
                                     cstr_lo,
                                     cstr_hi,
                                     var_lo,
                                     var_hi,
                                     var_types,
                                     &problem),
            CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER),
            CUOPT_SUCCESS);

  // Cold solve: presolve must shift x1's LB to 0 internally and split x2 into
  // v - w. uncrush_solution maps back to the original (3-var) coordinate
  // system before we read it via cuOptGetPrimalSolution.
  cuOptSolution solution = nullptr;
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  constexpr cuopt_float_t kTol = 1e-4;
  cuopt_float_t obj_val = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, -1.0, kTol);

  std::vector<cuopt_float_t> x3(3, 0.0);
  ASSERT_EQ(cuOptGetPrimalSolution(solution, x3.data()), CUOPT_SUCCESS);
  EXPECT_GE(x3[0], 2.0 - kTol);                            // LB active and respected
  EXPECT_NEAR(x3[0] + x3[1] + x3[2], 5.0, kTol);           // equality row satisfied
  EXPECT_NEAR(x3[1], 3.0, kTol);                           // free-var split round-trip

  // Append x4 (cost -2, coef 1 in row 0, [0, 10]) and resolve.
  {
    const cuopt_float_t add_obj[]   = {-2.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {10.0};
    const cuopt_int_t add_col_off[] = {0, 1};
    const cuopt_int_t add_row_idx[] = {0};
    const cuopt_float_t add_val[]   = {1.0};
    ASSERT_EQ(cuOptAddColumns(problem,
                              1,
                              add_obj,
                              add_v_lo,
                              add_v_hi,
                              add_col_off,
                              add_row_idx,
                              add_val,
                              nullptr),
              CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, -11.0, kTol);

  std::vector<cuopt_float_t> x4(4, 0.0);
  ASSERT_EQ(cuOptGetPrimalSolution(solution, x4.data()), CUOPT_SUCCESS);
  EXPECT_GE(x4[0], 2.0 - kTol);                                // LB still respected
  EXPECT_NEAR(x4[0] + x4[1] + x4[2] + x4[3], 5.0, kTol);       // row satisfied
  EXPECT_LT(x4[1], 0.0);                                       // x2 must go negative
                                                               // (only sound if v-w
                                                               // split survives the
                                                               // delta path)

  // Cross-check: cold-build the same 4-var LP and confirm objectives agree.
  // This guards against drift between the delta path's iterative resolve and
  // the from-scratch path on an LP that exercises both correctness transforms.
  {
    const cuopt_float_t cs_obj[]     = {1.0, -1.0, 1.0, -2.0};
    const cuopt_int_t cs_row_off[]   = {0, 4};
    const cuopt_int_t cs_row_idx[]   = {0, 1, 2, 3};
    const cuopt_float_t cs_row_val[] = {1.0, 1.0, 1.0, 1.0};
    const cuopt_float_t cs_cstr_lo[] = {5.0};
    const cuopt_float_t cs_cstr_hi[] = {5.0};
    const cuopt_float_t cs_var_lo[]  = {2.0, -CUOPT_INFINITY, 0.0, 0.0};
    const cuopt_float_t cs_var_hi[]  = {10.0, CUOPT_INFINITY, 10.0, 10.0};
    const char cs_types[]            = {CUOPT_CONTINUOUS,
                                        CUOPT_CONTINUOUS,
                                        CUOPT_CONTINUOUS,
                                        CUOPT_CONTINUOUS};

    cuOptOptimizationProblem cs_problem   = nullptr;
    ASSERT_EQ(cuOptCreateRangedProblem(1,
                                       4,
                                       CUOPT_MINIMIZE,
                                       0.0,
                                       cs_obj,
                                       cs_row_off,
                                       cs_row_idx,
                                       cs_row_val,
                                       cs_cstr_lo,
                                       cs_cstr_hi,
                                       cs_var_lo,
                                       cs_var_hi,
                                       cs_types,
                                       &cs_problem),
              CUOPT_SUCCESS);
    cuOptSolution cs_solution = nullptr;
    ASSERT_EQ(cuOptResolve(cs_problem, settings, &cs_solution), CUOPT_SUCCESS);
    ASSERT_EQ(cuOptGetTerminationStatus(cs_solution, &status), CUOPT_SUCCESS);
    EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

    cuopt_float_t cs_obj_val = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(cs_solution, &cs_obj_val), CUOPT_SUCCESS);
    EXPECT_NEAR(cs_obj_val, obj_val, kTol);

    if (cs_solution != nullptr) { cuOptDestroySolution(&cs_solution); }
    cuOptDestroyProblem(&cs_problem);
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// (m) delta_api.barrier_resolve_matches_cold_rebuild: a barrier resolve
// sequence that mixes column appends with a row append in the middle must, at
// every step, reach the same objective as the equivalent LP built and solved
// from scratch.
//
// Sequence:
//   step 0: solve base (2 vars, 1 row).
//   step 1: append column, resolve.
//   step 2: append column, resolve.
//   step 3: append row, resolve.
//   step 4: append column, resolve.
// At each step, build the equivalent LP from scratch and confirm the
// objective matches within tolerance.
// ----------------------------------------------------------------------------
TEST(delta_api, barrier_resolve_matches_cold_rebuild)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  // Base LP: min x + y s.t. x + y >= 1, x,y in [0, 10]. Optimum: x + y = 1,
  // obj = 1.
  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER),
            CUOPT_SUCCESS);

  cuOptSolution solution = nullptr;
  cuopt_int_t status     = -1;
  cuopt_float_t obj_val  = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  constexpr cuopt_float_t kTol = 1e-4;

  // Build the cold reference LP at the same step and confirm objectives
  // agree. Reference: equivalent problem constructed from scratch via
  // cuOptCreateRangedProblem with all mutations baked in.
  auto solve_cold_reference = [&settings,
                               &status,
                               kTol](cuopt_int_t n_cons,
                                     cuopt_int_t n_vars,
                                     const cuopt_float_t* obj,
                                     const cuopt_int_t* row_off,
                                     const cuopt_int_t* row_idx,
                                     const cuopt_float_t* row_val,
                                     const cuopt_float_t* c_lo,
                                     const cuopt_float_t* c_hi,
                                     const cuopt_float_t* v_lo,
                                     const cuopt_float_t* v_hi,
                                     const char* v_types,
                                     cuopt_float_t expected_obj) {
    cuOptOptimizationProblem cs = nullptr;
    ASSERT_EQ(cuOptCreateRangedProblem(n_cons,
                                       n_vars,
                                       CUOPT_MINIMIZE,
                                       0.0,
                                       obj,
                                       row_off,
                                       row_idx,
                                       row_val,
                                       c_lo,
                                       c_hi,
                                       v_lo,
                                       v_hi,
                                       v_types,
                                       &cs),
              CUOPT_SUCCESS);
    cuOptSolution cs_sol = nullptr;
    ASSERT_EQ(cuOptResolve(cs, settings, &cs_sol), CUOPT_SUCCESS);
    ASSERT_EQ(cuOptGetTerminationStatus(cs_sol, &status), CUOPT_SUCCESS);
    EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
    cuopt_float_t cs_obj = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(cs_sol, &cs_obj), CUOPT_SUCCESS);
    EXPECT_NEAR(cs_obj, expected_obj, kTol);
    cuOptDestroySolution(&cs_sol);
    cuOptDestroyProblem(&cs);
  };

  // Step 0: base solve. obj = 1.
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, 1.0, kTol);

  // Step 1: append column with cost -1, coef -1 in row 0. New row LHS:
  // x + y - z. Row >= 1, so z is bounded above by x + y - 1. With z in
  // [0, 10] the optimum is achieved at z = 10 (drives -z to -10), with
  // x + y = z + something >= 1; cheapest is x + y = 0 + 11 ... wait, var
  // bounds also matter. Let me just compute via reference rather than
  // reason in the comment.
  {
    const cuopt_float_t add_obj[]   = {-1.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {10.0};
    const cuopt_int_t add_col_off[] = {0, 1};
    const cuopt_int_t add_row_idx[] = {0};
    const cuopt_float_t add_val[]   = {-1.0};
    ASSERT_EQ(cuOptAddColumns(problem,
                              1,
                              add_obj,
                              add_v_lo,
                              add_v_hi,
                              add_col_off,
                              add_row_idx,
                              add_val,
                              nullptr),
              CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  {
    // Cold reference: min x+y-z s.t. x+y-z>=1, x,y in [0,10], z in [0,10].
    // For any z, need x + y >= 1 + z. Obj = (x+y) - z. Cheapest x+y is
    // 1 + z. So obj = (1+z) - z = 1. Optimum obj = 1, independent of z.
    const cuopt_float_t obj[]     = {1.0, 1.0, -1.0};
    const cuopt_int_t row_off[]   = {0, 3};
    const cuopt_int_t row_idx[]   = {0, 1, 2};
    const cuopt_float_t row_val[] = {1.0, 1.0, -1.0};
    const cuopt_float_t c_lo[]    = {1.0};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    solve_cold_reference(1, 3, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  // Step 2: append another column (cost 2, coef 1 in row 0). Splice again.
  {
    const cuopt_float_t add_obj[]   = {2.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {10.0};
    const cuopt_int_t add_col_off[] = {0, 1};
    const cuopt_int_t add_row_idx[] = {0};
    const cuopt_float_t add_val[]   = {1.0};
    ASSERT_EQ(cuOptAddColumns(problem,
                              1,
                              add_obj,
                              add_v_lo,
                              add_v_hi,
                              add_col_off,
                              add_row_idx,
                              add_val,
                              nullptr),
              CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  {
    const cuopt_float_t obj[]     = {1.0, 1.0, -1.0, 2.0};
    const cuopt_int_t row_off[]   = {0, 4};
    const cuopt_int_t row_idx[]   = {0, 1, 2, 3};
    const cuopt_float_t row_val[] = {1.0, 1.0, -1.0, 1.0};
    const cuopt_float_t c_lo[]    = {1.0};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS,
                                     CUOPT_CONTINUOUS};
    solve_cold_reference(1, 4, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  // Step 3: append a row x + 2y <= 5 (structural mutation).
  {
    const cuopt_float_t c_lo[]    = {-CUOPT_INFINITY};
    const cuopt_float_t c_hi[]    = {5.0};
    const cuopt_int_t row_off[]   = {0, 2};
    const cuopt_int_t col_idx[]   = {0, 1};
    const cuopt_float_t row_val[] = {1.0, 2.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, row_val), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  {
    const cuopt_float_t obj[]     = {1.0, 1.0, -1.0, 2.0};
    const cuopt_int_t row_off[]   = {0, 4, 6};
    const cuopt_int_t row_idx[]   = {0, 1, 2, 3, 0, 1};
    const cuopt_float_t row_val[] = {1.0, 1.0, -1.0, 1.0, 1.0, 2.0};
    const cuopt_float_t c_lo[]    = {1.0, -CUOPT_INFINITY};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY, 5.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS,
                                     CUOPT_CONTINUOUS};
    solve_cold_reference(2, 4, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  // Step 4: append another column (cost -3, coef 1 in both rows) after the
  // step-3 row add.
  {
    const cuopt_float_t add_obj[]   = {-3.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {10.0};
    const cuopt_int_t add_col_off[] = {0, 2};
    const cuopt_int_t add_row_idx[] = {0, 1};
    const cuopt_float_t add_val[]   = {1.0, 1.0};
    ASSERT_EQ(cuOptAddColumns(problem,
                              1,
                              add_obj,
                              add_v_lo,
                              add_v_hi,
                              add_col_off,
                              add_row_idx,
                              add_val,
                              nullptr),
              CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  {
    const cuopt_float_t obj[]     = {1.0, 1.0, -1.0, 2.0, -3.0};
    const cuopt_int_t row_off[]   = {0, 5, 8};
    const cuopt_int_t row_idx[]   = {0, 1, 2, 3, 4, 0, 1, 4};
    const cuopt_float_t row_val[] = {1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 2.0, 1.0};
    const cuopt_float_t c_lo[]    = {1.0, -CUOPT_INFINITY};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY, 5.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS,
                                     CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    solve_cold_reference(2, 5, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// delta_api.dual_simplex_presolve_rejected: the delta path requires presolve
// off on EVERY method, dual-simplex included. An explicit PSLP or Papilo
// presolver must be rejected with CUOPT_INVALID_ARGUMENT, and the rejection must
// leave the handle usable (turning presolve back off and resolving succeeds).
// ----------------------------------------------------------------------------
TEST(delta_api, dual_simplex_presolve_rejected)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  cuOptOptimizationProblem problem = nullptr;
  ASSERT_EQ(make_tiny_problem(&problem), CUOPT_SUCCESS);

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(cuOptCreateSolverSettings(&settings), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_DUAL_SIMPLEX),
            CUOPT_SUCCESS);

  cuOptSolution solution = nullptr;

  // PSLP on dual-simplex: rejected.
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_PSLP), CUOPT_SUCCESS);
  EXPECT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_INVALID_ARGUMENT);

  // Papilo on dual-simplex: also rejected (no per-method opt-in; presolve is
  // uniformly off on the delta path).
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_PAPILO),
            CUOPT_SUCCESS);
  EXPECT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_INVALID_ARGUMENT);

  // Handle stays usable: turn presolve off and resolve.
  ASSERT_EQ(cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_OFF), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// delta_api.pdlp_resolve_matches_cold_rebuild: the load-bearing PDLP delta
// correctness check (#24). A PDLP warm start must only change *speed*, never
// the answer. We drive a sequence of warm-started resolves through column
// appends and a row append, then assert the delta-path objective matches a
// cold, from-scratch cuOptCreateRangedProblem solve of the same accumulated
// problem. This guards the regression where injecting the full warm-start
// state (scaled-space iterate + restart/convergence bookkeeping from the prior
// matrix) made PDLP terminate early at a wrong point; the fix seeds only the
// iterate via set_initial_primal/dual_solution, leaving convergence tracking
// fresh.
//
// PDLP is first-order, so we solve to a tight tolerance and compare objectives
// with a loose absolute tolerance.
namespace {
cuopt_int_t make_tight_pdlp_settings(cuOptSolverSettings* settings_ptr)
{
  cuopt_int_t rc = cuOptCreateSolverSettings(settings_ptr);
  if (rc != CUOPT_SUCCESS) { return rc; }
  cuOptSetIntegerParameter(*settings_ptr, CUOPT_METHOD, CUOPT_METHOD_PDLP);
  cuOptSetFloatParameter(*settings_ptr, CUOPT_RELATIVE_GAP_TOLERANCE, 1e-8);
  cuOptSetFloatParameter(*settings_ptr, CUOPT_ABSOLUTE_GAP_TOLERANCE, 1e-10);
  cuOptSetFloatParameter(*settings_ptr, CUOPT_RELATIVE_PRIMAL_TOLERANCE, 1e-8);
  cuOptSetFloatParameter(*settings_ptr, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 1e-10);
  cuOptSetFloatParameter(*settings_ptr, CUOPT_RELATIVE_DUAL_TOLERANCE, 1e-8);
  cuOptSetFloatParameter(*settings_ptr, CUOPT_ABSOLUTE_DUAL_TOLERANCE, 1e-10);
  return CUOPT_SUCCESS;
}
}  // namespace

TEST(delta_api, pdlp_resolve_matches_cold_rebuild)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  constexpr cuopt_float_t kTol = 1e-3;

  // Base LP: min 5 x0  s.t.  x0 == 10,  x0 in [0,10].  Optimum: 50.
  cuOptOptimizationProblem problem = nullptr;
  {
    const cuopt_float_t obj[]     = {5.0};
    const cuopt_int_t row_off[]   = {0, 1};
    const cuopt_int_t row_idx[]   = {0};
    const cuopt_float_t row_val[] = {1.0};
    const cuopt_float_t c_lo[]    = {10.0};
    const cuopt_float_t c_hi[]    = {10.0};
    const cuopt_float_t v_lo[]    = {0.0};
    const cuopt_float_t v_hi[]    = {10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptCreateRangedProblem(1, 1, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &problem),
              CUOPT_SUCCESS);
  }

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_tight_pdlp_settings(&settings), CUOPT_SUCCESS);

  cuOptSolution solution = nullptr;
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  // Append cheaper columns one at a time, resolving (warm-started) after each.
  // Each new column shares row 0 (coeff 1) and is cheaper, so the optimum drops
  // 50 -> 40 -> 30 -> 20.
  const cuopt_float_t new_obj[] = {4.0, 3.0, 2.0};
  for (cuopt_float_t oc : new_obj) {
    const cuopt_float_t obj[]   = {oc};
    const cuopt_float_t v_lo[]  = {0.0};
    const cuopt_float_t v_hi[]  = {10.0};
    const cuopt_int_t col_off[] = {0, 1};
    const cuopt_int_t row_idx[] = {0};
    const cuopt_float_t val[]   = {1.0};
    const char vt[]             = {CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptAddColumns(problem, 1, obj, v_lo, v_hi, col_off, row_idx, val, vt),
              CUOPT_SUCCESS);
    ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  }

  // Append a row capping the cheapest variable x3 at 6, then resolve. Optimum
  // becomes 6*2 + 4*3 = 24 (x3=6, x2=4).
  {
    const cuopt_float_t c_lo[]  = {0.0};
    const cuopt_float_t c_hi[]  = {6.0};
    const cuopt_int_t row_off[] = {0, 1};
    const cuopt_int_t col_idx[] = {3};
    const cuopt_float_t val[]   = {1.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, val), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);

  cuopt_int_t status = -1;
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  cuopt_float_t delta_obj = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &delta_obj), CUOPT_SUCCESS);
  EXPECT_NEAR(delta_obj, 24.0, kTol);

  // Cold reference: build the fully-accumulated problem from scratch and solve
  // it with the same (fresh) PDLP settings. Must agree with the delta path.
  {
    const cuopt_float_t obj[]     = {5.0, 4.0, 3.0, 2.0};
    const cuopt_int_t row_off[]   = {0, 4, 5};
    const cuopt_int_t row_idx[]   = {0, 1, 2, 3, 3};
    const cuopt_float_t row_val[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    const cuopt_float_t c_lo[]    = {10.0, 0.0};
    const cuopt_float_t c_hi[]    = {10.0, 6.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0, 10.0};
    const char v_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

    cuOptOptimizationProblem cold = nullptr;
    ASSERT_EQ(cuOptCreateRangedProblem(2, 4, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &cold),
              CUOPT_SUCCESS);
    cuOptSolverSettings cold_settings = nullptr;
    ASSERT_EQ(make_tight_pdlp_settings(&cold_settings), CUOPT_SUCCESS);
    cuOptSolution cold_sol = nullptr;
    ASSERT_EQ(cuOptSolve(cold, cold_settings, &cold_sol), CUOPT_SUCCESS);
    cuopt_int_t cold_status = -1;
    ASSERT_EQ(cuOptGetTerminationStatus(cold_sol, &cold_status), CUOPT_SUCCESS);
    EXPECT_EQ(cold_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
    cuopt_float_t cold_obj = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(cold_sol, &cold_obj), CUOPT_SUCCESS);

    EXPECT_NEAR(delta_obj, cold_obj, kTol);

    if (cold_sol != nullptr) { cuOptDestroySolution(&cold_sol); }
    cuOptDestroySolverSettings(&cold_settings);
    cuOptDestroyProblem(&cold);
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}
// ----------------------------------------------------------------------------
// delta_api.dual_simplex_resolve_matches_cold_rebuild: the dual-simplex
// warm-state delta correctness check (#22). A warm-started dual-simplex resolve
// persists the converted LP + optimal basis across resolves and re-optimizes
// from the previous optimum on a tail-only structural extension (new columns
// enter nonbasic, new '<=' rows enter as cuts). It must only change *speed*,
// never the answer: at each step the delta-path objective must match a cold,
// from-scratch cuOptCreateRangedProblem solve of the same accumulated problem.
//
// This test exercises the mcfcg-supported case and all three regression fixes:
//   * 'E'/'L' base rows only (no '>=' — the warm path forces a cold rebuild on
//     any '>=' row because convert_user_problem negates 'G' rows, so an
//     appended column would land sign-flipped). mcfcg masters use 'E' (demand /
//     convexity) + 'L' (capacity) rows.
//   * an appended column with bounds [0, +inf) (a standard Dantzig-Wolfe master
//     column). An attractive [0,+inf) column makes the transplanted basis
//     dual-infeasible; the fix runs a dual phase-1 before phase-2 so the warm
//     terminus is truly optimal, not a parked-at-lower-bound point.
//   * a step that appends a column AND a row in the SAME resolve where the new
//     column has a nonzero in the NEW row (mcfcg's "new path uses an arc whose
//     capacity row is added the same round" case). The fix filters
//     appended-column entries to the old rows so add_cuts (not the column
//     splice) folds in the new-row coefficient — no double-count, no OOB.
//
// Sequence (single persistent handle so the warm basis is reused):
//   step 0: base solve. min x0 s.t. x0 == 5, x0 in [0,10] -> obj 5.
//   step 1: append x1 (cost -1, coef 1 in 'E' row 0, bounds [0,+INF)) -> obj -5.
//   step 2: append x2 (cost 2, coef 1 in row 0, bounds [0,10])        -> obj -5.
//   step 3: append '<=' row  x1 <= 3                                  -> obj -1.
//   step 4: append x3 (cost -4, [0,10], coef 1 in row 0) AND row x3<=2 in the
//           SAME resolve, x3 nonzero in BOTH row 0 and the new row     -> obj -11.
// At every step the warm delta-path objective is checked against an independent
// cold cuOptCreateRangedProblem solve of the same accumulated problem.
// ----------------------------------------------------------------------------
TEST(delta_api, dual_simplex_resolve_matches_cold_rebuild)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  // Base LP: min x0 s.t. x0 == 5 (an 'E' row), x0 in [0,10]. Optimum obj = 5.
  cuOptOptimizationProblem problem = nullptr;
  {
    const cuopt_float_t obj[]     = {1.0};
    const cuopt_int_t row_off[]   = {0, 1};
    const cuopt_int_t row_idx[]   = {0};
    const cuopt_float_t row_val[] = {1.0};
    const cuopt_float_t c_lo[]    = {5.0};
    const cuopt_float_t c_hi[]    = {5.0};
    const cuopt_float_t v_lo[]    = {0.0};
    const cuopt_float_t v_hi[]    = {10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptCreateRangedProblem(1, 1, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &problem),
              CUOPT_SUCCESS);
  }

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);

  cuOptSolution solution       = nullptr;
  cuopt_int_t status           = -1;
  cuopt_float_t obj_val        = std::numeric_limits<cuopt_float_t>::quiet_NaN();
  constexpr cuopt_float_t kTol = 1e-4;

  // Cold reference: build the equivalent accumulated LP from scratch with a
  // FRESH settings handle and confirm the objective matches the delta path.
  auto solve_cold_reference = [&status, kTol](cuopt_int_t n_cons,
                                              cuopt_int_t n_vars,
                                              const cuopt_float_t* obj,
                                              const cuopt_int_t* row_off,
                                              const cuopt_int_t* row_idx,
                                              const cuopt_float_t* row_val,
                                              const cuopt_float_t* c_lo,
                                              const cuopt_float_t* c_hi,
                                              const cuopt_float_t* v_lo,
                                              const cuopt_float_t* v_hi,
                                              const char* v_types,
                                              cuopt_float_t expected_obj) {
    cuOptOptimizationProblem cs = nullptr;
    ASSERT_EQ(cuOptCreateRangedProblem(n_cons, n_vars, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx,
                                       row_val, c_lo, c_hi, v_lo, v_hi, v_types, &cs),
              CUOPT_SUCCESS);
    cuOptSolverSettings cs_settings = nullptr;
    ASSERT_EQ(make_dual_simplex_settings(&cs_settings), CUOPT_SUCCESS);
    cuOptSolution cs_sol = nullptr;
    ASSERT_EQ(cuOptSolve(cs, cs_settings, &cs_sol), CUOPT_SUCCESS);
    ASSERT_EQ(cuOptGetTerminationStatus(cs_sol, &status), CUOPT_SUCCESS);
    EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
    cuopt_float_t cs_obj = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(cs_sol, &cs_obj), CUOPT_SUCCESS);
    EXPECT_NEAR(cs_obj, expected_obj, kTol);
    cuOptDestroySolution(&cs_sol);
    cuOptDestroySolverSettings(&cs_settings);
    cuOptDestroyProblem(&cs);
  };

  // Step 0: base solve -> obj 5 (cold-captures the basis).
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, 5.0, kTol);

  // Step 1: append x1 (cost -1, coef 1 in 'E' row 0, bounds [0,+INF)). The
  // attractive [0,+inf) column makes the warm basis dual-infeasible; the dual
  // phase-1 fix must drive it in. min x0 - x1 s.t. x0 + x1 == 5, x0,x1>=0,
  // x1 unbounded above -> x1 = 5, x0 = 0, obj = -5.
  {
    const cuopt_float_t add_obj[]   = {-1.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {CUOPT_INFINITY};
    const cuopt_int_t add_col_off[] = {0, 1};
    const cuopt_int_t add_row_idx[] = {0};
    const cuopt_float_t add_val[]   = {1.0};
    ASSERT_EQ(cuOptAddColumns(problem, 1, add_obj, add_v_lo, add_v_hi, add_col_off, add_row_idx,
                              add_val, nullptr),
              CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, -5.0, kTol);
  {
    const cuopt_float_t obj[]     = {1.0, -1.0};
    const cuopt_int_t row_off[]   = {0, 2};
    const cuopt_int_t row_idx[]   = {0, 1};
    const cuopt_float_t row_val[] = {1.0, 1.0};
    const cuopt_float_t c_lo[]    = {5.0};
    const cuopt_float_t c_hi[]    = {5.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, CUOPT_INFINITY};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    solve_cold_reference(1, 2, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  // Step 2: append x2 (cost 2, coef 1 in row 0, bounds [0,10]). x2 stays 0;
  // obj unchanged at -5.
  {
    const cuopt_float_t add_obj[]   = {2.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {10.0};
    const cuopt_int_t add_col_off[] = {0, 1};
    const cuopt_int_t add_row_idx[] = {0};
    const cuopt_float_t add_val[]   = {1.0};
    ASSERT_EQ(cuOptAddColumns(problem, 1, add_obj, add_v_lo, add_v_hi, add_col_off, add_row_idx,
                              add_val, nullptr),
              CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, -5.0, kTol);
  {
    const cuopt_float_t obj[]     = {1.0, -1.0, 2.0};
    const cuopt_int_t row_off[]   = {0, 3};
    const cuopt_int_t row_idx[]   = {0, 1, 2};
    const cuopt_float_t row_val[] = {1.0, 1.0, 1.0};
    const cuopt_float_t c_lo[]    = {5.0};
    const cuopt_float_t c_hi[]    = {5.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, CUOPT_INFINITY, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    solve_cold_reference(1, 3, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  // Step 3: append a '<=' row x1 <= 3 (add_cuts path). Now x1 capped at 3 ->
  // x0 = 5 - x1 = 2, obj = x0 - x1 = 2 - 3 = -1.
  {
    const cuopt_float_t c_lo[]    = {-CUOPT_INFINITY};
    const cuopt_float_t c_hi[]    = {3.0};
    const cuopt_int_t row_off[]   = {0, 1};
    const cuopt_int_t col_idx[]   = {1};
    const cuopt_float_t row_val[] = {1.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, row_val), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, -1.0, kTol);
  {
    const cuopt_float_t obj[]     = {1.0, -1.0, 2.0};
    const cuopt_int_t row_off[]   = {0, 3, 4};
    const cuopt_int_t row_idx[]   = {0, 1, 2, 1};
    const cuopt_float_t row_val[] = {1.0, 1.0, 1.0, 1.0};
    const cuopt_float_t c_lo[]    = {5.0, -CUOPT_INFINITY};
    const cuopt_float_t c_hi[]    = {5.0, 3.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, CUOPT_INFINITY, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    solve_cold_reference(2, 3, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  // Step 4 (BUG 2 scenario): append x3 (cost -4, [0,10], coef 1 in 'E' row 0)
  // AND append row x3 <= 2 in the SAME resolve. x3 has a nonzero in BOTH the
  // existing row 0 and the newly appended row -> the column-splice must NOT copy
  // the new-row entry (add_cuts folds it in). Optimum: x3 = 2 (cost -4, capped),
  // x1 = 3 (cost -1, capped), x0 = 5 - 3 - 2 = 0, x2 = 0.
  // obj = x0 - x1 + 2 x2 - 4 x3 = 0 - 3 + 0 - 8 = -11.
  {
    const cuopt_float_t add_obj[]   = {-4.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {10.0};
    const cuopt_int_t add_col_off[] = {0, 1};
    const cuopt_int_t add_row_idx[] = {0};
    const cuopt_float_t add_val[]   = {1.0};
    ASSERT_EQ(cuOptAddColumns(problem, 1, add_obj, add_v_lo, add_v_hi, add_col_off, add_row_idx,
                              add_val, nullptr),
              CUOPT_SUCCESS);
    // Row x3 <= 2 in the SAME resolve, referencing the just-added column 3.
    const cuopt_float_t c_lo[]    = {-CUOPT_INFINITY};
    const cuopt_float_t c_hi[]    = {2.0};
    const cuopt_int_t row_off[]   = {0, 1};
    const cuopt_int_t col_idx[]   = {3};
    const cuopt_float_t row_val[] = {1.0};
    ASSERT_EQ(cuOptAddRows(problem, 1, c_lo, c_hi, row_off, col_idx, row_val), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, -11.0, kTol);
  {
    const cuopt_float_t obj[]     = {1.0, -1.0, 2.0, -4.0};
    const cuopt_int_t row_off[]   = {0, 4, 5, 6};
    const cuopt_int_t row_idx[]   = {0, 1, 2, 3, 1, 3};
    const cuopt_float_t row_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const cuopt_float_t c_lo[]    = {5.0, -CUOPT_INFINITY, -CUOPT_INFINITY};
    const cuopt_float_t c_hi[]    = {5.0, 3.0, 2.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, CUOPT_INFINITY, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS,
                                     CUOPT_CONTINUOUS};
    solve_cold_reference(3, 4, obj, row_off, row_idx, row_val, c_lo, c_hi, v_lo, v_hi, v_types,
                         obj_val);
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// delta_api.delete_resolve_matches_cold_rebuild: a resolve sequence that DELETES
// a column and a row must reach the same objective as a from-scratch solve of
// the compacted problem. This is the correctness guard for the host-side CSR
// delete compaction AND the PDLP warm-start primal/dual vector compaction
// (compact_warm_start_columns / compact_warm_start_rows), which the append-only
// parity tests never exercise. PDLP is used so the warm-start vectors exist and
// are compacted on the delete.
// ----------------------------------------------------------------------------
TEST(delta_api, delete_resolve_matches_cold_rebuild)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  // Base LP (4 vars, 3 '>=' rows):
  //   min  x0 + 5 x1 + 2 x2 + 4 x3
  //   s.t. x0 + x1       >= 2   (r0)
  //        x1 + x2       >= 2   (r1)
  //        x2 + x3       >= 2   (r2)
  //   0 <= xi <= 10
  cuOptOptimizationProblem problem = nullptr;
  {
    const cuopt_float_t obj[]     = {1.0, 5.0, 2.0, 4.0};
    const cuopt_int_t row_off[]   = {0, 2, 4, 6};
    const cuopt_int_t row_idx[]   = {0, 1, 1, 2, 2, 3};
    const cuopt_float_t row_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const cuopt_float_t c_lo[]    = {2.0, 2.0, 2.0};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS,
                                     CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptCreateRangedProblem(3, 4, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &problem),
              CUOPT_SUCCESS);
  }

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_tight_pdlp_settings(&settings), CUOPT_SUCCESS);

  constexpr cuopt_float_t kTol = 1e-4;
  cuOptSolution solution       = nullptr;
  cuopt_int_t status           = -1;
  cuopt_float_t obj_val        = std::numeric_limits<cuopt_float_t>::quiet_NaN();

  // Step 0: base solve seeds the PDLP warm-start vectors (sized to 4 cols / 3 rows).
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);

  // Step 1: delete column x1 and row r1. The warm-start primal vector compacts
  // from 4 -> 3, the dual vector from 3 -> 2, and the CSR matrix is rewritten.
  {
    const cuopt_int_t del_col[] = {1};
    ASSERT_EQ(cuOptDeleteColumns(problem, 1, del_col), CUOPT_SUCCESS);
    const cuopt_int_t del_row[] = {1};
    ASSERT_EQ(cuOptDeleteRows(problem, 1, del_row), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);

  // Post-delete problem (surviving vars x0,x2,x3 -> 0,1,2; surviving rows r0,r2):
  //   min  x0 + 2 x2 + 4 x3
  //   s.t. x0       >= 2   (r0, x1 dropped)
  //        x2 + x3  >= 2   (r2)
  {
    const cuopt_float_t obj[]     = {1.0, 2.0, 4.0};
    const cuopt_int_t row_off[]   = {0, 1, 3};
    const cuopt_int_t row_idx[]   = {0, 1, 2};
    const cuopt_float_t row_val[] = {1.0, 1.0, 1.0};
    const cuopt_float_t c_lo[]    = {2.0, 2.0};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY, CUOPT_INFINITY};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    cuOptOptimizationProblem cold = nullptr;
    ASSERT_EQ(cuOptCreateRangedProblem(2, 3, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &cold),
              CUOPT_SUCCESS);
    cuOptSolverSettings cold_settings = nullptr;
    ASSERT_EQ(make_tight_pdlp_settings(&cold_settings), CUOPT_SUCCESS);
    cuOptSolution cold_sol = nullptr;
    ASSERT_EQ(cuOptSolve(cold, cold_settings, &cold_sol), CUOPT_SUCCESS);
    cuopt_float_t cold_obj = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(cold_sol, &cold_obj), CUOPT_SUCCESS);
    EXPECT_NEAR(obj_val, cold_obj, kTol);
    cuOptDestroySolution(&cold_sol);
    cuOptDestroySolverSettings(&cold_settings);
    cuOptDestroyProblem(&cold);
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// delta_api.set_objective_resolve_matches_cold_rebuild: a resolve after
// cuOptSetObjectiveCoefficients changes which solution is optimal, and the delta
// path must reach the same objective as a from-scratch solve with the new costs.
// The append-only parity tests never mutate an existing objective coefficient;
// this is the regression guard for the objective-scatter path (and for the class
// of bug where a resolve optimizes a stale objective).
// ----------------------------------------------------------------------------
TEST(delta_api, set_objective_resolve_matches_cold_rebuild)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  // Base: min 1*x0 + 3*x1 s.t. x0 + x1 >= 2, 0 <= xi <= 10. Optimum: x0=2, obj=2.
  cuOptOptimizationProblem problem = nullptr;
  {
    const cuopt_float_t obj[]     = {1.0, 3.0};
    const cuopt_int_t row_off[]   = {0, 2};
    const cuopt_int_t row_idx[]   = {0, 1};
    const cuopt_float_t row_val[] = {1.0, 1.0};
    const cuopt_float_t c_lo[]    = {2.0};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY};
    const cuopt_float_t v_lo[]    = {0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptCreateRangedProblem(1, 2, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &problem),
              CUOPT_SUCCESS);
  }

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);

  constexpr cuopt_float_t kTol = 1e-6;
  cuOptSolution solution       = nullptr;
  cuopt_float_t obj_val        = std::numeric_limits<cuopt_float_t>::quiet_NaN();

  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, 2.0, kTol);

  // Flip x0's cost to 5 so x1 (cost 3) becomes cheaper. New optimum: x1=2, obj=6.
  {
    const cuopt_int_t idx[]     = {0};
    const cuopt_float_t vals[]  = {5.0};
    ASSERT_EQ(cuOptSetObjectiveCoefficients(problem, 1, idx, vals), CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, 6.0, kTol);

  // Cold reference: same constraints with the updated objective.
  {
    const cuopt_float_t obj[]     = {5.0, 3.0};
    const cuopt_int_t row_off[]   = {0, 2};
    const cuopt_int_t row_idx[]   = {0, 1};
    const cuopt_float_t row_val[] = {1.0, 1.0};
    const cuopt_float_t c_lo[]    = {2.0};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY};
    const cuopt_float_t v_lo[]    = {0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    cuOptOptimizationProblem cold = nullptr;
    ASSERT_EQ(cuOptCreateRangedProblem(1, 2, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &cold),
              CUOPT_SUCCESS);
    cuOptSolverSettings cold_settings = nullptr;
    ASSERT_EQ(make_dual_simplex_settings(&cold_settings), CUOPT_SUCCESS);
    cuOptSolution cold_sol = nullptr;
    ASSERT_EQ(cuOptSolve(cold, cold_settings, &cold_sol), CUOPT_SUCCESS);
    cuopt_float_t cold_obj = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(cold_sol, &cold_obj), CUOPT_SUCCESS);
    EXPECT_NEAR(obj_val, cold_obj, kTol);
    cuOptDestroySolution(&cold_sol);
    cuOptDestroySolverSettings(&cold_settings);
    cuOptDestroyProblem(&cold);
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}

// ----------------------------------------------------------------------------
// delta_api.non_ranged_create_then_mutate: a problem created via the NON-ranged
// cuOptCreateProblem (constraint_sense + rhs) must mutate and resolve correctly.
// The first mutator triggers ensure_ranged_representation, which converts the
// row_types/rhs representation to ranged lower/upper bounds via
// optimization_problem_t::clear_row_types / clear_constraint_bounds. Every other
// test builds with cuOptCreateRangedProblem, so this is the only coverage of
// that conversion path.
// ----------------------------------------------------------------------------
TEST(delta_api, non_ranged_create_then_mutate)
{
  if (!gpu_available()) { GTEST_SKIP() << "no CUDA device — delta API requires GPU"; }

  // Non-ranged base: min x0 + x1 s.t. x0 + x1 >= 2 ('G' row, rhs 2), 0 <= xi <= 10.
  cuOptOptimizationProblem problem = nullptr;
  {
    const cuopt_float_t obj[]     = {1.0, 1.0};
    const cuopt_int_t row_off[]   = {0, 2};
    const cuopt_int_t row_idx[]   = {0, 1};
    const cuopt_float_t row_val[] = {1.0, 1.0};
    const char sense[]            = {'G'};
    const cuopt_float_t rhs[]     = {2.0};
    const cuopt_float_t v_lo[]    = {0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    ASSERT_EQ(cuOptCreateProblem(1, 2, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val, sense,
                                 rhs, v_lo, v_hi, v_types, &problem),
              CUOPT_SUCCESS);
  }

  cuOptSolverSettings settings = nullptr;
  ASSERT_EQ(make_dual_simplex_settings(&settings), CUOPT_SUCCESS);

  constexpr cuopt_float_t kTol = 1e-6;
  cuOptSolution solution       = nullptr;
  cuopt_int_t status           = -1;
  cuopt_float_t obj_val        = std::numeric_limits<cuopt_float_t>::quiet_NaN();

  // Append x2 (cost -1, coef 1 in row 0, bounds [0,10]). This first mutation
  // normalises the non-ranged problem to ranged form (clear_row_types /
  // clear_constraint_bounds), then appends. New optimum: x2 = 10 (cost -1),
  // x0 = x1 = 0, row 0 satisfied (x2 = 10 >= 2) -> obj = -10.
  {
    const cuopt_float_t add_obj[]   = {-1.0};
    const cuopt_float_t add_v_lo[]  = {0.0};
    const cuopt_float_t add_v_hi[]  = {10.0};
    const cuopt_int_t add_col_off[] = {0, 1};
    const cuopt_int_t add_row_idx[] = {0};
    const cuopt_float_t add_val[]   = {1.0};
    ASSERT_EQ(cuOptAddColumns(problem, 1, add_obj, add_v_lo, add_v_hi, add_col_off, add_row_idx,
                              add_val, nullptr),
              CUOPT_SUCCESS);
  }
  ASSERT_EQ(cuOptResolve(problem, settings, &solution), CUOPT_SUCCESS);
  ASSERT_EQ(cuOptGetTerminationStatus(solution, &status), CUOPT_SUCCESS);
  EXPECT_EQ(status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  ASSERT_EQ(cuOptGetObjectiveValue(solution, &obj_val), CUOPT_SUCCESS);
  EXPECT_NEAR(obj_val, -10.0, kTol);

  // Cold reference: the equivalent accumulated problem built ranged from scratch.
  {
    const cuopt_float_t obj[]     = {1.0, 1.0, -1.0};
    const cuopt_int_t row_off[]   = {0, 3};
    const cuopt_int_t row_idx[]   = {0, 1, 2};
    const cuopt_float_t row_val[] = {1.0, 1.0, 1.0};
    const cuopt_float_t c_lo[]    = {2.0};
    const cuopt_float_t c_hi[]    = {CUOPT_INFINITY};
    const cuopt_float_t v_lo[]    = {0.0, 0.0, 0.0};
    const cuopt_float_t v_hi[]    = {10.0, 10.0, 10.0};
    const char v_types[]          = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
    cuOptOptimizationProblem cold = nullptr;
    ASSERT_EQ(cuOptCreateRangedProblem(1, 3, CUOPT_MINIMIZE, 0.0, obj, row_off, row_idx, row_val,
                                       c_lo, c_hi, v_lo, v_hi, v_types, &cold),
              CUOPT_SUCCESS);
    cuOptSolverSettings cold_settings = nullptr;
    ASSERT_EQ(make_dual_simplex_settings(&cold_settings), CUOPT_SUCCESS);
    cuOptSolution cold_sol = nullptr;
    ASSERT_EQ(cuOptSolve(cold, cold_settings, &cold_sol), CUOPT_SUCCESS);
    cuopt_float_t cold_obj = std::numeric_limits<cuopt_float_t>::quiet_NaN();
    ASSERT_EQ(cuOptGetObjectiveValue(cold_sol, &cold_obj), CUOPT_SUCCESS);
    EXPECT_NEAR(obj_val, cold_obj, kTol);
    cuOptDestroySolution(&cold_sol);
    cuOptDestroySolverSettings(&cold_settings);
    cuOptDestroyProblem(&cold);
  }

  if (solution != nullptr) { cuOptDestroySolution(&solution); }
  cuOptDestroySolverSettings(&settings);
  cuOptDestroyProblem(&problem);
}
