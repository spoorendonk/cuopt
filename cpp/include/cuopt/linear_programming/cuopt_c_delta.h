/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#ifndef CUOPT_C_DELTA_API_H
#define CUOPT_C_DELTA_API_H

#include <cuopt/linear_programming/cuopt_c.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file cuopt_c_delta.h
 * @brief Incremental problem updates on a persistent ``cuOptOptimizationProblem``.
 *
 * The base ``cuOptCreate*Problem`` / ``cuOptSolve`` / ``cuOptDestroyProblem``
 * cycle requires rebuilding the LP and re-uploading to the GPU on every
 * resolve. Column generation and related workloads need an incremental path:
 * create a problem once, mutate it (add/delete cols/rows, bump costs),
 * resolve, repeat.
 *
 * Each mutator below operates on the existing problem handle. The underlying
 * solver (dual simplex / barrier / PDLP) is responsible for keeping its own
 * state consistent with the mutation — warm-starting where the method
 * supports it, rebuilding the minimum required structure otherwise.
 *
 * @par Lazy-rebuild semantics
 * Each mutator buffers its inputs on the host and returns without touching
 * the GPU. ``cuOptResolve`` drains the buffer in arrival order and applies
 * the mutations against the persistent device problem just before solving.
 * Consequence:
 *   - Public getters (``cuOptGetNumVariables`` / ``cuOptGetNumConstraints`` /
 *     ``cuOptGetNumNonZeros`` / ``cuOptGetObjectiveCoefficients`` /
 *     ``cuOptGetConstraintMatrix`` / etc.) reflect the LAST-RESOLVED state of
 *     the problem, NOT the post-pending state. Callers that need to inspect
 *     the post-mutation shape must call ``cuOptResolve`` first.
 *   - Index validation in mutators is performed against logical (post-pending)
 *     sizes maintained alongside the buffer, so e.g. ``cuOptAddRows`` can
 *     refer to columns added by a previous, still-pending ``cuOptAddColumns``
 *     within the same batch.
 *
 * @par Thread-safety
 * A ``cuOptOptimizationProblem`` handle is not safe for concurrent use from
 * multiple threads. Serialise mutations and resolves on a single handle.
 *
 * @par Index width
 * A persistent handle accumulates columns, rows, and nonzeros across many
 * resolves. All sizes and CSR offsets are ``cuopt_int_t`` (32-bit in the
 * default build), so a long-lived column-generation handle whose total nonzero
 * count would exceed ``INT_MAX`` requires a 64-bit ``cuopt_int_t`` build; the
 * same limit applies to the non-delta create/solve path.
 */

/**
 * @brief Append columns to a persistent optimization problem.
 *
 * The new columns are described in compressed sparse column (CSC) form:
 * ``column_starts[i]`` is the offset into ``row_indices`` / ``values`` where
 * the i-th new column's entries begin, with the i-th column ending at
 * ``column_starts[i+1]``.
 *
 * If the new columns have no coefficients, ``row_indices`` and ``values``
 * may be NULL (iff ``column_starts[num_columns] == 0``).
 *
 * @param[in] problem The persistent optimization problem handle.
 * @param[in] num_columns Number of columns to append. Pass 0 for a no-op.
 * @param[in] objective_coefficients Pointer to an array of type
 *            ``cuopt_float_t`` of size ``num_columns`` containing the new
 *            columns' objective coefficients.
 * @param[in] variable_lower_bounds Pointer to an array of type
 *            ``cuopt_float_t`` of size ``num_columns`` containing the new
 *            columns' lower bounds.
 * @param[in] variable_upper_bounds Pointer to an array of type
 *            ``cuopt_float_t`` of size ``num_columns`` containing the new
 *            columns' upper bounds.
 * @param[in] column_starts Pointer to an array of type ``cuopt_int_t`` of
 *            size ``num_columns + 1`` describing the CSC offsets.
 *            ``column_starts[num_columns]`` must equal the total number of
 *            new entries.
 * @param[in] row_indices Pointer to an array of type ``cuopt_int_t`` of
 *            size ``column_starts[num_columns]`` giving the row index of
 *            each new entry. May be NULL when there are no entries.
 * @param[in] values Pointer to an array of type ``cuopt_float_t`` of size
 *            ``column_starts[num_columns]`` giving the value of each new
 *            entry. May be NULL when there are no entries.
 * @param[in] variable_types Pointer to an array of type ``char`` of size
 *            ``num_columns`` containing the variable types
 *            (``CUOPT_CONTINUOUS`` or ``CUOPT_INTEGER``). Pass NULL to
 *            default to ``CUOPT_CONTINUOUS``.
 * @return CUOPT_SUCCESS on success, a cuOpt error code otherwise.
 */
cuopt_int_t cuOptAddColumns(cuOptOptimizationProblem problem,
                            cuopt_int_t num_columns,
                            const cuopt_float_t* objective_coefficients,
                            const cuopt_float_t* variable_lower_bounds,
                            const cuopt_float_t* variable_upper_bounds,
                            const cuopt_int_t* column_starts,
                            const cuopt_int_t* row_indices,
                            const cuopt_float_t* values,
                            const char* variable_types);

/**
 * @brief Append rows (constraints) to a persistent optimization problem.
 *
 * The new rows are described in compressed sparse row (CSR) form:
 * ``row_starts[i]`` is the offset into ``column_indices`` / ``values`` where
 * the i-th new row's entries begin, with the i-th row ending at
 * ``row_starts[i+1]``.
 *
 * Row bounds follow the existing ``cuOptCreateRangedProblem`` convention:
 * ``constraint_lower_bounds[i]`` and ``constraint_upper_bounds[i]`` together
 * encode equality / inequality / ranged constraints.
 *
 * For each new row, the ``column_indices`` slice
 * ``[row_starts[i], row_starts[i+1])`` MUST be sorted strictly ascending
 * with no duplicates. cuOpt does not re-sort delta-appended rows on resolve,
 * so callers are responsible for supplying sorted column indices.
 *
 * @param[in] problem The persistent optimization problem handle.
 * @param[in] num_rows Number of rows to append. Pass 0 for a no-op.
 * @param[in] constraint_lower_bounds Pointer to an array of type
 *            ``cuopt_float_t`` of size ``num_rows`` giving each new row's
 *            lower bound.
 * @param[in] constraint_upper_bounds Pointer to an array of type
 *            ``cuopt_float_t`` of size ``num_rows`` giving each new row's
 *            upper bound.
 * @param[in] row_starts Pointer to an array of type ``cuopt_int_t`` of size
 *            ``num_rows + 1`` describing the CSR offsets.
 *            ``row_starts[num_rows]`` must equal the total number of new
 *            entries.
 * @param[in] column_indices Pointer to an array of type ``cuopt_int_t`` of
 *            size ``row_starts[num_rows]`` giving the column index of each
 *            new entry. Within each row's slice the indices must be sorted
 *            strictly ascending.
 * @param[in] values Pointer to an array of type ``cuopt_float_t`` of size
 *            ``row_starts[num_rows]`` giving the value of each new entry.
 * @return CUOPT_SUCCESS on success, a cuOpt error code otherwise.
 */
cuopt_int_t cuOptAddRows(cuOptOptimizationProblem problem,
                         cuopt_int_t num_rows,
                         const cuopt_float_t* constraint_lower_bounds,
                         const cuopt_float_t* constraint_upper_bounds,
                         const cuopt_int_t* row_starts,
                         const cuopt_int_t* column_indices,
                         const cuopt_float_t* values);

/**
 * @brief Delete columns by sparse index list.
 *
 * Surviving columns compact in input order (the i-th surviving column keeps
 * its relative position among survivors), so a caller that applies the
 * identical compaction to its own index bookkeeping stays in sync with cuOpt.
 *
 * The "sorted unique" requirement lets cuOpt validate in O(num_indices) and
 * lets the caller skip a separate sort step in the common case where they
 * already track inactive columns in order.
 *
 * @param[in] problem The persistent optimization problem handle.
 * @param[in] num_indices Number of column indices to delete. Pass 0 for a
 *            no-op.
 * @param[in] indices Pointer to an array of type ``cuopt_int_t`` of size
 *            ``num_indices``. Must be sorted strictly ascending with no
 *            duplicates and every entry in ``[0, num_variables)``.
 * @return CUOPT_SUCCESS on success, CUOPT_INVALID_ARGUMENT if ``indices``
 *         violates the sorted-unique-in-range contract, a cuOpt error code
 *         otherwise.
 */
cuopt_int_t cuOptDeleteColumns(cuOptOptimizationProblem problem,
                               cuopt_int_t num_indices,
                               const cuopt_int_t* indices);

/**
 * @brief Delete rows by sparse index list.
 *
 * Same contract as ``cuOptDeleteColumns`` (sorted strictly ascending, no
 * duplicates, each in ``[0, num_constraints)``). Survivors compact in input
 * order.
 *
 * @param[in] problem The persistent optimization problem handle.
 * @param[in] num_indices Number of row indices to delete. Pass 0 for a no-op.
 * @param[in] indices Pointer to an array of type ``cuopt_int_t`` of size
 *            ``num_indices``. Must be sorted strictly ascending with no
 *            duplicates and every entry in ``[0, num_constraints)``.
 * @return CUOPT_SUCCESS on success, CUOPT_INVALID_ARGUMENT if ``indices``
 *         violates the contract, a cuOpt error code otherwise.
 */
cuopt_int_t cuOptDeleteRows(cuOptOptimizationProblem problem,
                            cuopt_int_t num_indices,
                            const cuopt_int_t* indices);

/**
 * @brief Update objective coefficients in a single device-side scatter.
 *
 * Issues one host-to-device copy of (indices, values) plus one scatter
 * kernel. Intended for column-generation slack-bump loops where the number
 * of touched coefficients per CG iteration is small but non-trivial: the
 * per-iter sync count is 1 regardless of how many coefficients change.
 *
 * Indices are not required to be sorted, and may repeat — repeated indices
 * result in last-written-wins. Pass ``num_indices`` == 1 in lieu of a scalar
 * setter.
 *
 * @param[in] problem The persistent optimization problem handle.
 * @param[in] num_indices Number of objective coefficients to update. Pass 0
 *            for a no-op.
 * @param[in] indices Pointer to an array of type ``cuopt_int_t`` of size
 *            ``num_indices``. Each entry must be in ``[0, num_variables)``.
 * @param[in] values Pointer to an array of type ``cuopt_float_t`` of size
 *            ``num_indices`` giving the new objective coefficient at each
 *            index.
 * @return CUOPT_SUCCESS on success, a cuOpt error code otherwise.
 */
cuopt_int_t cuOptSetObjectiveCoefficients(cuOptOptimizationProblem problem,
                                          cuopt_int_t num_indices,
                                          const cuopt_int_t* indices,
                                          const cuopt_float_t* values);

/**
 * @brief Resolve the (possibly mutated) problem.
 *
 * Reuses solver state (basis for dual simplex, warm-start iterates for
 * PDLP) where the configured method supports it. Barrier always restarts
 * from the analytic center — no warm start is applicable. The caller pays
 * the persistent-handle cost (no create/destroy per resolve); eliminating
 * the internal per-resolve GPU problem reconstruction in barrier and PDLP
 * is a planned perf follow-up.
 *
 * @par Solution-handle ownership
 * ``previous_solution_ptr`` is an in/out parameter. On the first call it
 * should point to NULL and cuOpt will allocate a fresh ``cuOptSolution``.
 * On subsequent calls, pass the pointer from the previous resolve — cuOpt
 * may reuse the existing object in place or destroy it and allocate a
 * replacement. If the implementation reallocates, it MUST destroy the prior
 * object first; the caller never needs to ``cuOptDestroySolution`` an old
 * handle it passed in.
 *
 * On non-success return (anything other than ``CUOPT_SUCCESS``), the object
 * referenced by ``*previous_solution_ptr`` is left unchanged and remains
 * owned by the caller — safe to call ``cuOptDestroySolution`` on, safe to
 * pass back to a later ``cuOptResolve``.
 *
 * @param[in] problem The persistent optimization problem handle.
 * @param[in] settings The solver settings handle.
 * @param[in,out] previous_solution_ptr Pointer to the prior solution handle
 *            (may be NULL on first call); receives the new solution handle
 *            on success.
 * @return CUOPT_SUCCESS on success, a cuOpt error code otherwise.
 */
cuopt_int_t cuOptResolve(cuOptOptimizationProblem problem,
                         cuOptSolverSettings settings,
                         cuOptSolution* previous_solution_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CUOPT_C_DELTA_API_H
