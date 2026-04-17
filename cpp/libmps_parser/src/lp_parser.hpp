/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace cuopt::mps_parser {

/**
 * @brief Parser for the LP format.
 *
 * The class is a thin holder for the parsed problem data. All parsing
 * machinery (tokenizer, expression/section parsers, token types) lives in
 * src/lp_parser.cpp and is never exposed.
 *
 * The public fields mirror mps_parser_t so the two parsers share a single
 * finalization path (see src/parser_finalize.hpp) and so tests and tools
 * can introspect the same shape of intermediate data from either parser.
 */
template <typename i_t, typename f_t>
class lp_parser_t {
 public:
  // Parses `file` and populates `problem`.
  lp_parser_t(mps_data_model_t<i_t, f_t>& problem, const std::string& file);

  // Intermediate parsed problem data (mirrors mps_parser_t's public fields).
  std::string problem_name{};
  std::vector<std::string> row_names{};
  std::vector<RowType> row_types{};
  std::string objective_name{"OBJ"};
  std::vector<std::string> var_names{};
  std::vector<char> var_types{};
  std::vector<std::vector<i_t>> A_indices{};
  std::vector<std::vector<f_t>> A_values{};
  std::vector<f_t> b_values{};
  std::vector<f_t> c_values{};
  f_t objective_offset_value{0};
  std::vector<f_t> variable_upper_bounds{};
  std::vector<f_t> variable_lower_bounds{};
  bool maximize{false};
  // Quadratic objective entries (row, col, value) in upper-triangular
  // QUADOBJ convention; finalize_problem() mirrors to the full symmetric
  // matrix and applies the *0.5 factor required by cuOpt's x^T Q x form.
  std::vector<std::tuple<i_t, i_t, f_t>> quadobj_entries{};
};

}  // namespace cuopt::mps_parser
