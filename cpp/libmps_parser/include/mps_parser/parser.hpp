/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser/lp_parser.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <string>
#include <string_view>

namespace cuopt::mps_parser {

/**
 * @brief Reads the equation from an MPS or QPS file.
 *
 * The input file can be a plain text file in MPS-/QPS-format or a compressed MPS/QPS
 * file (.mps.gz or .mps.bz2).
 *
 * Read this link http://lpsolve.sourceforge.net/5.5/mps-format.htm for more
 * details on both free and fixed MPS format.
 * This function supports both standard MPS files (for linear programming) and
 * QPS files (for quadratic programming). QPS files are MPS files with additional
 * sections:
 * - QUADOBJ: Defines quadratic terms in the objective function
 *
 * Note: Compressed MPS files .mps.gz, .mps.bz2 can only be read if the compression
 * libraries zlib or libbzip2 are installed, respectively.
 *
 * @param[in] mps_file_path Path to MPS/QPSfile.
 * @param[in] fixed_mps_format If MPS/QPS file should be parsed as fixed, false by default
 * @return mps_data_model_t A fully formed LP/QP problem which represents the given file
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_mps(const std::string& mps_file_path,
                                     bool fixed_mps_format = false);

/**
 * @brief Reads an optimization problem from a file, dispatching on the file
 *        extension.
 *
 * Case-insensitive `.lp` suffix routes to parse_lp(). Everything else —
 * including `.mps`, `.mps.gz`, `.mps.bz2`, and extensionless files — routes
 * to parse_mps() (with free-format parsing). This is the entry point of
 * choice for user-facing tools (CLI, C API) that want both formats to
 * "just work" without an explicit format flag.
 *
 * @param[in] path Path to the input file.
 * @return mps_data_model_t The parsed problem.
 */
template <typename i_t, typename f_t>
inline mps_data_model_t<i_t, f_t> parse_optimization_file(const std::string& path)
{
  constexpr std::string_view lp_suffix = ".lp";
  auto ends_with_ci                    = [](std::string_view s, std::string_view suffix) {
    if (s.size() < suffix.size()) return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
      char a = s[s.size() - suffix.size() + i];
      if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
      if (a != suffix[i]) return false;
    }
    return true;
  };
  if (ends_with_ci(path, lp_suffix)) return parse_lp<i_t, f_t>(path);
  return parse_mps<i_t, f_t>(path);
}

}  // namespace cuopt::mps_parser
