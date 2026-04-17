/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser/mps_data_model.hpp>

#include <string>

namespace cuopt::mps_parser {

/**
 * @brief Reads a linear, mixed-integer, or quadratic optimization problem from
 *        a file in LP format.
 *
 * The LP format is a human-readable alternative to MPS format. Several
 * optimization solvers use slightly different dialects; this parser
 * implements the dialect documented at:
 *   https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html#lp-format
 *
 * Scope: LP, MIP, and QP problems are supported. SOS constraints, PWL
 * objectives, semi-continuous variables, general constraints, and user cuts
 * cause a ValidationError when encountered.
 *
 * @param[in] lp_file_path Path to the LP file.
 * @return mps_data_model_t A fully formed LP/MIP/QP problem representing the
 *         given file.
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_lp(const std::string& lp_file_path);

}  // namespace cuopt::mps_parser
