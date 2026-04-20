/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mps_parser/lp_parser.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/parser.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuopt::mps_parser {

namespace {

constexpr double tolerance = 1e-9;

// Writes `content` to a freshly-created temp file, parses it, removes the
// file, and returns the resulting data model. Using a temp file keeps the
// public parse_lp() API unchanged while letting tests stay self-contained.
mps_data_model_t<int, double> parse_lp_string(const std::string& content)
{
  std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                              (std::string{"cuopt_lp_test_"} + std::to_string(::getpid()) + "_" +
                               std::to_string(std::rand()) + ".lp");
  {
    std::ofstream out(tmp);
    out << content;
  }
  auto model = parse_lp<int, double>(tmp.string());
  std::filesystem::remove(tmp);
  return model;
}

// Returns the index of `name` in the variable list, or -1 if absent.
int find_var(const mps_data_model_t<int, double>& m, const std::string& name)
{
  const auto& names = m.get_variable_names();
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return static_cast<int>(i);
  }
  return -1;
}

int find_row(const mps_data_model_t<int, double>& m, const std::string& name)
{
  const auto& names = m.get_row_names();
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return static_cast<int>(i);
  }
  return -1;
}

// Returns A[row, col] by scanning the CSR row. Zero if the entry is missing.
double a_entry(const mps_data_model_t<int, double>& m, int row, int col)
{
  const auto& offsets = m.get_constraint_matrix_offsets();
  const auto& indices = m.get_constraint_matrix_indices();
  const auto& values  = m.get_constraint_matrix_values();
  for (int k = offsets[row]; k < offsets[row + 1]; ++k) {
    if (indices[k] == col) return values[k];
  }
  return 0.0;
}

// Returns Q[row, col] by scanning the CSR row of the quadratic matrix.
double q_entry(const mps_data_model_t<int, double>& m, int row, int col)
{
  const auto& offsets = m.get_quadratic_objective_offsets();
  const auto& indices = m.get_quadratic_objective_indices();
  const auto& values  = m.get_quadratic_objective_values();
  if (offsets.empty()) return 0.0;
  for (int k = offsets[row]; k < offsets[row + 1]; ++k) {
    if (indices[k] == col) return values[k];
  }
  return 0.0;
}

}  // namespace

// ---------- 01_trivial ---------------------------------------------------

TEST(lp_parser, trivial)
{
  auto m = parse_lp_string(R"LP(
Minimize
  x
Subject To
 lb_constr: x >= 2.5
Bounds
 x <= 10
End
)LP");

  EXPECT_FALSE(m.get_sense());  // minimize
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  int x = find_var(m, "x");
  ASSERT_GE(x, 0);
  EXPECT_EQ(m.get_variable_types()[x], 'C');
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], 0.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[x], 10.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[x], 1.0, tolerance);

  ASSERT_EQ(m.get_row_names().size(), 1u);
  int r = find_row(m, "lb_constr");
  ASSERT_GE(r, 0);
  // 'G' relation ⇒ finite lower bound, +inf upper bound.
  EXPECT_NEAR(m.get_constraint_lower_bounds()[r], 2.5, tolerance);
  EXPECT_TRUE(std::isinf(m.get_constraint_upper_bounds()[r]));
  EXPECT_NEAR(a_entry(m, r, x), 1.0, tolerance);
}

// ---------- 02_basic_lp --------------------------------------------------

TEST(lp_parser, basic_lp_with_float_coefficients)
{
  auto m = parse_lp_string(R"LP(
Minimize
  x1 + x2
Subject To
 c1: 2.5 x1 + x2 <= 10
 c2: x1 + 1.5 x2 <= 8
 c3: x1 + x2 <= 6
End
)LP");

  EXPECT_EQ(m.get_variable_names().size(), 2u);
  int x1 = find_var(m, "x1");
  int x2 = find_var(m, "x2");
  ASSERT_GE(x1, 0);
  ASSERT_GE(x2, 0);
  // Default bounds for continuous variables.
  EXPECT_NEAR(m.get_variable_lower_bounds()[x1], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x1]));

  ASSERT_EQ(m.get_row_names().size(), 3u);
  int c1 = find_row(m, "c1");
  int c2 = find_row(m, "c2");
  ASSERT_GE(c1, 0);
  ASSERT_GE(c2, 0);
  EXPECT_NEAR(a_entry(m, c1, x1), 2.5, tolerance);
  EXPECT_NEAR(a_entry(m, c1, x2), 1.0, tolerance);
  EXPECT_NEAR(a_entry(m, c2, x2), 1.5, tolerance);
  EXPECT_NEAR(m.get_constraint_upper_bounds()[c1], 10.0, tolerance);
}

// ---------- 03_maximize --------------------------------------------------

TEST(lp_parser, maximize_flips_sense)
{
  auto m = parse_lp_string(R"LP(
Maximize
  3 x + 2 y
Subject To
 c1: x + y <= 6
 c2: 2 x + y <= 8
End
)LP");

  EXPECT_TRUE(m.get_sense());
  int x = find_var(m, "x");
  int y = find_var(m, "y");
  EXPECT_NEAR(m.get_objective_coefficients()[x], 3.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 2.0, tolerance);
}

// ---------- 04_equality_constraints --------------------------------------

TEST(lp_parser, equality_constraints)
{
  auto m = parse_lp_string(R"LP(
Minimize
  c1 + 2 c2 + 3 c3 + 4 c4
Subject To
 s1: c1 + c2 = 10
 s2: c3 + c4 = 12
 d1: c1 + c3 = 9
 d2: c2 + c4 = 13
End
)LP");

  ASSERT_EQ(m.get_row_names().size(), 4u);
  // All four are equality constraints ⇒ lb == ub for every row.
  const auto& clb = m.get_constraint_lower_bounds();
  const auto& cub = m.get_constraint_upper_bounds();
  for (size_t i = 0; i < clb.size(); ++i) {
    EXPECT_NEAR(clb[i], cub[i], tolerance);
  }
  int s1 = find_row(m, "s1");
  EXPECT_NEAR(m.get_constraint_lower_bounds()[s1], 10.0, tolerance);
  EXPECT_NEAR(m.get_constraint_upper_bounds()[s1], 10.0, tolerance);
}

// ---------- 05_mixed_constraints -----------------------------------------

TEST(lp_parser, mixed_constraint_relations)
{
  auto m = parse_lp_string(R"LP(
Minimize
  x + 2 y + 3 z
Subject To
 eq1: x + y + z = 10
 geq1: x + 2 y >= 6
 leq1: y + z <= 8
End
)LP");

  int eq  = find_row(m, "eq1");
  int geq = find_row(m, "geq1");
  int leq = find_row(m, "leq1");
  // Relation is recovered from the constraint lower/upper bounds:
  //   'E' ⇒ lb == ub
  //   'G' ⇒ ub = +inf
  //   'L' ⇒ lb = -inf
  EXPECT_NEAR(m.get_constraint_lower_bounds()[eq], m.get_constraint_upper_bounds()[eq], tolerance);
  EXPECT_NEAR(m.get_constraint_lower_bounds()[geq], 6.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_constraint_upper_bounds()[geq]));
  EXPECT_NEAR(m.get_constraint_upper_bounds()[leq], 8.0, tolerance);
  EXPECT_TRUE(std::isinf(-m.get_constraint_lower_bounds()[leq]));
}

// ---------- 06_free_variables --------------------------------------------

TEST(lp_parser, free_and_negative_lower_bound_variables)
{
  auto m = parse_lp_string(R"LP(
Minimize
  xfree + xneg_lb + xstd
Subject To
 sum_lb: xfree + xneg_lb + xstd >= 1
 diff_ub: xfree - xneg_lb <= 3
 xst_cap: xstd <= 5
Bounds
 xfree free
 -3 <= xneg_lb <= 10
End
)LP");

  int xf = find_var(m, "xfree");
  int xn = find_var(m, "xneg_lb");
  int xs = find_var(m, "xstd");
  EXPECT_TRUE(std::isinf(-m.get_variable_lower_bounds()[xf]));
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xf]));
  EXPECT_NEAR(m.get_variable_lower_bounds()[xn], -3.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xn], 10.0, tolerance);
  EXPECT_NEAR(m.get_variable_lower_bounds()[xs], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xs]));

  // - xneg_lb → coefficient -1 in the diff_ub row
  int dr = find_row(m, "diff_ub");
  EXPECT_NEAR(a_entry(m, dr, xn), -1.0, tolerance);
}

// ---------- 07_bounds_variety --------------------------------------------

TEST(lp_parser, bounds_variety)
{
  auto m = parse_lp_string(R"LP(
Minimize
  xfixed + xub_only + xlb_pos
Subject To
 c1: xfixed + xub_only + xlb_pos >= 1
Bounds
 xfixed = 3
 xub_only <= 7.5
 xlb_pos >= 2
End
)LP");

  int xfixed = find_var(m, "xfixed");
  int xub    = find_var(m, "xub_only");
  int xlb    = find_var(m, "xlb_pos");
  EXPECT_NEAR(m.get_variable_lower_bounds()[xfixed], 3.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xfixed], 3.0, tolerance);
  EXPECT_NEAR(m.get_variable_lower_bounds()[xub], 0.0, tolerance);
  EXPECT_NEAR(m.get_variable_upper_bounds()[xub], 7.5, tolerance);
  EXPECT_NEAR(m.get_variable_lower_bounds()[xlb], 2.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xlb]));
}

// ---------- 08_integer_mip -----------------------------------------------

TEST(lp_parser, general_integers)
{
  auto m = parse_lp_string(R"LP(
Maximize
  3 x + 5 y
Subject To
 c1: x + 2 y <= 12
 c2: 2 x + y <= 10
Generals
 x y
End
)LP");

  int x = find_var(m, "x");
  int y = find_var(m, "y");
  EXPECT_EQ(m.get_variable_types()[x], 'I');
  EXPECT_EQ(m.get_variable_types()[y], 'I');
  // Generals alone does NOT force [0,1]; default bounds remain [0, +inf).
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x]));
}

// ---------- 09_binary_mip ------------------------------------------------

TEST(lp_parser, binaries_set_zero_one_bounds)
{
  auto m = parse_lp_string(R"LP(
Maximize
  3 x1 + 5 x2 + 4 x3 + 2 x4
Subject To
 knapsack: 2 x1 + 3 x2 + x3 + x4 <= 5
Binaries
 x1 x2 x3 x4
End
)LP");

  for (const std::string& n : {"x1", "x2", "x3", "x4"}) {
    int v = find_var(m, n);
    EXPECT_EQ(m.get_variable_types()[v], 'I');
    EXPECT_NEAR(m.get_variable_lower_bounds()[v], 0.0, tolerance);
    EXPECT_NEAR(m.get_variable_upper_bounds()[v], 1.0, tolerance);
  }
}

// ---------- 10_mixed_mip -------------------------------------------------

TEST(lp_parser, mixed_continuous_integer_binary)
{
  auto m = parse_lp_string(R"LP(
Maximize
  3 xc + 4 xi + 7 xb
Subject To
 c1: xc + xi + xb <= 10
Generals
 xi
Binaries
 xb
End
)LP");

  int xc = find_var(m, "xc");
  int xi = find_var(m, "xi");
  int xb = find_var(m, "xb");
  EXPECT_EQ(m.get_variable_types()[xc], 'C');
  EXPECT_EQ(m.get_variable_types()[xi], 'I');
  EXPECT_EQ(m.get_variable_types()[xb], 'I');
  EXPECT_NEAR(m.get_variable_upper_bounds()[xb], 1.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[xi]));
}

// ---------- 11_qp_diagonal -----------------------------------------------

TEST(lp_parser, quadratic_diagonal_only)
{
  auto m = parse_lp_string(R"LP(
Minimize
  x + y + [ 2 x ^2 + 4 y ^2 ] / 2
Subject To
 c1: x + y >= 1
Bounds
 x free
 y free
End
)LP");

  ASSERT_TRUE(m.has_quadratic_objective());
  int x = find_var(m, "x");
  int y = find_var(m, "y");
  // LP [2 x^2]/2 = x^2  ⇒  Q[x,x] should be 1 in cuOpt's x^T Q x form.
  EXPECT_NEAR(q_entry(m, x, x), 1.0, tolerance);
  // LP [4 y^2]/2 = 2 y^2  ⇒  Q[y,y] = 2.
  EXPECT_NEAR(q_entry(m, y, y), 2.0, tolerance);
  EXPECT_NEAR(q_entry(m, x, y), 0.0, tolerance);
  // Linear part is preserved.
  EXPECT_NEAR(m.get_objective_coefficients()[x], 1.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 1.0, tolerance);
}

// ---------- 12_qp_cross_terms --------------------------------------------

TEST(lp_parser, quadratic_with_cross_terms)
{
  auto m = parse_lp_string(R"LP(
Minimize
  - 3 x - 4 y - 2 z + [ 2 x ^2 + 2 x * y + 2 y ^2 + 2 y * z + 2 z ^2 ] / 2
Subject To
 c1: x + y + z <= 10
 c2: x + y >= 1
End
)LP");

  ASSERT_TRUE(m.has_quadratic_objective());
  int x = find_var(m, "x");
  int y = find_var(m, "y");
  int z = find_var(m, "z");
  // Diagonal 2 x^2 / 2 = x^2  ⇒  Q[x,x] = 1, similarly for y, z.
  EXPECT_NEAR(q_entry(m, x, x), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, y, y), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, z, z), 1.0, tolerance);
  // Cross 2 x*y / 2 = x*y  ⇒  full matrix Q[x,y] = Q[y,x] = 0.5 each
  // (so that x^T Q x sums to x*y).
  EXPECT_NEAR(q_entry(m, x, y), 0.5, tolerance);
  EXPECT_NEAR(q_entry(m, y, x), 0.5, tolerance);
  EXPECT_NEAR(q_entry(m, y, z), 0.5, tolerance);
  EXPECT_NEAR(q_entry(m, z, y), 0.5, tolerance);
  // x and z have no cross term.
  EXPECT_NEAR(q_entry(m, x, z), 0.0, tolerance);

  EXPECT_NEAR(m.get_objective_coefficients()[x], -3.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], -4.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[z], -2.0, tolerance);
}

// ---------- 13_miqp ------------------------------------------------------

TEST(lp_parser, miqp_integer_with_quadratic_objective)
{
  auto m = parse_lp_string(R"LP(
Minimize
  - 4 xi - 2 xc + [ 2 xi ^2 + 2 xc ^2 ] / 2
Subject To
 c1: xi + xc <= 5
Bounds
 xi <= 4
Generals
 xi
End
)LP");

  int xi = find_var(m, "xi");
  int xc = find_var(m, "xc");
  EXPECT_EQ(m.get_variable_types()[xi], 'I');
  EXPECT_EQ(m.get_variable_types()[xc], 'C');
  EXPECT_NEAR(m.get_variable_upper_bounds()[xi], 4.0, tolerance);
  EXPECT_NEAR(q_entry(m, xi, xi), 1.0, tolerance);
  EXPECT_NEAR(q_entry(m, xc, xc), 1.0, tolerance);
}

// ---------- 14_infeasible: parsing success regardless of feasibility -----

TEST(lp_parser, infeasible_model_parses_faithfully)
{
  auto m = parse_lp_string(R"LP(
Minimize
  x + y
Subject To
 high: x + y >= 15
 low: x + y <= 8
Bounds
 x <= 5
 y <= 5
End
)LP");

  EXPECT_EQ(m.get_row_names().size(), 2u);
  int high = find_row(m, "high");
  int low  = find_row(m, "low");
  EXPECT_NEAR(m.get_constraint_lower_bounds()[high], 15.0, tolerance);
  EXPECT_NEAR(m.get_constraint_upper_bounds()[low], 8.0, tolerance);
}

// ---------- 15_unbounded: parser is agnostic to boundedness --------------

TEST(lp_parser, unbounded_model_parses)
{
  auto m = parse_lp_string(R"LP(
Maximize
  x + y
Subject To
 c1: x - y <= 5
End
)LP");

  int x = find_var(m, "x");
  EXPECT_NEAR(m.get_variable_lower_bounds()[x], 0.0, tolerance);
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x]));
  EXPECT_TRUE(m.get_sense());
}

// ---------- Error-path tests ---------------------------------------------

TEST(lp_parser, missing_objective_throws)
{
  EXPECT_THROW(parse_lp_string(R"LP(
Subject To
 c1: x + y <= 5
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unsupported_sos_section_throws)
{
  EXPECT_THROW(parse_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
SOS
 s1: S1 :: x : 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unsupported_semi_continuous_section_throws)
{
  EXPECT_THROW(parse_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
Semi-continuous
 x
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unsupported_pwlobj_section_throws)
{
  EXPECT_THROW(parse_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
PWLObj
 x: 0 0 1 1
End
)LP"),
               std::logic_error);
}

TEST(lp_parser, unknown_file_throws)
{
  auto call = [] { return parse_lp<int, double>("/definitely/does/not/exist.lp"); };
  EXPECT_THROW(call(), std::logic_error);
}

// ---------- Feature smoke tests ------------------------------------------

TEST(lp_parser, case_insensitive_section_keywords)
{
  auto m = parse_lp_string(R"LP(
MINIMIZE
  x
SUBJECT TO
 c1: x >= 1
BOUNDS
 x <= 5
END
)LP");
  int x  = find_var(m, "x");
  EXPECT_NEAR(m.get_variable_upper_bounds()[x], 5.0, tolerance);
}

TEST(lp_parser, backslash_comments_are_ignored)
{
  auto m = parse_lp_string(R"LP(
\ This is a comment
Minimize
  x \ trailing comment
Subject To \ another comment
 c1: x >= 1
End
)LP");
  int x  = find_var(m, "x");
  EXPECT_NEAR(m.get_objective_coefficients()[x], 1.0, tolerance);
}

TEST(lp_parser, missing_end_warns_but_succeeds)
{
  // No End — should still parse. (A warning is printed; see parse_all().)
  auto m = parse_lp_string(R"LP(
Minimize
  x
Subject To
 c1: x >= 1
)LP");
  EXPECT_EQ(m.get_variable_names().size(), 1u);
}

TEST(lp_parser, auto_generates_names_for_unlabeled_constraints)
{
  auto m = parse_lp_string(R"LP(
Minimize
  x + y
Subject To
 x + y <= 10
 x - y >= 0
End
)LP");
  ASSERT_EQ(m.get_row_names().size(), 2u);
  // Default auto-generated names are R0, R1.
  EXPECT_EQ(m.get_row_names()[0], "R0");
  EXPECT_EQ(m.get_row_names()[1], "R1");
}

TEST(lp_parser, infinity_keyword_in_bounds)
{
  auto m = parse_lp_string(R"LP(
Minimize
  x + y
Subject To
 c1: x + y >= 0
Bounds
 -inf <= x <= inf
 -infinity <= y
End
)LP");
  int x  = find_var(m, "x");
  int y  = find_var(m, "y");
  EXPECT_TRUE(std::isinf(-m.get_variable_lower_bounds()[x]));
  EXPECT_TRUE(std::isinf(m.get_variable_upper_bounds()[x]));
  EXPECT_TRUE(std::isinf(-m.get_variable_lower_bounds()[y]));
}

TEST(lp_parser, coefficient_one_implicit_with_leading_minus)
{
  auto m = parse_lp_string(R"LP(
Minimize
  - x + y
Subject To
 c1: - x + y <= 0
End
)LP");
  int x  = find_var(m, "x");
  int y  = find_var(m, "y");
  EXPECT_NEAR(m.get_objective_coefficients()[x], -1.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 1.0, tolerance);
  int r = find_row(m, "c1");
  EXPECT_NEAR(a_entry(m, r, x), -1.0, tolerance);
}

TEST(lp_parser, quadratic_without_slash_two_divides_coefficients_in_place)
{
  // Without "/ 2" the raw coefficient IS the actual coefficient.
  // [ 1 x^2 ]  ⇒  Q[x,x] = 1.
  auto m = parse_lp_string(R"LP(
Minimize
  [ 1 x ^2 ]
Subject To
 c1: x >= 1
End
)LP");
  ASSERT_TRUE(m.has_quadratic_objective());
  int x = find_var(m, "x");
  EXPECT_NEAR(q_entry(m, x, x), 1.0, tolerance);
}

TEST(lp_parser, duplicate_coefficient_accumulates)
{
  // Repeated variable in the objective should sum coefficients.
  auto m = parse_lp_string(R"LP(
Minimize
  2 x + 3 x + y
Subject To
 c1: x + y >= 1
End
)LP");
  int x  = find_var(m, "x");
  int y  = find_var(m, "y");
  EXPECT_NEAR(m.get_objective_coefficients()[x], 5.0, tolerance);
  EXPECT_NEAR(m.get_objective_coefficients()[y], 1.0, tolerance);
}

// ===========================================================================
// parse_optimization_file dispatch tests
//
// Verifies the extension-based dispatch used by cuopt_cli and the C API.
// ===========================================================================

namespace {

// Writes `content` to a temp file with the given suffix, parses it via
// parse_optimization_file, removes the file, and returns the resulting model.
mps_data_model_t<int, double> dispatch_parse(const std::string& content, const std::string& suffix)
{
  std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                              (std::string{"cuopt_dispatch_test_"} + std::to_string(::getpid()) +
                               "_" + std::to_string(std::rand()) + suffix);
  {
    std::ofstream out(tmp);
    out << content;
  }
  auto model = parse_optimization_file<int, double>(tmp.string());
  std::filesystem::remove(tmp);
  return model;
}

constexpr const char* kTrivialLp = R"LP(
Minimize
  x
Subject To
 c1: x >= 2.5
Bounds
 x <= 10
End
)LP";

constexpr const char* kTrivialMps = R"MPS(NAME trivial
ROWS
 N OBJ
 G c1
COLUMNS
 x OBJ 1
 x c1 1
RHS
 RHS1 c1 2.5
BOUNDS
 UP BND1 x 10
ENDATA
)MPS";

}  // namespace

TEST(parse_optimization_file, lowercase_lp_suffix_dispatches_to_lp_parser)
{
  auto m = dispatch_parse(kTrivialLp, ".lp");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
  EXPECT_NEAR(m.get_variable_upper_bounds()[0], 10.0, tolerance);
}

TEST(parse_optimization_file, uppercase_lp_suffix_dispatches_to_lp_parser)
{
  auto m = dispatch_parse(kTrivialLp, ".LP");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
}

TEST(parse_optimization_file, mps_suffix_dispatches_to_mps_parser)
{
  auto m = dispatch_parse(kTrivialMps, ".mps");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
  EXPECT_NEAR(m.get_variable_upper_bounds()[0], 10.0, tolerance);
}

TEST(parse_optimization_file, no_extension_dispatches_to_mps_parser)
{
  // Extensionless file should go to MPS (current behavior preserved).
  auto m = dispatch_parse(kTrivialMps, "");
  ASSERT_EQ(m.get_variable_names().size(), 1u);
  EXPECT_EQ(m.get_variable_names()[0], "x");
}

TEST(parse_optimization_file, non_lp_suffix_dispatches_to_mps_parser)
{
  // ".lp.gz" is NOT ".lp" (suffix is ".gz"), so it routes to MPS. The MPS
  // parser will then attempt gzip decompression — we just confirm dispatch
  // picked the MPS path by observing a specific MPS-style error message.
  std::filesystem::path tmp =
    std::filesystem::temp_directory_path() /
    (std::string{"cuopt_dispatch_gz_"} + std::to_string(::getpid()) + ".lp.gz");
  {
    std::ofstream(tmp) << "not a real gzipped file";
  }
  try {
    parse_optimization_file<int, double>(tmp.string());
    FAIL() << "Expected parse failure for bogus .lp.gz content";
  } catch (const std::logic_error& e) {
    // MPS parser complains about the .gz file; the key point is we did NOT
    // call parse_lp (which would give a different error message without
    // 'gz' / 'MPS' in it).
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("LP file") == std::string::npos)
      << "parse_optimization_file appears to have dispatched to parse_lp "
         "for a .lp.gz path; error was: "
      << msg;
  }
  std::filesystem::remove(tmp);
}

}  // namespace cuopt::mps_parser
