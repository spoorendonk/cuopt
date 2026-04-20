/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/common_utils.hpp>

#include <mps_parser.hpp>
#include <mps_parser/lp_parser.hpp>
#include <mps_parser/mps_writer.hpp>
#include <mps_parser/parser.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::mps_parser {

constexpr double tolerance = 1e-6;

mps_parser_t<int, double> read_from_mps(const std::string& file, bool fixed_format = true)
{
  std::string rel_file{};
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  // Empty problem not used in the test
  mps_data_model_t<int, double> problem;
  mps_parser_t<int, double> mps{problem, rel_file, fixed_format};
  return mps;
}

bool file_exists(const std::string& file)
{
  std::string rel_file{};
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  return std::filesystem::exists(rel_file);
}

namespace {

// Writes `content` to a freshly-created temp file, parses it, removes the
// file, and returns the resulting data model. Used by LP tests that encode
// their fixture inline rather than shipping a file under datasets/.
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

// ===========================================================================
// Per-fixture test classes. Each class describes one named problem fixture
// and owns the checker for that problem's expected parsed data model. The
// MPS and LP TEST_F cases within a fixture share the same `check_model`
// method, so the expected values live in exactly one place per fixture.
//
// All fixtures inherit a common base that supplies parse_mps_file and
// parse_lp_file helpers.
// ===========================================================================

class parser_fixture_base : public ::testing::Test {
 protected:
  static mps_data_model_t<int, double> parse_mps_file(const std::string& file,
                                                      bool fixed_format = true)
  {
    const std::string& root = cuopt::test::get_rapids_dataset_root_dir();
    return parse_mps<int, double>(root + "/" + file, fixed_format);
  }

  static mps_data_model_t<int, double> parse_lp_file(const std::string& file)
  {
    const std::string& root = cuopt::test::get_rapids_dataset_root_dir();
    return parse_lp<int, double>(root + "/" + file);
  }
};

// 2 vars (continuous, default [0,inf) bounds), 2 <= constraints.
//   min 0.2*VAR1 + 0.1*VAR2
//   ROW1: 3*VAR1 + 4*VAR2 <= 5.4
//   ROW2: 2.7*VAR1 + 10.1*VAR2 <= 4.9
class good_mps_1_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_FALSE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    ASSERT_EQ(2, m.get_n_constraints());
    EXPECT_EQ("VAR1", m.get_variable_names()[0]);
    EXPECT_EQ("VAR2", m.get_variable_names()[1]);
    EXPECT_EQ("ROW1", m.get_row_names()[0]);
    EXPECT_EQ("ROW2", m.get_row_names()[1]);
    EXPECT_EQ('C', m.get_variable_types()[0]);
    EXPECT_EQ('C', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
    EXPECT_NEAR(0.2, m.get_objective_coefficients()[0], tolerance);
    EXPECT_NEAR(0.1, m.get_objective_coefficients()[1], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[0]);
    EXPECT_NEAR(5.4, m.get_constraint_upper_bounds()[0], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[1]);
    EXPECT_NEAR(4.9, m.get_constraint_upper_bounds()[1], tolerance);
    const auto& off = m.get_constraint_matrix_offsets();
    const auto& idx = m.get_constraint_matrix_indices();
    const auto& val = m.get_constraint_matrix_values();
    ASSERT_EQ(3u, off.size());
    EXPECT_EQ(0, off[0]);
    EXPECT_EQ(2, off[1]);
    EXPECT_EQ(4, off[2]);
    EXPECT_EQ(0, idx[0]);
    EXPECT_NEAR(3.0, val[0], tolerance);
    EXPECT_EQ(1, idx[1]);
    EXPECT_NEAR(4.0, val[1], tolerance);
    EXPECT_EQ(0, idx[2]);
    EXPECT_NEAR(2.7, val[2], tolerance);
    EXPECT_EQ(1, idx[3]);
    EXPECT_NEAR(10.1, val[3], tolerance);
    EXPECT_FALSE(m.has_quadratic_objective());
  }
};

// min 2x - y; x+y <= 3; 0<=x<=1, 1<=y<=2.
class up_low_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_FALSE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    ASSERT_EQ(1, m.get_n_constraints());
    EXPECT_EQ("x", m.get_variable_names()[0]);
    EXPECT_EQ("y", m.get_variable_names()[1]);
    EXPECT_EQ("con", m.get_row_names()[0]);
    EXPECT_NEAR(2.0, m.get_objective_coefficients()[0], tolerance);
    EXPECT_NEAR(-1.0, m.get_objective_coefficients()[1], tolerance);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(2.0, m.get_variable_upper_bounds()[1]);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[0]);
    EXPECT_NEAR(3.0, m.get_constraint_upper_bounds()[0], tolerance);
    const auto& val = m.get_constraint_matrix_values();
    ASSERT_EQ(2u, val.size());
    EXPECT_NEAR(1.0, val[0], tolerance);
    EXPECT_NEAR(1.0, val[1], tolerance);
  }
};

// good-mps-1 objective/matrix/rows; -1 <= VAR1 <= inf, 0 <= VAR2 <= 2.
class some_var_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(-1.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(2.0, m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 fixed at 2; VAR2 default [0, inf).
class fixed_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(2.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(2.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 free (-inf, +inf); VAR2 default [0, +inf).
class free_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 lower=-inf (MI in MPS / -inf in LP), upper default +inf; VAR2 default.
// Effective bounds match free_var_bound_test — the two fixtures differ only in
// how the lower -inf is spelled (free vs explicit -inf bound).
class lower_inf_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 upper=+inf (PL in MPS / inf in LP); both default lower 0. Effective
// bounds match two default [0, +inf) variables.
class upper_inf_var_bound_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// 2 integer vars bounded [0, 10]; max 100 VAR1 + 150 VAR2;
// 8000 VAR1 + 4000 VAR2 <= 40000 ; 15 VAR1 + 30 VAR2 <= 200.
class mip_with_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_TRUE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    ASSERT_EQ(2, m.get_n_constraints());
    EXPECT_EQ("VAR1", m.get_variable_names()[0]);
    EXPECT_EQ("VAR2", m.get_variable_names()[1]);
    EXPECT_EQ('I', m.get_variable_types()[0]);
    EXPECT_EQ('I', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(10.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(10.0, m.get_variable_upper_bounds()[1]);
    EXPECT_NEAR(100.0, m.get_objective_coefficients()[0], tolerance);
    EXPECT_NEAR(150.0, m.get_objective_coefficients()[1], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[0]);
    EXPECT_NEAR(40000.0, m.get_constraint_upper_bounds()[0], tolerance);
    EXPECT_EQ(-std::numeric_limits<double>::infinity(), m.get_constraint_lower_bounds()[1]);
    EXPECT_NEAR(200.0, m.get_constraint_upper_bounds()[1], tolerance);
    const auto& val = m.get_constraint_matrix_values();
    ASSERT_EQ(4u, val.size());
    EXPECT_NEAR(8000.0, val[0], tolerance);
    EXPECT_NEAR(4000.0, val[1], tolerance);
    EXPECT_NEAR(15.0, val[2], tolerance);
    EXPECT_NEAR(30.0, val[3], tolerance);
  }
};

// Like mip_with_bounds but VAR1 is binary ([0,1]) and VAR2 is continuous,
// default upper +inf. (MPS: no explicit bounds on integer => [0,1]. LP: VAR1
// listed under Binaries.)
class mip_no_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_TRUE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ('I', m.get_variable_types()[0]);
    EXPECT_EQ('C', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), m.get_variable_upper_bounds()[1]);
  }
};

// VAR1 binary ([0,1]); VAR2 continuous with explicit upper 10.
class mip_partial_bounds_test : public parser_fixture_base {
 protected:
  static void check_model(const mps_data_model_t<int, double>& m)
  {
    EXPECT_TRUE(m.get_sense());
    ASSERT_EQ(2, m.get_n_variables());
    EXPECT_EQ('I', m.get_variable_types()[0]);
    EXPECT_EQ('C', m.get_variable_types()[1]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[0]);
    EXPECT_EQ(1.0, m.get_variable_upper_bounds()[0]);
    EXPECT_EQ(0.0, m.get_variable_lower_bounds()[1]);
    EXPECT_EQ(10.0, m.get_variable_upper_bounds()[1]);
  }
};

TEST(mps_parser, bad_mps_files)
{
  std::stringstream ss;
  static constexpr int NumMpsFiles = 15;
  for (int i = 1; i <= NumMpsFiles; ++i) {
    ss << "linear_programming/bad-mps-" << i << ".mps";
    // Check if file exists
    if (file_exists(ss.str())) ASSERT_THROW(read_from_mps(ss.str()), std::logic_error);
    ss.str(std::string{});
    ss.clear();
  }
}

TEST_F(good_mps_1_test, mps)
{
  check_model(parse_mps_file("linear_programming/good-mps-1.mps"));
  // Parser-struct fields that are MPS-only (not exposed via the data model).
  auto mps = read_from_mps("linear_programming/good-mps-1.mps");
  EXPECT_EQ("good-1", mps.problem_name);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
}

TEST_F(good_mps_1_test, lp) { check_model(parse_lp_file("linear_programming/good-mps-1.lp")); }

TEST(mps_parser, good_mps_file_clrf)
{
  auto mps = read_from_mps("linear_programming/good-mps-1-clrf.mps");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser, good_mps_free_file_clrf)
{
  auto mps = read_from_mps("linear_programming/good-mps-1-clrf.mps", false);
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser, good_mps_file_comments)
{
  auto mps = read_from_mps("linear_programming/good-mps-1-comments.mps", false);
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(1), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(1), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser, good_mps_file_no_name)
{
  // Should not throw an error
  read_from_mps("linear_programming/good-mps-fixed-no-name.mps");
}

TEST(mps_parser, good_mps_file_empty_name)
{
  // Should not throw an error
  read_from_mps("linear_programming/good-mps-fixed-empty-name.mps");
}

TEST(mps_parser, good_mps_file_2)
{
  auto mps = read_from_mps("linear_programming/good-fixed-mps-2.mps");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("RO W1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VA R1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}

TEST(mps_parser_free_format, free_format_mps_file_1)
{  // tests for arbitrary spacing in rows, column, rhs
  auto mps = read_from_mps("linear_programming/free-format-mps-1.mps", false);
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
  EXPECT_EQ(false, mps.maximize);
}

TEST(mps_parser_free_format, bad_free_format_mps_with_spaces_in_names)
{
  ASSERT_THROW(read_from_mps("linear_programming/good-fixed-mps-2.mps", false), std::logic_error);
}

TEST(mps_parser_free_format, bad_mps_files_free_format)
{
  std::stringstream ss;
  static constexpr int NumMpsFiles = 13;
  for (int i = 1; i <= NumMpsFiles; ++i) {
    ss << "linear_programming/bad-mps-" << i << ".mps";
    if (file_exists(ss.str())) ASSERT_THROW(read_from_mps(ss.str(), false), std::logic_error);
    ss.str(std::string{});
    ss.clear();
  }
}

TEST_F(up_low_bounds_test, mps)
{
  check_model(parse_mps_file("linear_programming/lp_model_with_var_bounds.mps", false));
  auto mps = read_from_mps("linear_programming/lp_model_with_var_bounds.mps", false);
  EXPECT_EQ("lp_model_with_var_bounds", mps.problem_name);
  EXPECT_EQ("OBJ", mps.objective_name);
  ASSERT_EQ(int(1), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
}

TEST_F(up_low_bounds_test, lp)
{
  check_model(parse_lp_file("linear_programming/lp_model_with_var_bounds.lp"));
}

TEST_F(good_mps_1_test, mps_free_format)
{
  // free-format-mps-1.mps encodes the same problem as good-mps-1 with default
  // [0, +inf) bounds (no BOUNDS section), so it satisfies the same checker.
  check_model(parse_mps_file("linear_programming/free-format-mps-1.mps", false));
}

TEST_F(some_var_bounds_test, mps)
{
  check_model(parse_mps_file("linear_programming/good-mps-some-var-bounds.mps"));
}

TEST_F(some_var_bounds_test, lp)
{
  check_model(parse_lp_file("linear_programming/good-mps-some-var-bounds.lp"));
}

TEST_F(fixed_var_bound_test, mps)
{
  check_model(parse_mps_file("linear_programming/good-mps-fixed-var.mps"));
}

TEST_F(fixed_var_bound_test, lp)
{
  check_model(parse_lp_file("linear_programming/good-mps-fixed-var.lp"));
}

TEST_F(free_var_bound_test, mps)
{
  check_model(parse_mps_file("linear_programming/good-mps-free-var.mps"));
}

TEST_F(free_var_bound_test, lp)
{
  check_model(parse_lp_file("linear_programming/good-mps-free-var.lp"));
}

TEST_F(lower_inf_var_bound_test, mps)
{
  check_model(parse_mps_file("linear_programming/good-mps-lower-bound-inf-var.mps"));
}

TEST_F(lower_inf_var_bound_test, lp)
{
  check_model(parse_lp_file("linear_programming/good-mps-lower-bound-inf-var.lp"));
}

TEST(mps_bounds, rhs_cost)
{
  auto mps = read_from_mps("linear_programming/good-mps-rhs-cost.mps");

  // objective value offset should be set to -5
  EXPECT_EQ(int(-5), mps.objective_offset_value);
}

TEST_F(upper_inf_var_bound_test, mps)
{
  check_model(parse_mps_file("linear_programming/good-mps-upper-bound-inf-var.mps"));
}

TEST_F(upper_inf_var_bound_test, lp)
{
  check_model(parse_lp_file("linear_programming/good-mps-upper-bound-inf-var.lp"));
}

TEST(mps_ranges, fixed_ranges)
{
  std::string file = "linear_programming/good-mps-fixed-ranges.mps";
  auto mps         = read_from_mps(file);

  EXPECT_NEAR(4.2, mps.ranges_values[0], tolerance);   //  ROW1 range value
  EXPECT_NEAR(3.4, mps.ranges_values[1], tolerance);   //  ROW2 range value
  EXPECT_NEAR(-1.6, mps.ranges_values[2], tolerance);  // ROW3 range value
  EXPECT_NEAR(3.4, mps.ranges_values[3], tolerance);   //  ROW3 range value

  std::string rel_file{};
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  auto data_model                         = parse_mps<int, double>(rel_file, true);

  EXPECT_NEAR(1.2, data_model.get_constraint_lower_bounds()[0], tolerance);  // ROW1 lower bound
  EXPECT_NEAR(5.4, data_model.get_constraint_upper_bounds()[0], tolerance);  // ROW1 upper bound
  EXPECT_NEAR(1.5, data_model.get_constraint_lower_bounds()[1], tolerance);  // ROW2 lower bound
  EXPECT_NEAR(4.9, data_model.get_constraint_upper_bounds()[1], tolerance);  // ROW2 upper bound
  EXPECT_NEAR(
    7.9, data_model.get_constraint_lower_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    9.5, data_model.get_constraint_upper_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    3.5, data_model.get_constraint_lower_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(
    6.9, data_model.get_constraint_upper_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_lower_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_upper_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_lower_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_upper_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
}

TEST(mps_ranges, free_ranges)
{
  std::string file = "linear_programming/good-mps-free-ranges.mps";
  auto mps         = read_from_mps(file, false);

  EXPECT_NEAR(4.2, mps.ranges_values[0], tolerance);   //  ROW1 range value
  EXPECT_NEAR(3.4, mps.ranges_values[1], tolerance);   //  ROW2 range value
  EXPECT_NEAR(-1.6, mps.ranges_values[2], tolerance);  // ROW3 range value
  EXPECT_NEAR(3.4, mps.ranges_values[3], tolerance);   //  ROW3 range value

  std::string rel_file{};
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  auto data_model                         = parse_mps<int, double>(rel_file, false);

  EXPECT_NEAR(1.2, data_model.get_constraint_lower_bounds()[0], tolerance);  // ROW1 lower bound
  EXPECT_NEAR(5.4, data_model.get_constraint_upper_bounds()[0], tolerance);  // ROW1 upper bound
  EXPECT_NEAR(1.5, data_model.get_constraint_lower_bounds()[1], tolerance);  // ROW2 lower bound
  EXPECT_NEAR(4.9, data_model.get_constraint_upper_bounds()[1], tolerance);  // ROW2 upper bound
  EXPECT_NEAR(
    7.9, data_model.get_constraint_lower_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    9.5, data_model.get_constraint_upper_bounds()[2], tolerance);  // ROW3, equal constraint
  EXPECT_NEAR(
    3.5, data_model.get_constraint_lower_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(
    6.9, data_model.get_constraint_upper_bounds()[3], tolerance);  // ROW4, equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_lower_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(3.9,
              data_model.get_constraint_upper_bounds()[4],
              tolerance);  // ROW5, lower turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_lower_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
  EXPECT_NEAR(4.9,
              data_model.get_constraint_upper_bounds()[5],
              tolerance);  // ROW6, greater turned into equal constraint
}

TEST(mps_name, two_objectives)
{
  std::string file = "linear_programming/good-mps-fixed-two-objectives.mps";
  auto mps         = read_from_mps(file, false);

  // Objective name should be first one found and not trigger an error
  EXPECT_EQ(mps.objective_name, "COST");
}

TEST(mps_objname, two_objectives)
{
  std::string file = "linear_programming/good-mps-fixed-two-objectives-objname.mps";
  auto mps         = read_from_mps(file, false);

  // Objective name is the second one found since it's specified as objname
  EXPECT_EQ(mps.objective_name, "COST6679327");
}

TEST(mps_objname, two_objectives_next_line)
{
  std::string file = "linear_programming/good-mps-fixed-two-objectives-objname-next-line.mps";
  auto mps         = read_from_mps(file, false);

  // Objective name is the second one found since it's specified as objname
  EXPECT_EQ(mps.objective_name, "COST6679327");
}

TEST(mps_objname, bad_after)
{
  std::string file = "linear_programming/bad-mps-fixed-objname-after-rows.mps";
  ASSERT_THROW(read_from_mps(file, false), std::logic_error);
}

TEST(mps_objname, bad_no_fixed)
{
  std::string file = "linear_programming/bad-mps-fixed-objname-after-rows.mps";
  ASSERT_THROW(read_from_mps(file, true), std::logic_error);
}

TEST(mps_ranges, bad_name)
{
  ASSERT_THROW(read_from_mps("linear_programming/bad-mps-fixed-ranges-name.mps", false),
               std::logic_error);
}

TEST(mps_ranges, bad_value)
{
  ASSERT_THROW(read_from_mps("linear_programming/bad-mps-fixed-ranges-value.mps", false),
               std::logic_error);
}

TEST(mps_bounds, unsupported_or_invalid_mps_types)
{
  std::stringstream ss;
  static constexpr int NumMpsFiles = 2;
  for (int i = 1; i <= NumMpsFiles; ++i) {
    ss << "linear_programming/bad-mps-bound-" << i << ".mps";
    ASSERT_THROW(read_from_mps(ss.str(), false), std::logic_error);
    ss.str(std::string{});
    ss.clear();
  };
}

TEST_F(mip_with_bounds_test, mps)
{
  check_model(parse_mps_file("mixed_integer_programming/good-mip-mps-1.mps", false));
  auto mps = read_from_mps("mixed_integer_programming/good-mip-mps-1.mps", false);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
}

TEST_F(mip_with_bounds_test, lp)
{
  check_model(parse_lp_file("mixed_integer_programming/good-mip-mps-1.lp"));
}

TEST(mps_parser, good_mps_file_mip_no_marker)
{
  auto mps = read_from_mps("mixed_integer_programming/good-mip-mps-1-no-mark.mps", false);

  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(8000., mps.A_values[0][0]);
  EXPECT_EQ(4000., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(15., mps.A_values[1][0]);
  EXPECT_EQ(30., mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(40000., mps.b_values[0]);
  EXPECT_EQ(200., mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(100., mps.c_values[0]);
  EXPECT_EQ(150., mps.c_values[1]);
  ASSERT_EQ(int(2), mps.var_types.size());
  EXPECT_EQ('I', mps.var_types[0]);
  EXPECT_EQ('I', mps.var_types[1]);
  ASSERT_EQ(int(2), mps.variable_lower_bounds.size());
  EXPECT_EQ(0., mps.variable_lower_bounds[0]);
  EXPECT_EQ(0., mps.variable_lower_bounds[1]);
  ASSERT_EQ(int(2), mps.variable_upper_bounds.size());
  EXPECT_EQ(10., mps.variable_upper_bounds[0]);
  EXPECT_EQ(10., mps.variable_upper_bounds[1]);
}

TEST_F(mip_no_bounds_test, mps)
{
  check_model(parse_mps_file("mixed_integer_programming/good-mip-mps-no-bounds.mps", false));
}

TEST_F(mip_no_bounds_test, lp)
{
  check_model(parse_lp_file("mixed_integer_programming/good-mip-mps-no-bounds.lp"));
}

TEST_F(mip_partial_bounds_test, mps)
{
  check_model(parse_mps_file("mixed_integer_programming/good-mip-mps-partial-bounds.mps", false));
}

TEST_F(mip_partial_bounds_test, lp)
{
  check_model(parse_lp_file("mixed_integer_programming/good-mip-mps-partial-bounds.lp"));
}

#ifdef MPS_PARSER_WITH_BZIP2
TEST(mps_parser, good_mps_file_bzip2_compressed)
{
  auto mps = read_from_mps("linear_programming/good-mps-1.mps.bz2");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}
#endif  // MPS_PARSER_WITH_BZIP2

#ifdef MPS_PARSER_WITH_ZLIB
TEST(mps_parser, good_mps_file_zlib_compressed)
{
  auto mps = read_from_mps("linear_programming/good-mps-1.mps.gz");
  EXPECT_EQ("good-1", mps.problem_name);
  ASSERT_EQ(int(2), mps.row_names.size());
  EXPECT_EQ("ROW1", mps.row_names[0]);
  EXPECT_EQ("ROW2", mps.row_names[1]);
  ASSERT_EQ(int(2), mps.row_types.size());
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[0]);
  EXPECT_EQ(LesserThanOrEqual, mps.row_types[1]);
  EXPECT_EQ("COST", mps.objective_name);
  ASSERT_EQ(int(2), mps.var_names.size());
  EXPECT_EQ("VAR1", mps.var_names[0]);
  EXPECT_EQ("VAR2", mps.var_names[1]);
  ASSERT_EQ(int(2), mps.A_indices.size());
  ASSERT_EQ(int(2), mps.A_indices[0].size());
  EXPECT_EQ(int(0), mps.A_indices[0][0]);
  EXPECT_EQ(int(1), mps.A_indices[0][1]);
  ASSERT_EQ(int(2), mps.A_indices[1].size());
  EXPECT_EQ(int(0), mps.A_indices[1][0]);
  EXPECT_EQ(int(1), mps.A_indices[1][1]);
  ASSERT_EQ(int(2), mps.A_values.size());
  ASSERT_EQ(int(2), mps.A_values[0].size());
  EXPECT_EQ(3., mps.A_values[0][0]);
  EXPECT_EQ(4., mps.A_values[0][1]);
  ASSERT_EQ(int(2), mps.A_values[1].size());
  EXPECT_EQ(2.7, mps.A_values[1][0]);
  EXPECT_EQ(10.1, mps.A_values[1][1]);
  ASSERT_EQ(int(2), mps.b_values.size());
  EXPECT_EQ(5.4, mps.b_values[0]);
  EXPECT_EQ(4.9, mps.b_values[1]);
  ASSERT_EQ(int(2), mps.c_values.size());
  EXPECT_EQ(0.2, mps.c_values[0]);
  EXPECT_EQ(0.1, mps.c_values[1]);
}
#endif  // MPS_PARSER_WITH_ZLIB

// ================================================================================================
// QPS (Quadratic Programming) Support Tests
// ================================================================================================

// QPS-specific tests for quadratic programming support
TEST(qps_parser, quadratic_objective_basic)
{
  // Create a simple QPS test to verify quadratic objective parsing
  // This would require actual QPS test files - for now, test the API
  mps_data_model_t<int, double> model;

  // Test setting quadratic objective matrix
  std::vector<double> Q_values = {2.0, 1.0, 1.0, 2.0};  // 2x2 matrix
  std::vector<int> Q_indices   = {0, 1, 0, 1};
  std::vector<int> Q_offsets   = {0, 2, 4};  // CSR offsets

  model.set_quadratic_objective_matrix(Q_values.data(),
                                       Q_values.size(),
                                       Q_indices.data(),
                                       Q_indices.size(),
                                       Q_offsets.data(),
                                       Q_offsets.size());

  // Verify the data was stored correctly
  EXPECT_TRUE(model.has_quadratic_objective());
  EXPECT_EQ(4, model.get_quadratic_objective_values().size());
  EXPECT_EQ(2.0, model.get_quadratic_objective_values()[0]);
  EXPECT_EQ(1.0, model.get_quadratic_objective_values()[1]);
}

// Test actual QPS files from the dataset
TEST(qps_parser, test_qps_files)
{
  // Test QP_Test_1.qps if it exists
  if (file_exists("quadratic_programming/QP_Test_1.qps")) {
    auto parsed_data = parse_mps<int, double>(
      cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_1.qps", false);

    EXPECT_EQ("QP_Test_1", parsed_data.get_problem_name());
    EXPECT_EQ(2, parsed_data.get_n_variables());    // C------1 and C------2
    EXPECT_EQ(1, parsed_data.get_n_constraints());  // R------1
    EXPECT_TRUE(parsed_data.has_quadratic_objective());

    // Check variable bounds
    const auto& lower_bounds = parsed_data.get_variable_lower_bounds();
    const auto& upper_bounds = parsed_data.get_variable_upper_bounds();

    EXPECT_NEAR(2.0, lower_bounds[0], tolerance);    // C------1 lower bound
    EXPECT_NEAR(50.0, upper_bounds[0], tolerance);   // C------1 upper bound
    EXPECT_NEAR(-50.0, lower_bounds[1], tolerance);  // C------2 lower bound
    EXPECT_NEAR(50.0, upper_bounds[1], tolerance);   // C------2 upper bound
  }

  // Test QP_Test_2.qps if it exists
  if (file_exists("quadratic_programming/QP_Test_2.qps")) {
    auto parsed_data = parse_mps<int, double>(
      cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_2.qps", false);

    EXPECT_EQ("QP_Test_2", parsed_data.get_problem_name());
    EXPECT_EQ(3, parsed_data.get_n_variables());    // C------1, C------2, C------3
    EXPECT_EQ(1, parsed_data.get_n_constraints());  // R------1
    EXPECT_TRUE(parsed_data.has_quadratic_objective());

    // Check that quadratic objective matrix has values
    const auto& Q_values = parsed_data.get_quadratic_objective_values();
    EXPECT_GT(Q_values.size(), 0) << "Quadratic objective should have non-zero elements";
  }
}

// ================================================================================================
// MPS Round-Trip Tests (Read -> Write -> Read -> Compare)
// ================================================================================================

// Helper function to compare two data models
template <typename i_t, typename f_t>
void compare_data_models(const mps_data_model_t<i_t, f_t>& original,
                         const mps_data_model_t<i_t, f_t>& reloaded,
                         f_t tol = 1e-9)
{
  // Compare basic dimensions
  EXPECT_EQ(original.get_n_variables(), reloaded.get_n_variables());
  EXPECT_EQ(original.get_n_constraints(), reloaded.get_n_constraints());

  // Compare objective coefficients
  auto orig_c   = original.get_objective_coefficients();
  auto reload_c = reloaded.get_objective_coefficients();
  ASSERT_EQ(orig_c.size(), reload_c.size());
  for (size_t i = 0; i < orig_c.size(); ++i) {
    EXPECT_NEAR(orig_c[i], reload_c[i], tol) << "Objective coefficient mismatch at index " << i;
  }

  // Compare constraint matrix values
  auto orig_A   = original.get_constraint_matrix_values();
  auto reload_A = reloaded.get_constraint_matrix_values();
  ASSERT_EQ(orig_A.size(), reload_A.size());
  for (size_t i = 0; i < orig_A.size(); ++i) {
    EXPECT_NEAR(orig_A[i], reload_A[i], tol) << "Constraint matrix value mismatch at index " << i;
  }

  // Compare constraint matrix indices
  auto orig_A_idx   = original.get_constraint_matrix_indices();
  auto reload_A_idx = reloaded.get_constraint_matrix_indices();
  ASSERT_EQ(orig_A_idx.size(), reload_A_idx.size());
  for (size_t i = 0; i < orig_A_idx.size(); ++i) {
    EXPECT_EQ(orig_A_idx[i], reload_A_idx[i]) << "Constraint matrix index mismatch at index " << i;
  }

  // Compare constraint matrix offsets
  auto orig_A_off   = original.get_constraint_matrix_offsets();
  auto reload_A_off = reloaded.get_constraint_matrix_offsets();
  ASSERT_EQ(orig_A_off.size(), reload_A_off.size());
  for (size_t i = 0; i < orig_A_off.size(); ++i) {
    EXPECT_EQ(orig_A_off[i], reload_A_off[i]) << "Constraint matrix offset mismatch at index " << i;
  }

  // Compare variable bounds
  auto orig_lb   = original.get_variable_lower_bounds();
  auto reload_lb = reloaded.get_variable_lower_bounds();
  ASSERT_EQ(orig_lb.size(), reload_lb.size());
  for (size_t i = 0; i < orig_lb.size(); ++i) {
    if (std::isinf(orig_lb[i]) && std::isinf(reload_lb[i])) {
      EXPECT_EQ(std::signbit(orig_lb[i]), std::signbit(reload_lb[i]))
        << "Variable lower bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_lb[i], reload_lb[i], tol) << "Variable lower bound mismatch at index " << i;
    }
  }

  auto orig_ub   = original.get_variable_upper_bounds();
  auto reload_ub = reloaded.get_variable_upper_bounds();
  ASSERT_EQ(orig_ub.size(), reload_ub.size());
  for (size_t i = 0; i < orig_ub.size(); ++i) {
    if (std::isinf(orig_ub[i]) && std::isinf(reload_ub[i])) {
      EXPECT_EQ(std::signbit(orig_ub[i]), std::signbit(reload_ub[i]))
        << "Variable upper bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_ub[i], reload_ub[i], tol) << "Variable upper bound mismatch at index " << i;
    }
  }

  // Compare constraint bounds
  auto orig_cl   = original.get_constraint_lower_bounds();
  auto reload_cl = reloaded.get_constraint_lower_bounds();
  ASSERT_EQ(orig_cl.size(), reload_cl.size());
  for (size_t i = 0; i < orig_cl.size(); ++i) {
    if (std::isinf(orig_cl[i]) && std::isinf(reload_cl[i])) {
      EXPECT_EQ(std::signbit(orig_cl[i]), std::signbit(reload_cl[i]))
        << "Constraint lower bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_cl[i], reload_cl[i], tol)
        << "Constraint lower bound mismatch at index " << i;
    }
  }

  auto orig_cu   = original.get_constraint_upper_bounds();
  auto reload_cu = reloaded.get_constraint_upper_bounds();
  ASSERT_EQ(orig_cu.size(), reload_cu.size());
  for (size_t i = 0; i < orig_cu.size(); ++i) {
    if (std::isinf(orig_cu[i]) && std::isinf(reload_cu[i])) {
      EXPECT_EQ(std::signbit(orig_cu[i]), std::signbit(reload_cu[i]))
        << "Constraint upper bound infinity sign mismatch at index " << i;
    } else {
      EXPECT_NEAR(orig_cu[i], reload_cu[i], tol)
        << "Constraint upper bound mismatch at index " << i;
    }
  }

  // Compare quadratic objective if present
  EXPECT_EQ(original.has_quadratic_objective(), reloaded.has_quadratic_objective());
  if (original.has_quadratic_objective() && reloaded.has_quadratic_objective()) {
    auto orig_Q       = original.get_quadratic_objective_values();
    auto orig_Q_idx   = original.get_quadratic_objective_indices();
    auto orig_Q_off   = original.get_quadratic_objective_offsets();
    auto reload_Q     = reloaded.get_quadratic_objective_values();
    auto reload_Q_idx = reloaded.get_quadratic_objective_indices();
    auto reload_Q_off = reloaded.get_quadratic_objective_offsets();

    // Compare Q matrix structure and values
    ASSERT_EQ(orig_Q.size(), reload_Q.size()) << "Q values size mismatch";
    ASSERT_EQ(orig_Q_idx.size(), reload_Q_idx.size()) << "Q indices size mismatch";
    ASSERT_EQ(orig_Q_off.size(), reload_Q_off.size()) << "Q offsets size mismatch";

    for (size_t i = 0; i < orig_Q.size(); ++i) {
      EXPECT_NEAR(orig_Q[i], reload_Q[i], tol) << "Q value mismatch at index " << i;
    }
    for (size_t i = 0; i < orig_Q_idx.size(); ++i) {
      EXPECT_EQ(orig_Q_idx[i], reload_Q_idx[i]) << "Q index mismatch at index " << i;
    }
    for (size_t i = 0; i < orig_Q_off.size(); ++i) {
      EXPECT_EQ(orig_Q_off[i], reload_Q_off[i]) << "Q offset mismatch at index " << i;
    }
  }
}

TEST(mps_roundtrip, linear_programming_basic)
{
  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/linear_programming/good-mps-1.mps";
  std::string temp_file = "/tmp/mps_roundtrip_lp_test.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, true);

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

TEST(mps_roundtrip, linear_programming_with_bounds)
{
  if (!file_exists("linear_programming/lp_model_with_var_bounds.mps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/linear_programming/lp_model_with_var_bounds.mps";
  std::string temp_file = "/tmp/mps_roundtrip_lp_bounds_test.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, false);

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

TEST(mps_roundtrip, quadratic_programming_qp_test_1)
{
  if (!file_exists("quadratic_programming/QP_Test_1.qps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_1.qps";
  std::string temp_file = "/tmp/mps_roundtrip_qp_test_1.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, false);
  ASSERT_TRUE(original.has_quadratic_objective()) << "Original should have quadratic objective";

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);
  ASSERT_TRUE(reloaded.has_quadratic_objective()) << "Reloaded should have quadratic objective";

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

TEST(mps_roundtrip, quadratic_programming_qp_test_2)
{
  if (!file_exists("quadratic_programming/QP_Test_2.qps")) {
    GTEST_SKIP() << "Test file not found";
  }

  std::string input_file =
    cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/QP_Test_2.qps";
  std::string temp_file = "/tmp/mps_roundtrip_qp_test_2.mps";

  // Read original
  auto original = parse_mps<int, double>(input_file, false);
  ASSERT_TRUE(original.has_quadratic_objective()) << "Original should have quadratic objective";

  // Write to temp file
  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  // Read back
  auto reloaded = parse_mps<int, double>(temp_file, false);
  ASSERT_TRUE(reloaded.has_quadratic_objective()) << "Reloaded should have quadratic objective";

  // Compare
  compare_data_models(original, reloaded);

  // Cleanup
  std::filesystem::remove(temp_file);
}

// ================================================================================================
// LP -> MPS Round-Trip Tests (Read LP -> Write MPS -> Read MPS -> Compare)
// ================================================================================================
// Parses an LP file, writes the resulting data model out as MPS, reads it
// back, and checks that the reloaded data model matches the one produced by
// the LP parser. Exercises the LP reader + the writer + the MPS reader end
// to end, without trusting any direct LP<->MPS comparison.

TEST_F(good_mps_1_test, lp_roundtrip)
{
  std::string temp_file = "/tmp/lp_roundtrip_lp_basic.mps";

  auto original = parse_lp_file("linear_programming/good-mps-1.lp");

  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  auto reloaded = parse_mps<int, double>(temp_file, false);

  compare_data_models(original, reloaded);

  std::filesystem::remove(temp_file);
}

TEST_F(up_low_bounds_test, lp_roundtrip)
{
  std::string temp_file = "/tmp/lp_roundtrip_lp_bounds.mps";

  auto original = parse_lp_file("linear_programming/lp_model_with_var_bounds.lp");

  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  auto reloaded = parse_mps<int, double>(temp_file, false);

  compare_data_models(original, reloaded);

  std::filesystem::remove(temp_file);
}

TEST_F(mip_with_bounds_test, lp_roundtrip)
{
  std::string temp_file = "/tmp/lp_roundtrip_mip_basic.mps";

  auto original = parse_lp_file("mixed_integer_programming/good-mip-mps-1.lp");

  mps_writer_t<int, double> writer(original);
  writer.write(temp_file);

  auto reloaded = parse_mps<int, double>(temp_file, false);

  compare_data_models(original, reloaded);

  std::filesystem::remove(temp_file);
}

// ================================================================================================
// LP syntax / feature / error-path tests (parse_lp on inline LP content)
// ================================================================================================

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

// ================================================================================================
// parse_optimization_file dispatch tests
//
// Verifies the extension-based dispatch used by cuopt_cli and the C API.
// ================================================================================================

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
