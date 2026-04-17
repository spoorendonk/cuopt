/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mps_parser/lp_parser.hpp>

#include <lp_parser.hpp>
#include <parser_finalize.hpp>
#include <utilities/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::mps_parser {

namespace {

// ===========================================================================
// Small character / string helpers
// ===========================================================================

bool is_name_start_char(char c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool is_name_char(char c) { return is_name_start_char(c) || (c >= '0' && c <= '9') || c == '.'; }

char to_lower(char c)
{
  if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
  return c;
}

std::string lowercase(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back(to_lower(c));
  return out;
}

// ===========================================================================
// LP section-keyword classifiers (case-insensitive; callers pass lowercased)
// ===========================================================================

bool is_objective_min_keyword(std::string_view lower)
{
  return lower == "minimize" || lower == "minimum" || lower == "min";
}

bool is_objective_max_keyword(std::string_view lower)
{
  return lower == "maximize" || lower == "maximum" || lower == "max";
}

bool is_bounds_keyword(std::string_view lower) { return lower == "bounds" || lower == "bound"; }

bool is_generals_keyword(std::string_view lower)
{
  return lower == "generals" || lower == "general" || lower == "gen" || lower == "integer" ||
         lower == "integers";
}

bool is_binaries_keyword(std::string_view lower)
{
  return lower == "binaries" || lower == "binary" || lower == "bin";
}

bool is_end_keyword(std::string_view lower) { return lower == "end"; }

bool is_free_keyword(std::string_view lower) { return lower == "free"; }

bool is_infinity_text(std::string_view lower) { return lower == "inf" || lower == "infinity"; }

// ===========================================================================
// Token stream
// ===========================================================================

// Kinds of tokens produced by the LP tokenizer. The grammar is small enough
// that a hand-written scanner is easier to follow than a regex engine.
enum class LpTokenKind {
  Number,     // 12, -3.5, 1e-6
  Name,       // variable names and section keywords (also the literal "inf")
  Plus,       // +
  Minus,      // -
  Star,       // *
  Caret,      // ^
  Slash,      // /
  LessEq,     // <= (and < treated as <=)
  GreaterEq,  // >= (and > treated as >=)
  Equal,      // =
  LBracket,   // [
  RBracket,   // ]
  Colon,      // :
  Eof,
};

struct LpToken {
  LpTokenKind kind;
  // Owned copy of the token text so the token stream is independent of the
  // backing file buffer.
  std::string text;
  int line;
  // True when this is the first non-whitespace/non-comment token on its line.
  // Used to detect section headers without emitting newline tokens.
  bool is_line_start;
};

// ===========================================================================
// Parsing engine — holds all transient parsing state and writes directly
// into the lp_parser_t's public fields. Strictly internal to this TU.
// ===========================================================================

template <typename i_t, typename f_t>
class LpParseEngine {
 public:
  LpParseEngine(lp_parser_t<i_t, f_t>& out, const std::string& file);

 private:
  lp_parser_t<i_t, f_t>& out_;
  std::vector<LpToken> tokens_;
  size_t tok_pos_{0};

  std::unordered_map<std::string, i_t> var_names_map_{};
  std::unordered_map<std::string, i_t> row_names_map_{};
  std::unordered_set<i_t> bounds_defined_for_var_id_{};
  // Counter used to generate row names for unlabeled constraints (R0, R1, ...).
  i_t anon_row_counter_{0};

  // File → token stream.
  void read_and_tokenize(const std::string& file);
  void tokenize(const std::string& text);

  // Token stream helpers.
  const LpToken& peek(size_t lookahead = 0) const;
  const LpToken& advance();
  bool at_eof() const;
  bool match(LpTokenKind kind);
  void expect(LpTokenKind kind, const char* context);
  static bool name_equals_ci(const LpToken& tok, std::string_view lower);
  bool is_infinity_keyword(const LpToken& tok) const;
  f_t number_from_text(const std::string& text) const;

  // Variable bookkeeping.
  i_t get_or_add_var(std::string_view name);

  // Top-level dispatch.
  void parse_all();

  // Section parsers.
  void parse_objective_section();
  void parse_constraints_section();
  void parse_bounds_section();
  void parse_integer_list_section(bool is_binary);

  // Expression parsers.
  struct LinearTerm {
    i_t var_id;
    f_t coeff;
  };
  void parse_linear_expression(std::vector<LinearTerm>& out_terms, f_t& out_constant);
  void parse_quadratic_bracket(std::vector<LinearTerm>& out_linear, int outer_sign);

  // Atomic readers.
  f_t parse_signed_number();

  // Section header classification.
  enum class SectionKind {
    None,
    Objective,
    Constraints,
    LazyConstraints,
    Bounds,
    Generals,
    Binaries,
    End,
  };
  SectionKind try_consume_section_header();
  void reject_unsupported_section();
  bool at_section_boundary() const;
};

// ---- Constructor ----------------------------------------------------------

template <typename i_t, typename f_t>
LpParseEngine<i_t, f_t>::LpParseEngine(lp_parser_t<i_t, f_t>& out, const std::string& file)
  : out_(out)
{
  read_and_tokenize(file);
  parse_all();
}

// ---- File I/O + tokenizer -------------------------------------------------

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::read_and_tokenize(const std::string& file)
{
  std::ifstream in(file, std::ios::binary);
  mps_parser_expects(in.good(),
                     error_type_t::ValidationError,
                     "Error opening LP file! Given path: %s",
                     file.c_str());
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string text = std::move(ss).str();
  tokenize(text);
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::tokenize(const std::string& text)
{
  size_t i       = 0;
  int line       = 1;
  bool at_start  = true;  // next non-whitespace token starts a new line
  const size_t n = text.size();

  auto push = [&](LpTokenKind kind, std::string s) {
    tokens_.push_back(LpToken{kind, std::move(s), line, at_start});
    at_start = false;
  };

  while (i < n) {
    char c = text[i];

    if (c == '\n') {
      ++line;
      at_start = true;
      ++i;
      continue;
    }
    if (c == '\r') {
      ++i;
      continue;
    }
    if (c == '\\') {  // LP comment: '\' through end of line
      while (i < n && text[i] != '\n')
        ++i;
      continue;
    }
    if (c == ' ' || c == '\t') {
      ++i;
      continue;
    }

    // Single-character punctuation.
    switch (c) {
      case '+':
        push(LpTokenKind::Plus, "+");
        ++i;
        continue;
      case '-':
        push(LpTokenKind::Minus, "-");
        ++i;
        continue;
      case '*':
        push(LpTokenKind::Star, "*");
        ++i;
        continue;
      case '^':
        push(LpTokenKind::Caret, "^");
        ++i;
        continue;
      case '/':
        push(LpTokenKind::Slash, "/");
        ++i;
        continue;
      case '[':
        push(LpTokenKind::LBracket, "[");
        ++i;
        continue;
      case ']':
        push(LpTokenKind::RBracket, "]");
        ++i;
        continue;
      case ':':
        push(LpTokenKind::Colon, ":");
        ++i;
        continue;
      case '=':
        push(LpTokenKind::Equal, "=");
        ++i;
        continue;
      default: break;
    }

    // Relation operators. Our LP dialect treats bare '<' as '<=' and bare
    // '>' as '>='; we do the same for robustness.
    if (c == '<') {
      if (i + 1 < n && text[i + 1] == '=') {
        push(LpTokenKind::LessEq, "<=");
        i += 2;
      } else {
        push(LpTokenKind::LessEq, "<");
        ++i;
      }
      continue;
    }
    if (c == '>') {
      if (i + 1 < n && text[i + 1] == '=') {
        push(LpTokenKind::GreaterEq, ">=");
        i += 2;
      } else {
        push(LpTokenKind::GreaterEq, ">");
        ++i;
      }
      continue;
    }

    // Numbers: [0-9]+ ('.' [0-9]*)? ([eE] [+-]? [0-9]+)?  |  '.' [0-9]+ ...
    if ((c >= '0' && c <= '9') ||
        (c == '.' && i + 1 < n && text[i + 1] >= '0' && text[i + 1] <= '9')) {
      size_t start = i;
      while (i < n && text[i] >= '0' && text[i] <= '9')
        ++i;
      if (i < n && text[i] == '.') {
        ++i;
        while (i < n && text[i] >= '0' && text[i] <= '9')
          ++i;
      }
      if (i < n && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < n && (text[i] == '+' || text[i] == '-')) ++i;
        mps_parser_expects(i < n && text[i] >= '0' && text[i] <= '9',
                           error_type_t::ValidationError,
                           "Malformed number (missing exponent digits) at line %d",
                           line);
        while (i < n && text[i] >= '0' && text[i] <= '9')
          ++i;
      }
      push(LpTokenKind::Number, text.substr(start, i - start));
      continue;
    }

    // Names: [A-Za-z_] [A-Za-z0-9_.]*
    if (is_name_start_char(c)) {
      size_t start = i;
      while (i < n && is_name_char(text[i]))
        ++i;
      push(LpTokenKind::Name, text.substr(start, i - start));
      continue;
    }

    mps_parser_expects(false,
                       error_type_t::ValidationError,
                       "Unexpected character '%c' (0x%02x) at line %d in LP file",
                       c,
                       static_cast<unsigned>(static_cast<unsigned char>(c)),
                       line);
  }

  tokens_.push_back(LpToken{LpTokenKind::Eof, "", line, true});
}

// ---- Token stream helpers --------------------------------------------------

template <typename i_t, typename f_t>
const LpToken& LpParseEngine<i_t, f_t>::peek(size_t lookahead) const
{
  size_t idx = tok_pos_ + lookahead;
  if (idx >= tokens_.size()) return tokens_.back();  // guaranteed Eof token
  return tokens_[idx];
}

template <typename i_t, typename f_t>
const LpToken& LpParseEngine<i_t, f_t>::advance()
{
  const LpToken& t = tokens_[tok_pos_];
  if (tok_pos_ + 1 < tokens_.size()) ++tok_pos_;
  return t;
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::at_eof() const
{
  return peek().kind == LpTokenKind::Eof;
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::match(LpTokenKind kind)
{
  if (peek().kind == kind) {
    advance();
    return true;
  }
  return false;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::expect(LpTokenKind kind, const char* context)
{
  mps_parser_expects(peek().kind == kind,
                     error_type_t::ValidationError,
                     "LP parse error at line %d: expected %s, got '%s'",
                     peek().line,
                     context,
                     peek().text.c_str());
  advance();
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::name_equals_ci(const LpToken& tok, std::string_view lower)
{
  if (tok.kind != LpTokenKind::Name) return false;
  if (tok.text.size() != lower.size()) return false;
  for (size_t i = 0; i < tok.text.size(); ++i) {
    if (to_lower(tok.text[i]) != lower[i]) return false;
  }
  return true;
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::is_infinity_keyword(const LpToken& tok) const
{
  return tok.kind == LpTokenKind::Name && is_infinity_text(lowercase(tok.text));
}

template <typename i_t, typename f_t>
f_t LpParseEngine<i_t, f_t>::number_from_text(const std::string& text) const
{
  try {
    if constexpr (std::is_same_v<f_t, float>) {
      return std::stof(text);
    } else {
      return std::stod(text);
    }
  } catch (...) {
    mps_parser_expects(false,
                       error_type_t::ValidationError,
                       "LP parse error: could not parse number '%s'",
                       text.c_str());
  }
  return f_t(0);  // unreachable; mps_parser_expects throws
}

// ---- Variable bookkeeping --------------------------------------------------

template <typename i_t, typename f_t>
i_t LpParseEngine<i_t, f_t>::get_or_add_var(std::string_view name)
{
  std::string key(name);
  auto it = var_names_map_.find(key);
  if (it != var_names_map_.end()) return it->second;
  i_t id = static_cast<i_t>(out_.var_names.size());
  out_.var_names.push_back(key);
  var_names_map_.emplace(std::move(key), id);
  out_.var_types.push_back('C');
  out_.c_values.push_back(f_t(0));
  out_.variable_lower_bounds.push_back(f_t(0));
  out_.variable_upper_bounds.push_back(std::numeric_limits<f_t>::infinity());
  return id;
}

// ---- Section header detection ---------------------------------------------

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::at_section_boundary() const
{
  if (at_eof()) return true;
  const LpToken& t = peek();
  if (!t.is_line_start || t.kind != LpTokenKind::Name) return false;
  std::string lower = lowercase(t.text);

  if (is_objective_min_keyword(lower) || is_objective_max_keyword(lower)) return true;
  if (is_bounds_keyword(lower)) return true;
  if (is_generals_keyword(lower)) return true;
  if (is_binaries_keyword(lower)) return true;
  if (is_end_keyword(lower)) return true;

  // Multi-word section headers: "Subject To", "Such That", "Lazy Constraints",
  // and the unsupported "User Cuts" / "General Constraints".
  const LpToken& t2 = peek(1);
  if (lower == "subject" && name_equals_ci(t2, "to")) return true;
  if (lower == "such" && name_equals_ci(t2, "that")) return true;
  if (lower == "st" || lower == "s.t.") return true;
  if (lower == "lazy" && name_equals_ci(t2, "constraints")) return true;
  if (lower == "user" && name_equals_ci(t2, "cuts")) return true;
  // "General Constraints" is unsupported but still a section boundary;
  // reject_unsupported_section() throws before we'd keep parsing.
  if (lower == "general" && name_equals_ci(t2, "constraints")) return true;

  // Unsupported single-token sections.
  if (lower == "sos") return true;
  if (lower == "semi" && peek(1).kind == LpTokenKind::Minus &&
      name_equals_ci(peek(2), "continuous"))
    return true;
  if (lower == "sc") return true;
  if (lower == "pwlobj") return true;
  if (lower == "scenarios" || lower == "scenario") return true;

  return false;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::reject_unsupported_section()
{
  std::string name  = peek().text;
  std::string lower = lowercase(name);
  // Compose a useful display name for multi-word headers.
  if (lower == "semi" && peek(1).kind == LpTokenKind::Minus &&
      name_equals_ci(peek(2), "continuous")) {
    name = "Semi-continuous";
  } else if (lower == "user" && name_equals_ci(peek(1), "cuts")) {
    name = "User Cuts";
  } else if (lower == "general" && name_equals_ci(peek(1), "constraints")) {
    name = "General Constraints";
  }
  mps_parser_expects(false,
                     error_type_t::ValidationError,
                     "LP section '%s' is not supported (scope is LP/MIP/QP only)",
                     name.c_str());
}

template <typename i_t, typename f_t>
typename LpParseEngine<i_t, f_t>::SectionKind LpParseEngine<i_t, f_t>::try_consume_section_header()
{
  if (at_eof()) return SectionKind::None;
  const LpToken& t = peek();
  mps_parser_expects(t.is_line_start && t.kind == LpTokenKind::Name,
                     error_type_t::ValidationError,
                     "LP parse error at line %d: expected section header, got '%s'",
                     t.line,
                     t.text.c_str());
  std::string lower = lowercase(t.text);

  if (is_objective_min_keyword(lower)) {
    out_.maximize = false;
    advance();
    return SectionKind::Objective;
  }
  if (is_objective_max_keyword(lower)) {
    out_.maximize = true;
    advance();
    return SectionKind::Objective;
  }
  if (lower == "subject" && name_equals_ci(peek(1), "to")) {
    advance();
    advance();
    return SectionKind::Constraints;
  }
  if (lower == "such" && name_equals_ci(peek(1), "that")) {
    advance();
    advance();
    return SectionKind::Constraints;
  }
  if (lower == "st" || lower == "s.t.") {
    advance();
    return SectionKind::Constraints;
  }
  if (lower == "lazy" && name_equals_ci(peek(1), "constraints")) {
    advance();
    advance();
    return SectionKind::LazyConstraints;
  }
  if (is_bounds_keyword(lower)) {
    advance();
    return SectionKind::Bounds;
  }
  if (is_generals_keyword(lower)) {
    // "General" alone means Generals; "General Constraints" is unsupported.
    if (lower == "general" && name_equals_ci(peek(1), "constraints")) {
      reject_unsupported_section();
    }
    advance();
    return SectionKind::Generals;
  }
  if (is_binaries_keyword(lower)) {
    advance();
    return SectionKind::Binaries;
  }
  if (is_end_keyword(lower)) {
    advance();
    return SectionKind::End;
  }

  // Known unsupported sections → throw with a clear message.
  reject_unsupported_section();
  return SectionKind::None;  // unreachable
}

// ---- Expression parsing ---------------------------------------------------

template <typename i_t, typename f_t>
f_t LpParseEngine<i_t, f_t>::parse_signed_number()
{
  int sign = 1;
  if (match(LpTokenKind::Minus)) {
    sign = -1;
  } else {
    match(LpTokenKind::Plus);  // optional leading '+'
  }
  if (is_infinity_keyword(peek())) {
    advance();
    return sign > 0 ? std::numeric_limits<f_t>::infinity() : -std::numeric_limits<f_t>::infinity();
  }
  mps_parser_expects(peek().kind == LpTokenKind::Number,
                     error_type_t::ValidationError,
                     "LP parse error at line %d: expected a number, got '%s'",
                     peek().line,
                     peek().text.c_str());
  f_t val = number_from_text(peek().text);
  advance();
  return sign * val;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_linear_expression(std::vector<LinearTerm>& out_terms,
                                                      f_t& out_constant)
{
  out_constant = f_t(0);
  int sign     = 1;
  bool first   = true;

  while (true) {
    // A quadratic bracket ends the linear expression. If the bracket is
    // preceded by a sign, leave the sign unconsumed so the caller can
    // attribute it to the bracket.
    if (peek().kind == LpTokenKind::LBracket) break;
    if ((peek().kind == LpTokenKind::Plus || peek().kind == LpTokenKind::Minus) &&
        peek(1).kind == LpTokenKind::LBracket) {
      break;
    }

    if (peek().kind == LpTokenKind::Plus) {
      advance();
      sign = 1;
    } else if (peek().kind == LpTokenKind::Minus) {
      advance();
      sign = -1;
    } else if (!first) {
      // No sign between terms → expression ends here. (Relation tokens,
      // ']', section headers, EOF all terminate.)
      break;
    }

    // A term is: (number ('*')?)? varname  |  number (constant)  |  varname.
    // 'inf' is a bounds-only keyword and never appears here.
    f_t coeff      = f_t(1);
    bool had_coeff = false;
    if (peek().kind == LpTokenKind::Number) {
      coeff     = number_from_text(peek().text);
      had_coeff = true;
      advance();
      match(LpTokenKind::Star);  // optional '*'
    }

    if (peek().kind == LpTokenKind::Name && !at_section_boundary() &&
        !is_free_keyword(lowercase(peek().text)) && !is_infinity_keyword(peek())) {
      std::string var_name = peek().text;
      advance();
      i_t id = get_or_add_var(var_name);
      out_terms.push_back({id, sign * coeff});
    } else if (had_coeff) {
      // It was a pure number → contributes to the constant.
      out_constant += sign * coeff;
    } else {
      // Nothing consumed this iteration → not a term, stop.
      if (!first) {
        // We consumed a sign without a term: malformed.
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "LP parse error at line %d: expected a term after '+' or '-'",
                           peek().line);
      }
      break;
    }

    first = false;
    sign  = 1;
  }
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_quadratic_bracket(std::vector<LinearTerm>& out_linear,
                                                      int outer_sign)
{
  expect(LpTokenKind::LBracket, "'[' at start of quadratic section");

  // Accumulate raw LP-format entries first (diagonal vs off-diagonal), then
  // apply the /2 convention and outer sign after we see the closing bracket.
  std::vector<std::tuple<i_t, i_t, f_t>> raw_quad;

  int sign   = 1;
  bool first = true;
  while (peek().kind != LpTokenKind::RBracket) {
    mps_parser_expects(!at_eof(),
                       error_type_t::ValidationError,
                       "LP parse error: unterminated quadratic '[' section");

    if (peek().kind == LpTokenKind::Plus) {
      advance();
      sign = 1;
    } else if (peek().kind == LpTokenKind::Minus) {
      advance();
      sign = -1;
    } else if (!first) {
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected '+' or '-' between "
                         "quadratic terms, got '%s'",
                         peek().line,
                         peek().text.c_str());
    }

    f_t coeff = f_t(1);
    if (peek().kind == LpTokenKind::Number) {
      coeff = number_from_text(peek().text);
      advance();
      match(LpTokenKind::Star);  // optional
    }

    mps_parser_expects(peek().kind == LpTokenKind::Name,
                       error_type_t::ValidationError,
                       "LP parse error at line %d: expected variable name in quadratic term",
                       peek().line);
    std::string var1 = peek().text;
    advance();
    i_t i1 = get_or_add_var(var1);

    if (match(LpTokenKind::Caret)) {
      // Must be "^ 2".
      mps_parser_expects(peek().kind == LpTokenKind::Number && peek().text == "2",
                         error_type_t::ValidationError,
                         "LP parse error at line %d: only 'x ^ 2' is supported in quadratic "
                         "terms (got '%s')",
                         peek().line,
                         peek().text.c_str());
      advance();
      raw_quad.emplace_back(i1, i1, sign * coeff);
    } else if (match(LpTokenKind::Star)) {
      mps_parser_expects(peek().kind == LpTokenKind::Name,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected variable name after '*' in "
                         "quadratic cross term",
                         peek().line);
      std::string var2 = peek().text;
      advance();
      i_t i2 = get_or_add_var(var2);
      // Store in upper-triangular form (i <= j) to match QUADOBJ convention.
      i_t a = std::min(i1, i2);
      i_t b = std::max(i1, i2);
      raw_quad.emplace_back(a, b, sign * coeff);
    } else {
      // Purely linear term inside the brackets — permitted as long as the
      // surrounding /2 convention is respected (the linear term is scaled
      // the same way as the quadratic ones).
      out_linear.push_back({i1, sign * coeff});
    }

    first = false;
    sign  = 1;
  }
  expect(LpTokenKind::RBracket, "closing ']' of quadratic section");

  // Optional "/ 2" suffix.
  bool has_div2 = false;
  if (peek().kind == LpTokenKind::Slash && peek(1).kind == LpTokenKind::Number &&
      peek(1).text == "2") {
    advance();
    advance();
    has_div2 = true;
  }

  // Apply the /2 convention and the QUADOBJ-convention scaling so that
  // finalize_problem()'s expansion to full symmetric and *0.5 factor yield
  // the right Q for cuOpt's 'x^T Q x' form.
  //
  //   LP term ([...]/2):                    →  quadobj entry
  //     diagonal  c x^2   (actual = c/2)    →  c
  //     off-diag  c x*y   (actual = c/2)    →  c/2
  //   LP term ([...]) (no /2):
  //     diagonal  c x^2   (actual = c)      →  2c
  //     off-diag  c x*y   (actual = c)      →  c
  const f_t sign_scale = static_cast<f_t>(outer_sign);
  for (auto& [a, b, v] : raw_quad) {
    if (a == b) {
      // diagonal: * 2 when no /2
      if (!has_div2) v *= f_t(2);
    } else {
      // off-diagonal: / 2 when /2 present
      if (has_div2) v /= f_t(2);
    }
    out_.quadobj_entries.emplace_back(a, b, sign_scale * v);
  }
  // Linear terms inside the brackets also pick up the /2 scaling, then the
  // outer sign.
  if (has_div2) {
    for (auto& lt : out_linear)
      lt.coeff /= f_t(2);
  }
  if (outer_sign < 0) {
    for (auto& lt : out_linear)
      lt.coeff = -lt.coeff;
  }
}

// ---- Section bodies -------------------------------------------------------

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_objective_section()
{
  // Optional "name:" label.
  if (peek().kind == LpTokenKind::Name && peek(1).kind == LpTokenKind::Colon &&
      !at_section_boundary()) {
    out_.objective_name = peek().text;
    advance();
    advance();
  }

  std::vector<LinearTerm> linear;
  f_t constant = 0;
  parse_linear_expression(linear, constant);

  // Optional quadratic bracket, possibly preceded by a sign. In this LP
  // dialect the bracket sits inside the objective expression and can
  // appear before or after linear terms; we support one bracket followed
  // by (possibly) more linear terms.
  int quad_sign = 1;
  if (peek().kind == LpTokenKind::Plus && peek(1).kind == LpTokenKind::LBracket) {
    advance();
  } else if (peek().kind == LpTokenKind::Minus && peek(1).kind == LpTokenKind::LBracket) {
    advance();
    quad_sign = -1;
  }
  if (peek().kind == LpTokenKind::LBracket) {
    std::vector<LinearTerm> in_bracket_linear;
    parse_quadratic_bracket(in_bracket_linear, quad_sign);
    for (const auto& lt : in_bracket_linear)
      linear.push_back(lt);

    // More linear terms may follow the bracket.
    std::vector<LinearTerm> more;
    f_t more_constant = 0;
    parse_linear_expression(more, more_constant);
    for (const auto& lt : more)
      linear.push_back(lt);
    constant += more_constant;
  }

  // Apply linear terms to the objective vector. Coefficients accumulate in
  // case the same variable appears twice.
  for (const auto& lt : linear) {
    if (static_cast<size_t>(lt.var_id) >= out_.c_values.size()) {
      out_.c_values.resize(lt.var_id + 1, f_t(0));
    }
    out_.c_values[lt.var_id] += lt.coeff;
  }
  // A constant term in the objective becomes the objective offset.
  out_.objective_offset_value += constant;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_constraints_section()
{
  // Lazy constraints are treated as regular constraints (matching the MPS
  // parser's LAZYCONS handling).
  while (!at_section_boundary()) {
    // Optional "name:" label — present iff the first two tokens are Name + ':'.
    std::string row_name;
    if (peek().kind == LpTokenKind::Name && peek(1).kind == LpTokenKind::Colon) {
      row_name = peek().text;
      advance();
      advance();
    } else {
      row_name = "R" + std::to_string(anon_row_counter_++);
    }

    std::vector<LinearTerm> linear;
    f_t lhs_constant = 0;
    parse_linear_expression(linear, lhs_constant);

    RowType row_type{};
    if (peek().kind == LpTokenKind::LessEq) {
      row_type = LesserThanOrEqual;
      advance();
    } else if (peek().kind == LpTokenKind::GreaterEq) {
      row_type = GreaterThanOrEqual;
      advance();
    } else if (peek().kind == LpTokenKind::Equal) {
      row_type = Equality;
      advance();
    } else {
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected a relation operator "
                         "(<=, >=, =) in constraint, got '%s'",
                         peek().line,
                         peek().text.c_str());
    }

    f_t rhs_value = parse_signed_number();
    // Any constant that appeared on the LHS is moved to the RHS with a sign flip.
    rhs_value -= lhs_constant;

    // Register the row.
    mps_parser_expects(row_names_map_.find(row_name) == row_names_map_.end(),
                       error_type_t::ValidationError,
                       "Duplicate constraint name '%s'",
                       row_name.c_str());
    i_t row_id = static_cast<i_t>(out_.row_names.size());
    out_.row_names.push_back(row_name);
    row_names_map_.emplace(row_name, row_id);
    out_.row_types.push_back(row_type);
    out_.b_values.push_back(rhs_value);

    // Collect the constraint row of A. Coefficients accumulate for repeated
    // variables; we sort by var_id for deterministic CSR output.
    std::unordered_map<i_t, f_t> row_coeffs;
    for (const auto& lt : linear)
      row_coeffs[lt.var_id] += lt.coeff;
    std::vector<std::pair<i_t, f_t>> ordered(row_coeffs.begin(), row_coeffs.end());
    std::sort(ordered.begin(), ordered.end());
    std::vector<i_t> indices;
    std::vector<f_t> values;
    indices.reserve(ordered.size());
    values.reserve(ordered.size());
    for (const auto& [vid, val] : ordered) {
      if (val == f_t(0)) continue;
      indices.push_back(vid);
      values.push_back(val);
    }
    out_.A_indices.push_back(std::move(indices));
    out_.A_values.push_back(std::move(values));
  }
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_bounds_section()
{
  while (!at_section_boundary()) {
    // Either starts with a variable name or with a signed number. 'inf' /
    // 'infinity' tokens are Names but only valid in the lb-first form.
    if (peek().kind == LpTokenKind::Name && !is_infinity_keyword(peek())) {
      std::string var_name = peek().text;
      advance();
      i_t vid = get_or_add_var(var_name);
      bounds_defined_for_var_id_.insert(vid);

      // Suffix after the name.
      if (peek().kind == LpTokenKind::Name && is_free_keyword(lowercase(peek().text))) {
        advance();
        out_.variable_lower_bounds[vid] = -std::numeric_limits<f_t>::infinity();
        out_.variable_upper_bounds[vid] = std::numeric_limits<f_t>::infinity();
      } else if (match(LpTokenKind::LessEq)) {
        // x <= ub
        out_.variable_upper_bounds[vid] = parse_signed_number();
      } else if (match(LpTokenKind::GreaterEq)) {
        // x >= lb
        out_.variable_lower_bounds[vid] = parse_signed_number();
      } else if (match(LpTokenKind::Equal)) {
        // x = value (fixed)
        f_t v                           = parse_signed_number();
        out_.variable_lower_bounds[vid] = v;
        out_.variable_upper_bounds[vid] = v;
      } else {
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "LP parse error at line %d: expected 'free', '<=', '>=' or '=' "
                           "after variable name in Bounds section, got '%s'",
                           peek().line,
                           peek().text.c_str());
      }
    } else {
      // lb <= x [<= ub]
      f_t lb = parse_signed_number();
      expect(LpTokenKind::LessEq, "'<=' in 'lb <= var' bound");
      mps_parser_expects(peek().kind == LpTokenKind::Name && !is_infinity_keyword(peek()),
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected variable name after 'lb <='",
                         peek().line);
      std::string var_name = peek().text;
      advance();
      i_t vid = get_or_add_var(var_name);
      bounds_defined_for_var_id_.insert(vid);
      out_.variable_lower_bounds[vid] = lb;
      if (match(LpTokenKind::LessEq)) { out_.variable_upper_bounds[vid] = parse_signed_number(); }
    }
  }
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_integer_list_section(bool is_binary)
{
  while (!at_section_boundary()) {
    mps_parser_expects(peek().kind == LpTokenKind::Name,
                       error_type_t::ValidationError,
                       "LP parse error at line %d: expected variable name in %s section, got '%s'",
                       peek().line,
                       is_binary ? "Binaries" : "Generals",
                       peek().text.c_str());
    std::string var_name = peek().text;
    advance();
    i_t vid             = get_or_add_var(var_name);
    out_.var_types[vid] = 'I';
    if (is_binary) {
      out_.variable_lower_bounds[vid] = f_t(0);
      out_.variable_upper_bounds[vid] = f_t(1);
      bounds_defined_for_var_id_.insert(vid);
    }
  }
}

// ---- Top-level dispatch ----------------------------------------------------

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_all()
{
  bool saw_objective = false;
  bool saw_end       = false;

  while (!at_eof()) {
    SectionKind kind = try_consume_section_header();
    switch (kind) {
      case SectionKind::Objective:
        mps_parser_expects(!saw_objective,
                           error_type_t::ValidationError,
                           "LP parse error: multiple objective sections");
        parse_objective_section();
        saw_objective = true;
        break;
      case SectionKind::Constraints:
      case SectionKind::LazyConstraints: parse_constraints_section(); break;
      case SectionKind::Bounds: parse_bounds_section(); break;
      case SectionKind::Generals: parse_integer_list_section(false); break;
      case SectionKind::Binaries: parse_integer_list_section(true); break;
      case SectionKind::End:
        saw_end = true;
        break;  // Break out of the switch; the check below ends parsing.
      case SectionKind::None: break;
    }
    if (saw_end) break;  // Anything after 'End' is ignored.
  }
  if (!saw_end) { printf("LP parser: 'End' section is missing\n"); }
  mps_parser_expects(saw_objective,
                     error_type_t::ValidationError,
                     "LP parse error: no objective (Minimize/Maximize) section found");
}

}  // namespace

// ===========================================================================
// lp_parser_t — thin public wrapper. All parsing state/types live in the
// anonymous namespace above.
// ===========================================================================

template <typename i_t, typename f_t>
lp_parser_t<i_t, f_t>::lp_parser_t(mps_data_model_t<i_t, f_t>& problem, const std::string& file)
{
  LpParseEngine<i_t, f_t> engine(*this, file);
  detail::finalize_problem(problem, *this);
}

template class lp_parser_t<int, float>;
template class lp_parser_t<int, double>;

// ===========================================================================
// Public parse_lp()
// ===========================================================================

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_lp(const std::string& lp_file_path)
{
  mps_data_model_t<i_t, f_t> problem;
  lp_parser_t<i_t, f_t> parser(problem, lp_file_path);
  return problem;
}

template mps_data_model_t<int, float> parse_lp<int, float>(const std::string&);
template mps_data_model_t<int, double> parse_lp<int, double>(const std::string&);

}  // namespace cuopt::mps_parser
