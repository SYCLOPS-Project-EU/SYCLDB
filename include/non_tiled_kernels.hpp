#include <sycl/sycl.hpp>

template <typename T>
inline T HASH (T X, T Y, T Z) {
  return ((X - Z) % Y);
}

// enumeration class taken from:
// https://github.com/hyrise/sql-parser/blob/master/src/sql/Expr.h#L37
enum OperatorType {
  kOpNone,

  // Ternary operator
  kOpBetween,

  // n-nary special case
  kOpCase,
  kOpCaseListElement,  // `WHEN expr THEN expr`

  // Binary operators.
  kOpPlus,
  kOpMinus,
  kOpAsterisk,
  kOpSlash,
  kOpPercentage,
  kOpCaret,

  kOpEquals,
  kOpNotEquals,
  kOpLess,
  kOpLessEq,
  kOpGreater,
  kOpGreaterEq,
  kOpLike,
  kOpNotLike,
  kOpILike,
  kOpAnd,
  kOpOr,
  kOpIn,
  kOpConcat,

  // Unary operators.
  kOpNot,
  kOpUnaryMinus,
  kOpIsNull,
  kOpExists
};

enum comp_op { EQ, NE, LT, LE, GT, GE };
enum logical_op { NONE, AND, OR };

template <typename T> inline bool compare(comp_op CO, T a, T b) {
  switch (CO) {
    case EQ:
      return a == b;
    case NE:
      return a != b;
    case LT:
      return a < b;
    case LE:
      return a <= b;
    case GT:
      return a > b;
    case GE:
      return a >= b;
    default:
      return false;
  }
}

// takes the result of the comparison
inline bool logical(logical_op LO, bool a, bool b) {
  switch (LO) {
    case NONE:
      return b;
    case AND:
      return a && b;
    case OR:
      return a || b;
    default:
      return false;
  }
}

constexpr comp_op get_comp_op(OperatorType op) {
  switch (op) {
  case kOpEquals:
    return EQ;
  case kOpLess:
    return LT;
  case kOpLessEq:
    return LE;
  case kOpGreater:
    return GT;
  case kOpGreaterEq:
    return GE;
  default:
    return EQ;
  }
}

constexpr logical_op get_logical_op(OperatorType op) {
  switch (op) {
  case kOpNone:
    return NONE;
  case kOpAnd:
    return AND;
  case kOpOr:
    return OR;
  default:
    return NONE;
  }
}

template <typename T>
sycl::event selection_kernel(sycl::queue& q, T *col, bool* sf,
                             int col_len, OperatorType ker_parent_op,
                             OperatorType ker_expr_op, int ker_ref_value) {
  auto logic = get_logical_op(ker_parent_op);
  auto comp = get_comp_op(ker_expr_op);
  return q.parallel_for(col_len, [=](sycl::id<1> idx) {
    sf[idx] = logical(logic, sf[idx], compare<int>(comp, col[idx], ker_ref_value));
  });
}

template <typename T>
sycl::event selection_kernel(sycl::queue& q, sycl::buffer<T> col_b, sycl::buffer<bool> sf_b,
                             int col_len, OperatorType ker_parent_op,
                             OperatorType ker_expr_op, int ker_ref_value) {
  auto logic = get_logical_op(ker_parent_op);
  auto comp = get_comp_op(ker_expr_op);
  return q.submit([&](sycl::handler &cgh) {
    sycl::accessor sf(sf_b, cgh, sycl::read_write);
    sycl::accessor col(col_b, cgh, sycl::read_only);
    cgh.parallel_for(col_len, [=](sycl::id<1> idx) {
      sf[idx] = logical(logic, sf[idx], compare<int>(comp, col[idx], ker_ref_value));
    });
  });
}

sycl::event build_keys_only(sycl::queue& q, int *keys, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    if (sf[idx]) {
      ht[HASH<uint64_t>(keys[idx], ht_len, key_min)] = 1;
    }
  });
}

sycl::event build_keys_only(sycl::queue& q, sycl::buffer<int> keys_b, sycl::buffer<bool> sf_b,
                            int num_entries, sycl::buffer<int> ht_b, int ht_len, int key_min) {
  return q.submit([&](sycl::handler &cgh) {
    sycl::accessor ht(ht_b, cgh, sycl::write_only);
    sycl::accessor sf(sf_b, cgh, sycl::read_only);
    sycl::accessor keys(keys_b, cgh, sycl::read_only);
    cgh.parallel_for(num_entries, [=](sycl::id<1> idx) {
      if (sf[idx]) {
        ht[HASH<uint64_t>(keys[idx], ht_len, key_min)] = 1;
      }
    });
  });
}

sycl::event build_keys_vals(sycl::queue& q, int *keys, int *vals, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    if (sf[idx]) {
      int hash = HASH<uint64_t>(keys[idx], ht_len, key_min);
      ht[hash << 1] = 1;
      ht[(hash << 1) + 1] = vals[idx];
    }
  });
}

sycl::event build_keys_vals(sycl::queue& q, sycl::buffer<int> keys_b, sycl::buffer<int> vals_b, sycl::buffer<bool> sf_b,
                            int num_entries, sycl::buffer<int> ht_b, int ht_len, int key_min) {
  return q.submit([&](sycl::handler &cgh) {
    sycl::accessor ht(ht_b, cgh, sycl::write_only);
    sycl::accessor sf(sf_b, cgh, sycl::read_only);
    sycl::accessor keys(keys_b, cgh, sycl::read_only);
    sycl::accessor vals(vals_b, cgh, sycl::read_only);
    cgh.parallel_for(num_entries, [=](sycl::id<1> idx) {
      if (sf[idx]) {
        int hash = HASH<uint64_t>(keys[idx], ht_len, key_min);
        ht[hash << 1] = 1;
        ht[(hash << 1) + 1] = vals[idx];
      }
    });
  });
}

sycl::event probe_keys_only(sycl::queue& q, int *keys, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    if (sf[idx]) {
      sf[idx] = ht[HASH<uint64_t>(keys[idx], ht_len, key_min)];
    }
  });
}

sycl::event probe_keys_only(sycl::queue& q, sycl::buffer<int> keys_b, sycl::buffer<bool> sf_b,
                            int num_entries, sycl::buffer<int> ht_b, int ht_len, int key_min) {
  return q.submit([&](sycl::handler &cgh) {
    sycl::accessor ht(ht_b, cgh, sycl::read_only);
    sycl::accessor sf(sf_b, cgh, sycl::read_write);
    sycl::accessor keys(keys_b, cgh, sycl::read_only);
    cgh.parallel_for(num_entries, [=](sycl::id<1> idx) {
      if (sf[idx]) {
        sf[idx] = ht[HASH<uint64_t>(keys[idx], ht_len, key_min)];
      }
    });
  });
}

sycl::event probe_keys_vals(sycl::queue& q, int *keys, int *join_vals, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    if (sf[idx]) {
      int hash = HASH<uint64_t>(keys[idx], ht_len, key_min);
      uint64_t slot = *reinterpret_cast<uint64_t *>(&ht[hash << 1]);
      if (slot) {
        join_vals[idx] = ht[(hash << 1) + 1];
      }
      else {
        sf[idx] = false;
      }
    }
  });
}

sycl::event probe_keys_vals(sycl::queue& q, sycl::buffer<int> keys_b, sycl::buffer<int> join_vals_b, sycl::buffer<bool> sf_b,
                            int num_entries, sycl::buffer<int> ht_b, int ht_len, int key_min) {
  return q.submit([&](sycl::handler &cgh) {
    sycl::accessor ht(ht_b, cgh, sycl::read_only);
    sycl::accessor sf(sf_b, cgh, sycl::read_write);
    sycl::accessor keys(keys_b, cgh, sycl::read_only);
    sycl::accessor join_vals(join_vals_b, cgh, sycl::write_only);
    cgh.parallel_for(num_entries, [=](sycl::id<1> idx) {
      if (sf[idx]) {
        int hash = HASH<uint64_t>(keys[idx], ht_len, key_min);
        uint64_t slot = *reinterpret_cast<const uint64_t *>(&ht[hash << 1]);
        if (slot) {
          join_vals[idx] = ht[(hash << 1) + 1];
        }
        else {
          sf[idx] = false;
        }
      }
    });
  });
}

template <typename T>
sycl::event sum_kernel(sycl::queue& q, T *col1, T *col2, bool* sf, int num_entries,
                       OperatorType op, unsigned long long *res) {
  return q.parallel_for(num_entries, sycl::reduction(res, sycl::plus<>()), [=](sycl::id<1> idx, auto &sum) {
    if (sf[idx]) {
      switch (op) {
      case kOpPlus:
        sum.combine(col1[idx] + col2[idx]);
        break;
      case kOpMinus:
        sum.combine(col1[idx] - col2[idx]);
        break;
      case kOpAsterisk:
        sum.combine(col1[idx] * col2[idx]);
        break;
      case kOpSlash:
        sum.combine(col1[idx] / col2[idx]);
        break;
      case kOpNone:
        sum.combine(col1[idx]);
        break;
      default:
        break;
      }
    }
  });
}

template <typename T>
sycl::event sum_kernel(sycl::queue& q, sycl::buffer<T> col1_b, sycl::buffer<T> col2_b, sycl::buffer<bool> sf_b,
                       int num_entries, OperatorType op, unsigned long long *res) {
  return q.submit([&](sycl::handler &cgh) {
    sycl::accessor sf(sf_b, cgh, sycl::read_only);
    sycl::accessor col1(col1_b, cgh, sycl::read_only);
    sycl::accessor col2(col2_b, cgh, sycl::read_only);
    cgh.parallel_for(num_entries, sycl::reduction(res, sycl::plus<>()), [=](sycl::id<1> idx, auto &sum) {
      if (sf[idx]) {
        switch (op) {
        case kOpPlus:
          sum.combine(col1[idx] + col2[idx]);
          break;
        case kOpMinus:
          sum.combine(col1[idx] - col2[idx]);
          break;
        case kOpAsterisk:
          sum.combine(col1[idx] * col2[idx]);
          break;
        case kOpSlash:
          sum.combine(col1[idx] / col2[idx]);
          break;
        case kOpNone:
          sum.combine(col1[idx]);
          break;
        default:
          break;
        }
      }
    });
  });
}
