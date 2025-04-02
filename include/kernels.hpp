#pragma once

#include <sycl/sycl.hpp>

template <typename T = uint64_t>
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
void selection_element(bool &sf, T col, logical_op logic,
                       comp_op comp, T ker_ref_value) {
  sf = logical(logic, sf, compare<int>(comp, col, ker_ref_value));
}

template <typename T>
sycl::event selection_kernel(sycl::queue& q, T *col, bool* sf,
                             int col_len, OperatorType ker_parent_op,
                             OperatorType ker_expr_op, T ker_ref_value) {
  auto logic = get_logical_op(ker_parent_op);
  auto comp = get_comp_op(ker_expr_op);
  return q.parallel_for(col_len, [=](sycl::id<1> idx) {
    selection_element(sf[idx], col[idx], logic, comp, ker_ref_value);
  });
}

template <typename T>
sycl::event selection_kernel(sycl::queue& q, sycl::buffer<T> col_b, sycl::buffer<bool> sf_b,
                             int col_len, OperatorType ker_parent_op,
                             OperatorType ker_expr_op, T ker_ref_value) {
  auto logic = get_logical_op(ker_parent_op);
  auto comp = get_comp_op(ker_expr_op);
  return q.submit([&](sycl::handler &cgh) {
    sycl::accessor sf(sf_b, cgh, sycl::read_write);
    sycl::accessor col(col_b, cgh, sycl::read_only);
    cgh.parallel_for(col_len, [=](sycl::id<1> idx) {
      selection_element(sf[idx], col[idx], logic, comp, ker_ref_value);
    });
  });
}

void build_keys_element(int keys, bool sf, int num_entries,
                        int* ht, int ht_len, int key_min) {
  if (sf) {
    ht[HASH<uint64_t>(keys, ht_len, key_min)] = 1;
  }
}

sycl::event build_keys_only(sycl::queue& q, int *keys, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    build_keys_element(keys[idx], sf[idx], num_entries, ht, ht_len, key_min);
  });
}

void build_keys_vals_element(int keys, int vals, bool sf, int num_entries,
                             int* ht, int ht_len, int key_min) {
  if (sf) {
    int hash = HASH<uint64_t>(keys, ht_len, key_min);
    ht[hash << 1] = 1;
    ht[(hash << 1) + 1] = vals;
  }
}

sycl::event build_keys_vals(sycl::queue& q, int *keys, int *vals, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    build_keys_vals_element(keys[idx], vals[idx], sf[idx], num_entries, ht, ht_len, key_min);
  });
}

void probe_keys_element(int keys, bool& sf, int num_entries,
                        int* ht, int ht_len, int key_min) {
  if (sf) {
    sf = ht[HASH<uint64_t>(keys, ht_len, key_min)];
  }
}

sycl::event probe_keys_only(sycl::queue& q, int *keys, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    probe_keys_element(keys[idx], sf[idx], num_entries, ht, ht_len, key_min);
  });
}

void probe_keys_vals_element(int keys, int& join_vals, bool& sf, int num_entries,
                             int* ht, int ht_len, int key_min) {
  if (sf) {
    int hash = HASH<uint64_t>(keys, ht_len, key_min);
    uint64_t slot = *reinterpret_cast<uint64_t *>(&ht[hash << 1]);
    if (slot) {
      join_vals = (slot >> 32);
    }
    else {
      sf = false;
    }
  }
}

sycl::event probe_keys_vals(sycl::queue& q, int *keys, int *join_vals, bool* sf, int num_entries,
                            int *ht, int ht_len, int key_min) {
  return q.parallel_for(num_entries, [=](sycl::id<1> idx) {
    probe_keys_vals_element(keys[idx], join_vals[idx], sf[idx], num_entries, ht, ht_len, key_min);
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

// TODO: move to 1d kernel always
#ifndef N_BLOCK_THREADS
#define N_BLOCK_THREADS 128
#endif
#ifndef N_ITEMS_PER_THREAD
#define N_ITEMS_PER_THREAD 4
#endif

#define TILE_ITEMS (N_BLOCK_THREADS * N_ITEMS_PER_THREAD)

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoadDirect(const unsigned int tid, T *block_itr,
                            T (&items)[ITEMS_PER_THREAD]) {
  T *thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoadDirect(const unsigned int tid, T *block_itr,
                            T (&items)[ITEMS_PER_THREAD], int num_items) {
  T *thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoad(T *inp, T (&items)[ITEMS_PER_THREAD], int num_items,
                      const sycl::nd_item<1> &item_ct1) {
  T *block_itr = inp;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), block_itr, items);
  } else {
    BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), block_itr, items, num_items);
  }
}

template <typename K> inline K atomicCAS(K *val, K expected, K desired) {
  K expected_value = expected;
  auto atm = sycl::atomic_ref<K, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(*val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void
BlockBuildDirectSelectivePHT_1(int tid, K (&keys)[ITEMS_PER_THREAD],
                               int (&selection_flags)[ITEMS_PER_THREAD], K *ht,
                               int ht_len, K keys_min) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);
      atomicCAS<K>(&ht[hash], 0, keys[ITEM]);
    }
  }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void
BlockBuildDirectSelectivePHT_1(int tid, K (&items)[ITEMS_PER_THREAD],
                               int (&selection_flags)[ITEMS_PER_THREAD], K *ht,
                               int ht_len, K keys_min, int num_items) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);
        atomicCAS<K>(&ht[hash], 0, items[ITEM]);
      }
    }
  }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_1(K (&keys)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, K keys_min,
                                     int num_items,
                                     const sycl::nd_item<1> &item_ct1) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), keys, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), keys, selection_flags, ht, ht_len, keys_min,
        num_items);
  }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_1(K (&keys)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, int num_items,
                                     const sycl::nd_item<1> &item_ct1) {
  BlockBuildSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
      keys, selection_flags, ht, ht_len, 0, item_ct1, num_items);
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildDirectSelectivePHT_2(
    int tid, K (&keys)[ITEMS_PER_THREAD], V (&res)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD], K *ht, int ht_len, K keys_min) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);
      atomicCAS<K>(&ht[hash << 1], 0, keys[ITEM]);
      ht[(hash << 1) + 1] = res[ITEM];
    }
  }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void
BlockBuildDirectSelectivePHT_2(int tid, K (&keys)[ITEMS_PER_THREAD],
                               V (&res)[ITEMS_PER_THREAD],
                               int (&selection_flags)[ITEMS_PER_THREAD], K *ht,
                               int ht_len, K keys_min, int num_items) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(keys[ITEM], ht_len, keys_min);
        atomicCAS<K>(&ht[hash << 1], 0, keys[ITEM]);
        ht[(hash << 1) + 1] = res[ITEM];
      }
    }
  }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_2(K (&keys)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, K keys_min,
                                     int num_items,
                                     const sycl::nd_item<1> &item_ct1) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
        keys_min);
  } else {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
        keys_min, num_items);
  }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_2(K (&keys)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, int num_items,
                                     const sycl::nd_item<1> &item_ct1) {
  BlockBuildSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
      keys, res, selection_flags, ht, ht_len, 0, item_ct1, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void InitFlags(int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = 1;
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void BlockPredDirect(int tid, T (&items)[ITEMS_PER_THREAD],
                            SelectOp select_op,
                            int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = select_op(items[ITEM]);
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void
BlockPredDirect(int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op,
                int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      selection_flags[ITEM] = select_op(items[ITEM]);
    }
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void BlockPred(T (&items)[ITEMS_PER_THREAD], SelectOp select_op,
                      int (&selection_flags)[ITEMS_PER_THREAD], int num_items,
                      const sycl::nd_item<1> &item_ct1) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, select_op, selection_flags);
  } else {
    BlockPredDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, select_op, selection_flags, num_items);
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void BlockPredAndDirect(int tid, T (&items)[ITEMS_PER_THREAD],
                               SelectOp select_op,
                               int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void
BlockPredAndDirect(int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op,
                   int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
    }
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void BlockPredAnd(T (&items)[ITEMS_PER_THREAD], SelectOp select_op,
                         int (&selection_flags)[ITEMS_PER_THREAD],
                         int num_items, const sycl::nd_item<1> &item_ct1) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredAndDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, select_op, selection_flags);
  } else {
    BlockPredAndDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, select_op, selection_flags, num_items);
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void BlockPredOrDirect(int tid, T (&items)[ITEMS_PER_THREAD],
                              SelectOp select_op,
                              int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void
BlockPredOrDirect(int tid, T (&items)[ITEMS_PER_THREAD], SelectOp select_op,
                  int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
    }
  }
}

template <typename T, typename SelectOp, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
inline void BlockPredOr(T (&items)[ITEMS_PER_THREAD], SelectOp select_op,
                        int (&selection_flags)[ITEMS_PER_THREAD], int num_items,
                        const sycl::nd_item<1> &item_ct1) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, select_op, selection_flags);
  } else {
    BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, select_op, selection_flags, num_items);
  }
}

template <typename T> struct LessThan {
  T compare;

  inline LessThan(T compare) : compare(compare) {}

  inline bool operator()(const T &a) const { return (a < compare); }
};

template <typename T> struct GreaterThan {
  T compare;

  inline GreaterThan(T compare) : compare(compare) {}

  inline bool operator()(const T &a) const { return (a > compare); }
};

template <typename T> struct LessThanEq {
  T compare;

  inline LessThanEq(T compare) : compare(compare) {}

  inline bool operator()(const T &a) const { return (a <= compare); }
};

template <typename T> struct GreaterThanEq {
  T compare;

  inline GreaterThanEq(T compare) : compare(compare) {}

  inline bool operator()(const T &a) const { return (a >= compare); }
};

template <typename T> struct Eq {
  T compare;

  inline Eq(T compare) : compare(compare) {}

  inline bool operator()(const T &a) const { return (a == compare); }
};

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredLT(T (&items)[ITEMS_PER_THREAD], T compare,
                        int (&selection_flags)[ITEMS_PER_THREAD], int num_items,
                        const sycl::nd_item<1> &item_ct1) {
  LessThan<T> select_op(compare);
  BlockPred<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAndLT(T (&items)[ITEMS_PER_THREAD], T compare,
                           int (&selection_flags)[ITEMS_PER_THREAD],
                           int num_items, const sycl::nd_item<1> &item_ct1) {
  LessThan<T> select_op(compare);
  BlockPredAnd<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredGT(T (&items)[ITEMS_PER_THREAD], T compare,
                        int (&selection_flags)[ITEMS_PER_THREAD], int num_items,
                        const sycl::nd_item<1> &item_ct1) {
  GreaterThan<T> select_op(compare);
  BlockPred<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAndGT(T (&items)[ITEMS_PER_THREAD], T compare,
                           int (&selection_flags)[ITEMS_PER_THREAD],
                           int num_items, const sycl::nd_item<1> &item_ct1) {
  GreaterThan<T> select_op(compare);
  BlockPredAnd<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredLTE(T (&items)[ITEMS_PER_THREAD], T compare,
                         int (&selection_flags)[ITEMS_PER_THREAD],
                         int num_items, const sycl::nd_item<1> &item_ct1) {
  LessThanEq<T> select_op(compare);
  BlockPred<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAndLTE(T (&items)[ITEMS_PER_THREAD], T compare,
                            int (&selection_flags)[ITEMS_PER_THREAD],
                            int num_items, const sycl::nd_item<1> &item_ct1) {
  LessThanEq<T> select_op(compare);
  BlockPredAnd<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredGTE(T (&items)[ITEMS_PER_THREAD], T compare,
                         int (&selection_flags)[ITEMS_PER_THREAD],
                         int num_items, const sycl::nd_item<1> &item_ct1) {
  GreaterThanEq<T> select_op(compare);
  BlockPred<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAndGTE(T (&items)[ITEMS_PER_THREAD], T compare,
                            int (&selection_flags)[ITEMS_PER_THREAD],
                            int num_items, const sycl::nd_item<1> &item_ct1) {
  GreaterThanEq<T> select_op(compare);
  BlockPredAnd<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredEQ(T (&items)[ITEMS_PER_THREAD], T compare,
                        int (&selection_flags)[ITEMS_PER_THREAD], int num_items,
                        const sycl::nd_item<1> &item_ct1) {
  Eq<T> select_op(compare);
  BlockPred<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAndEQ(T (&items)[ITEMS_PER_THREAD], T compare,
                           int (&selection_flags)[ITEMS_PER_THREAD],
                           int num_items, const sycl::nd_item<1> &item_ct1) {
  Eq<T> select_op(compare);
  BlockPredAnd<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOrEQ(T (&items)[ITEMS_PER_THREAD], T compare,
                          int (&selection_flags)[ITEMS_PER_THREAD],
                          int num_items, const sycl::nd_item<1> &item_ct1) {
  Eq<T> select_op(compare);
  BlockPredOr<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, select_op, selection_flags, num_items, item_ct1);
}
