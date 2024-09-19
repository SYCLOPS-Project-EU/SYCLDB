#include <sycl/sycl.hpp>

#pragma once

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
