#include <sycl/sycl.hpp>

#pragma once

#define HASH(X, Y, Z) ((X - Z) % Y)

// https://github.com/zjin-lcf/HeCBench/blob/master/src/graphB%2B-sycl/kernels.h#L4-L13
template <typename K> inline K atomicCAS(K *val, K expected, K desired) {
  K expected_value = expected;
  auto atm = sycl::atomic_ref<K, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(*val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_1(int tid, K (&items)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, K keys_min) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(items[ITEM], ht_len, keys_min);

      K slot = ht[hash];
      if (slot != 0) {
        selection_flags[ITEM] = 1;
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_1(int tid, K (&items)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, K keys_min,
                                     int num_items) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        K slot = ht[hash];
        if (slot != 0) {
          selection_flags[ITEM] = 1;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeAndPHT_1(K (&items)[ITEMS_PER_THREAD],
                               int (&selection_flags)[ITEMS_PER_THREAD], K *ht,
                               int ht_len, K keys_min, int num_items,
                               const sycl::nd_item<1> &item_ct1) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), items, selection_flags, ht, ht_len, keys_min,
        num_items);
  }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeAndPHT_1(K (&items)[ITEMS_PER_THREAD],
                               int (&selection_flags)[ITEMS_PER_THREAD], K *ht,
                               int ht_len, int num_items,
                               const sycl::nd_item<1> &item_ct1) {
  BlockProbeAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, selection_flags, ht, ht_len, 0, item_ct1, num_items);
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_2(int tid, K (&keys)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, K keys_min) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      uint64_t slot = *reinterpret_cast<uint64_t *>(&ht[hash << 1]);
      if (slot != 0) {
        res[ITEM] = (slot >> 32);
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_2(int tid, K (&items)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&selection_flags)[ITEMS_PER_THREAD],
                                     K *ht, int ht_len, K keys_min,
                                     int num_items) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        uint64_t slot = *reinterpret_cast<uint64_t *>(&ht[hash << 1]);
        if (slot != 0) {
          res[ITEM] = (slot >> 32);
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeAndPHT_2(K (&keys)[ITEMS_PER_THREAD],
                               V (&res)[ITEMS_PER_THREAD],
                               int (&selection_flags)[ITEMS_PER_THREAD], K *ht,
                               int ht_len, K keys_min, int num_items,
                               const sycl::nd_item<1> &item_ct1) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
        keys_min);
  } else {
    BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
        keys_min, num_items);
  }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void
BlockProbeAndPHT_2(K (&keys)[ITEMS_PER_THREAD], V (&res)[ITEMS_PER_THREAD],
                   int (&selection_flags)[ITEMS_PER_THREAD], K *ht, int ht_len,
                   int num_items, const sycl::nd_item<1> &item_ct1) {
  BlockProbeAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
      keys, res, selection_flags, ht, ht_len, 0, item_ct1, num_items);
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
      // K old = dpct::atomic_compare_exchange_strong<
      //     sycl::access::address_space::generic_space>(&ht[hash], 0,
      //     keys[ITEM]);
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
        // K old = dpct::atomic_compare_exchange_strong<
        //     sycl::access::address_space::generic_space>(&ht[hash], 0,
        //                                                 items[ITEM]);
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
      // K old = dpct::atomic_compare_exchange_strong<
      //     sycl::access::address_space::generic_space>(&ht[hash << 1], 0,
      //                                                 keys[ITEM]);
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
        // K old = dpct::atomic_compare_exchange_strong<
        //     sycl::access::address_space::generic_space>(&ht[hash << 1], 0,
        //                                                 keys[ITEM]);
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
