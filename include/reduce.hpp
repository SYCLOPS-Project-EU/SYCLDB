#include <sycl/sycl.hpp>

#pragma once

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline T BlockSum(T item, T *shared, const sycl::nd_item<1> &item_ct1) {
  item_ct1.barrier(sycl::access::fence_space::local_space);

  T val = item;
  const int warp_size = 32;
  const int lid = item_ct1.get_local_id(0);
  int lane = lid % warp_size;
  int wid = lid / warp_size;

  auto sg = item_ct1.get_sub_group();

// Calculate sum across warp
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += sycl::shift_group_left(sg, val, offset);
  }

  // Store sum in buffer
  if (lane == 0) {
    shared[wid] = val;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Load the sums into the first warp
  val = (lid < item_ct1.get_local_range(0) / warp_size) ? shared[lane] : 0;

  // Calculate sum of sums
  if (wid == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      val += sycl::shift_group_left(sg, val, offset);
    }
  }

  return val;
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline T BlockSum(T (&items)[ITEMS_PER_THREAD], T *shared,
                  const sycl::nd_item<1> &item_ct1) {
  T thread_sum = 0;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    thread_sum += items[ITEM];
  }

  return BlockSum(thread_sum, shared, item_ct1);
}
