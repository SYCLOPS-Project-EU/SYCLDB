#pragma once
#include "constants.hpp"
#include "select.hpp"
#include "join.hpp"
#include "load.hpp"

#include <sycl/sycl.hpp>

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void sum_kernel(int *col1, int *col2, bool *sf, int num_entries,
                unsigned long long *res, OperatorType op,
                const sycl::nd_item<1> &item) {

  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  bool selection_flags[ITEMS_PER_THREAD];

  const int tile_offset = item.get_group(0) * TILE_ITEMS;
  const int num_tiles = (num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
  const int num_tile_items = (item.get_group(0) == num_tiles - 1)
                                 ? num_entries - tile_offset
                                 : TILE_ITEMS;

  block_load<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);

  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, col1, items, num_tile_items);
  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, col2, items2, num_tile_items);

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (item.get_local_id(0) + (BLOCK_THREADS * i) < num_tile_items)
      if (selection_flags[i]) {
        auto sum_obj =
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(res[0]);
        switch (op) {
        case kOpPlus:
          sum_obj.fetch_add(items[i] + items2[i]);
          break;
        case kOpMinus:
          sum_obj.fetch_add(items[i] - items2[i]);
          break;
        case kOpAsterisk:
          sum_obj.fetch_add(items[i] * items2[i]);
          break;
        case kOpSlash:
          sum_obj.fetch_add(items[i] / items2[i]);
          break;
        case kOpNone:
          sum_obj.fetch_add(items[i]);
          break;
        default:
          break;
        }
      }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void selection_kernel(sycl::nd_item<1> item, int *col, bool *sf,
                      int col_len, OperatorType ker_parent_op,
                      OperatorType ker_expr_op, int ker_ref_value) {
  int items[ITEMS_PER_THREAD];
  bool selection_flags[ITEMS_PER_THREAD];

  const int tile_offset = item.get_group(0) * TILE_ITEMS;
  const int num_tiles = (col_len + TILE_ITEMS - 1) / TILE_ITEMS;
  const int num_tile_items =
      (item.get_group(0) == num_tiles - 1) ? col_len - tile_offset : TILE_ITEMS;

  block_load<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);

  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, col, items, num_tile_items);
  block_select<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      get_comp_op(ker_expr_op), get_logical_op(ker_parent_op),
      item.get_local_id(0), items, selection_flags, ker_ref_value,
      num_tile_items);

  block_unload<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void set_selection_flags_kernel(sycl::nd_item<1> item,
  bool *sf, int col_len) {
  bool selection_flags[ITEMS_PER_THREAD];

  const int tile_offset = item.get_group(0) * TILE_ITEMS;
  const int num_tiles = (col_len + TILE_ITEMS - 1) / TILE_ITEMS;
  const int num_tile_items =
      (item.get_group(0) == num_tiles - 1) ? col_len - tile_offset : TILE_ITEMS;

  block_load<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);

  set_selection_flags<BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), selection_flags, num_tile_items);

  block_unload<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void build_keys_only_kernel(int *keys, bool *sf, int num_entries,
                            int *ht, int ht_len, int key_min, const sycl::nd_item<1> &item) {

  int items[ITEMS_PER_THREAD];
  bool selection_flags[ITEMS_PER_THREAD];

  const int tile_offset = item.get_group(0) * TILE_ITEMS;
  const int num_tiles = (num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
  const int num_tile_items = (item.get_group(0) == num_tiles - 1)
                                 ? num_entries - tile_offset
                                 : TILE_ITEMS;

  block_load<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);

  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, keys, items, num_tile_items);

  build_keys_only<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), items, selection_flags, ht, ht_len, key_min, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void build_keys_vals_kernel(int *keys, int *vals, bool *sf, int num_entries,
                            int *ht, int ht_len, int key_min, const sycl::nd_item<1> &item) {

  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  bool selection_flags[ITEMS_PER_THREAD];

  const int tile_offset = item.get_group(0) * TILE_ITEMS;
  const int num_tiles = (num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
  const int num_tile_items = (item.get_group(0) == num_tiles - 1)
                                 ? num_entries - tile_offset
                                 : TILE_ITEMS;

  block_load<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);

  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, keys, items, num_tile_items);
  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, vals, items2, num_tile_items);

  build_keys_vals<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), items, items2, selection_flags, ht, ht_len, key_min, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void probe_keys_only_kernel(int *keys, bool *sf, int num_entries,
                            int *ht, int ht_len, int key_min, const sycl::nd_item<1> &item) {

  int items[ITEMS_PER_THREAD];
  bool selection_flags[ITEMS_PER_THREAD];

  const int tile_offset = item.get_group(0) * TILE_ITEMS;
  const int num_tiles = (num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
  const int num_tile_items = (item.get_group(0) == num_tiles - 1)
                                 ? num_entries - tile_offset
                                 : TILE_ITEMS;

  block_load<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);

  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, keys, items, num_tile_items);

  probe_keys_only<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), items, selection_flags, ht, ht_len, key_min, num_tile_items);

  block_unload<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void probe_keys_vals_kernel(int *keys, int *vals, bool *sf, int num_entries,
                            int *ht, int ht_len, int key_min, const sycl::nd_item<1> &item) {

  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  bool selection_flags[ITEMS_PER_THREAD];

  const int tile_offset = item.get_group(0) * TILE_ITEMS;
  const int num_tiles = (num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
  const int num_tile_items = (item.get_group(0) == num_tiles - 1)
                                 ? num_entries - tile_offset
                                 : TILE_ITEMS;

  block_load<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);

  block_load<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, keys, items, num_tile_items);

  probe_keys_vals<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), items, items2, selection_flags, ht, ht_len, key_min, num_tile_items);

  block_unload<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, vals, items2, num_tile_items);

  block_unload<bool, BLOCK_THREADS, ITEMS_PER_THREAD>(
      item.get_local_id(0), tile_offset, sf, selection_flags, num_tile_items);
}
