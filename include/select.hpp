#pragma once
#include "kernels.hpp"

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void block_select(comp_op CO, logical_op LO, int tid, T (&items)[ITEMS_PER_THREAD], bool (&sf)[ITEMS_PER_THREAD], T ref, int num_tile_items) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (tid + (i * BLOCK_THREADS) < num_tile_items) {
            sf[i] = logical(LO, sf[i], compare<T>(CO, items[i], ref));
        }
    }
}
