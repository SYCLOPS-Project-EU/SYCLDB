#pragma once
#include <sycl/sycl.hpp>

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void block_load(int tid, int tile_offset, T *data, T (&items)[ITEMS_PER_THREAD], int num_tile_items) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (tid + (i * BLOCK_THREADS) < num_tile_items) {
            items[i] = data[tile_offset + tid + (i * BLOCK_THREADS)];
        }
    }
}

// Specialization for accessors if needed, but we'll use pointers for now.
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void block_load(int tid, int tile_offset, const T *data, T (&items)[ITEMS_PER_THREAD], int num_tile_items) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (tid + (i * BLOCK_THREADS) < num_tile_items) {
            items[i] = data[tile_offset + tid + (i * BLOCK_THREADS)];
        }
    }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void block_unload(int tid, int tile_offset, T *data, T (&items)[ITEMS_PER_THREAD], int num_tile_items) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (tid + (i * BLOCK_THREADS) < num_tile_items) {
            data[tile_offset + tid + (i * BLOCK_THREADS)] = items[i];
        }
    }
}
