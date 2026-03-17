#pragma once

#define HASH(X, Y, Z) ((X - Z) % Y)

// omitting the rows which did not pass earlier selections
// simply shifts the column to now start with its minimum value
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void build_keys_only(int tid, K (&keys)[ITEMS_PER_THREAD],
                            bool (&sf)[ITEMS_PER_THREAD], K *ht, int ht_len,
                            K keys_min, int num_items) {
  // if not in the last block, no need to check for out of bounds
  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (sf[i]) {
        // atomicCAS<K>(&ht[HASH(keys[i], ht_len, keys_min)], 0, keys[i]);
        ht[HASH(keys[i], ht_len, keys_min)] = keys[i];
      }
    }
  } else {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (tid + (i * BLOCK_THREADS) < num_items) {
        if (sf[i]) {
          // atomicCAS<K>(&ht[HASH(keys[i], ht_len, keys_min)], 0, keys[i]);
          ht[HASH(keys[i], ht_len, keys_min)] = keys[i];
        }
      }
    }
  }
}

// omitting the rows which did not pass earlier selections
// first, shifts the key column to now start with its minimum value
// then store the keys in odd slots and the values in even slots
template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void build_keys_vals(int tid, K (&keys)[ITEMS_PER_THREAD],
                            V (&vals)[ITEMS_PER_THREAD],
                            bool (&sf)[ITEMS_PER_THREAD], K *ht, int ht_len,
                            K keys_min, int num_items) {
  // if not in the last block, no need to check for out of bounds
  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (sf[i]) {
        int hash = HASH(keys[i], ht_len, keys_min);
        // atomicCAS<K>(&ht[hash << 1], 0, keys[i]);
        ht[hash << 1] = keys[i];
        ht[(hash << 1) + 1] = vals[i];
      }
    }
  } else {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (tid + (i * BLOCK_THREADS) < num_items) {
        if (sf[i]) {
          int hash = HASH(keys[i], ht_len, keys_min);
          // atomicCAS<K>(&ht[hash << 1], 0, keys[i]);
          ht[hash << 1] = keys[i];
          ht[(hash << 1) + 1] = vals[i];
        }
      }
    }
  }
}

// update the selection flags of the probed table by build lookup
// a build with keys only is probed with probe keys only (<=>)
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void probe_keys_only(int tid, K (&keys)[ITEMS_PER_THREAD],
                            bool (&sf)[ITEMS_PER_THREAD], K *ht, int ht_len,
                            K keys_min, int num_items) {
  // if not in the last block, no need to check for out of bounds
  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (sf[i]) {
        sf[i] = ht[HASH(keys[i], ht_len, keys_min)];
      }
    }
  } else {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (tid + (i * BLOCK_THREADS) < num_items) {
        if (sf[i]) {
          sf[i] = ht[HASH(keys[i], ht_len, keys_min)];
        }
      }
    }
  }
}

// analogous update of selection flags for a probed table with keys and values
// temp_vals is now a newly created array to store the values of the build
template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void probe_keys_vals(int tid, K (&keys)[ITEMS_PER_THREAD],
                            V (&temp_vals)[ITEMS_PER_THREAD],
                            bool (&sf)[ITEMS_PER_THREAD], K *ht, int ht_len,
                            K keys_min, int num_items) {
  // if not in the last block, no need to check for out of bounds
  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (sf[i]) {
        int hash = HASH(keys[i], ht_len, keys_min);
        // obtain the key-value pair from a single lookup
        uint64_t slot = *reinterpret_cast<uint64_t *>(&ht[hash << 1]);
        if (slot != 0) {
          temp_vals[i] = (slot >> 32);
        } else {
          sf[i] = 0;
        }
      }
    }
  } else {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (tid + (i * BLOCK_THREADS) < num_items) {
        if (sf[i]) {
          int hash = HASH(keys[i], ht_len, keys_min);
          uint64_t slot = *reinterpret_cast<uint64_t *>(&ht[hash << 1]);
          if (slot != 0) {
            temp_vals[i] = (slot >> 32);
          } else {
            sf[i] = 0;
          }
        }
      }
    }
  }
}

// there might be cases, in which all values are selected (i.e. no selection)
// need to manually set the selection flags to true
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void set_selection_flags(int tid, bool (&sf)[ITEMS_PER_THREAD],
                                int num_items) {
  // if not in the last block, no need to check for out of bounds
  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
      sf[i] = true;
  } else {
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (tid + (i * BLOCK_THREADS) < num_items)
        sf[i] = true;
    }
  }
}
