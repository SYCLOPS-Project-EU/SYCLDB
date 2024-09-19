#include <getopt.h>
#include <sycl/sycl.hpp>

#include "exchange.hpp"
#include "utils/generator.h"

#ifndef N_BLOCK_THREADS
#define N_BLOCK_THREADS 128
#endif
#ifndef N_ITEMS_PER_THREAD
#define N_ITEMS_PER_THREAD 4
#endif

#define TILE_ITEMS (N_BLOCK_THREADS * N_ITEMS_PER_THREAD)

using namespace std;

//---------------------------------------------------------------------
// Implements Join Operator
// There are two variants: tiling and non-tiling
//---------------------------------------------------------------------

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
sycl::event build_tiling(int *dim_key, int *dim_val, int num_tuples,
                         int *hash_table, int num_slots, sycl::queue &q) {
  sycl::range<1> gws((num_tuples + TILE_ITEMS - 1) / TILE_ITEMS *
                     N_BLOCK_THREADS);
  sycl::range<1> lws(N_BLOCK_THREADS);
  return q.parallel_for(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item_ct1) {
        int items[ITEMS_PER_THREAD];
        int items2[ITEMS_PER_THREAD];
        int selection_flags[ITEMS_PER_THREAD];

        int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
        int num_tiles = (num_tuples + TILE_ITEMS - 1) / TILE_ITEMS;
        int num_tile_items = TILE_ITEMS;

        if (item_ct1.get_group(0) == num_tiles - 1) {
          num_tile_items = num_tuples - tile_offset;
        }

        InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
        BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            dim_key + tile_offset, items, num_tile_items, item_ct1);
        BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            dim_val + tile_offset, items2, num_tile_items, item_ct1);
        BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, items2, selection_flags, hash_table, num_slots, 0,
            num_tile_items, item_ct1);
      });
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
sycl::event probe_tiling(int *fact_fkey, int *fact_val, int num_tuples,
                         int *hash_table, int num_slots,
                         unsigned long long *res, sycl::queue &q) {
  sycl::range<1> gws((num_tuples + TILE_ITEMS - 1) / TILE_ITEMS *
                     N_BLOCK_THREADS);
  sycl::range<1> lws(N_BLOCK_THREADS);
  return q.parallel_for(
      sycl::nd_range<1>(gws, lws), sycl::reduction(res, sycl::plus<>()),
      [=](sycl::nd_item<1> item_ct1, auto &sum) {
        // Load a tile striped across threads
        int selection_flags[ITEMS_PER_THREAD];
        int keys[ITEMS_PER_THREAD];
        int vals[ITEMS_PER_THREAD];
        int join_vals[ITEMS_PER_THREAD];

        int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
        int num_tiles = (num_tuples + TILE_ITEMS - 1) / TILE_ITEMS;
        int num_tile_items = TILE_ITEMS;

        if (item_ct1.get_group(0) == num_tiles - 1) {
          num_tile_items = num_tuples - tile_offset;
        }

        InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
        BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            fact_fkey + tile_offset, keys, num_tile_items, item_ct1);
        BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            fact_val + tile_offset, vals, num_tile_items, item_ct1);

        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            keys, join_vals, selection_flags, hash_table, num_slots, 0,
            num_tile_items, item_ct1);

        const auto lid = item_ct1.get_local_id(0);

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
          if ((lid + (BLOCK_THREADS * ITEM) < num_tile_items))
            if (selection_flags[ITEM])
              sum.combine(vals[ITEM] * join_vals[ITEM]);
        }
      });
}

sycl::event probe_nontiling(int *fact_fkey, int *fact_val, int num_tuples,
                            int *hash_table, int num_slots,
                            unsigned long long *res, sycl::queue &q) {
  return q.parallel_for(num_tuples, sycl::reduction(res, sycl::plus<>()),
                        [=](sycl::id<1> idx, auto &sum) {
                          long long hash = HASH(fact_fkey[idx], num_slots, 0);
                          uint64_t slot = *reinterpret_cast<uint64_t *>(
                              &hash_table[hash << 1]);
                          if (slot != 0) {
                            sum.combine(fact_val[idx] * (int)(slot >> 32));
                          }
                        });
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char **argv) {
  int num_fact = 256 * 4 << 20; // probe table size
  int num_dim = 16 * 4 << 20;   // build table size
  int num_gpus = 1;
  int num_partitions = 1;
  int repetitions = 10;
  int tiling = 0;

  int c;
  while ((c = getopt(argc, argv, "t:f:d:g:p:r:")) != -1) {
    switch (c) {
    case 't':
      tiling = atoi(optarg);
      break;
    case 'f':
      num_fact = atoi(optarg);
      break;
    case 'd':
      num_dim = atoi(optarg);
      break;
    case 'g':
      num_gpus = atoi(optarg);
      break;
    case 'p':
      num_partitions = atoi(optarg);
      break;
    case 'r':
      repetitions = atoi(optarg);
      break;
    default:
      abort();
    }
  }
  cout << "tiling: " << tiling << endl;

  sycl::queue cpu_queue{
      sycl::default_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling()}};

  BuildData<int> build_tables[1];
  build_tables[0].h_filter_col = NULL;
  build_tables[0].h_dim_key = NULL;
  build_tables[0].h_dim_val = NULL;
  build_tables[0].num_tuples = num_dim;
  build_tables[0].num_slots = num_dim;
  build_tables[0].build_function = [&](int *, int *dim_key, int *dim_val,
                                       int *hash_table, sycl::queue queue,
                                       sycl::event &event) {
    event = build_tiling<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
        dim_key, dim_val, build_tables[0].num_tuples, hash_table,
        build_tables[0].num_slots, queue);
  };

  create_relation_pk(build_tables[0].h_dim_key, build_tables[0].h_dim_val,
                     num_dim, cpu_queue);

  ProbeData<int, unsigned long long> prob;
  prob.n_probes = 2;
  prob.h_lo_data = new int *[prob.n_probes];
  prob.h_lo_data[0] = NULL;
  prob.h_lo_data[1] = NULL;
  prob.len_each_probe = num_fact;
  prob.res_size = 1;
  prob.res_array_cols = 1;
  prob.res_idx = 0;
  if (!tiling) {
    prob.probe_function = [&](int **probe_data, int partition_len,
                              int **hash_tables, unsigned long long *res,
                              sycl::queue queue, sycl::event &event) {
      event = probe_nontiling(probe_data[0], probe_data[1], partition_len,
                              hash_tables[0], build_tables[0].num_slots, res,
                              queue);
    };
  } else {
    prob.probe_function = [&](int **probe_data, int partition_len,
                              int **hash_tables, unsigned long long *res,
                              sycl::queue queue, sycl::event &event) {
      event = probe_tiling<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
          probe_data[0], probe_data[1], partition_len, hash_tables[0],
          build_tables[0].num_slots, res, queue);
    };
  }

  create_relation_fk(prob.h_lo_data[0], prob.h_lo_data[1], num_fact, num_dim,
                     cpu_queue);

  cout << "Query: join" << endl;
  exchange_operator_wrapper<int, unsigned long long>(
      build_tables, 1, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}
