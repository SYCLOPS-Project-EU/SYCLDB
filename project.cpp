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
// Implements Projection Operator
// There are two variants: tiling and non-tiling
//---------------------------------------------------------------------

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void project(float *in1, float *in2, float *out, int num_items,
             const sycl::nd_item<1> &item_ct1) {
  float items[ITEMS_PER_THREAD];
  float items2[ITEMS_PER_THREAD];
  float res[ITEMS_PER_THREAD];

  int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
  int num_tiles = (num_items + TILE_ITEMS - 1) / TILE_ITEMS;
  int num_tile_items = TILE_ITEMS;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_items - tile_offset;
  }

  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in1 + tile_offset, items,
                                                    num_tile_items, item_ct1);
  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in2 + tile_offset, items2,
                                                    num_tile_items, item_ct1);
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (item_ct1.get_local_id(0) + (ITEM * BLOCK_THREADS) < num_tile_items) {
      res[ITEM] = 2 * items[ITEM] + 3 * items2[ITEM];
    }
  }

  BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>(out + tile_offset, res,
                                                     num_tile_items, item_ct1);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
sycl::event project_tiling(float *in1, float *in2, float *out, int num_items,
                           sycl::queue &q) {
  sycl::range<1> gws((num_items + TILE_ITEMS - 1) / TILE_ITEMS *
                     N_BLOCK_THREADS);
  sycl::range<1> lws(N_BLOCK_THREADS);
  return q.parallel_for(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item_ct1) {
        float items[ITEMS_PER_THREAD];
        float items2[ITEMS_PER_THREAD];
        float res[ITEMS_PER_THREAD];

        int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
        int num_tiles = (num_items + TILE_ITEMS - 1) / TILE_ITEMS;
        int num_tile_items = TILE_ITEMS;

        if (item_ct1.get_group(0) == num_tiles - 1) {
          num_tile_items = num_items - tile_offset;
        }

        BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(
            in1 + tile_offset, items, num_tile_items, item_ct1);
        BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(
            in2 + tile_offset, items2, num_tile_items, item_ct1);
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
          if (item_ct1.get_local_id(0) + (ITEM * BLOCK_THREADS) <
              num_tile_items) {
            res[ITEM] = 2 * items[ITEM] + 3 * items2[ITEM];
          }
        }

        BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>(
            out + tile_offset, res, num_tile_items, item_ct1);
      });
}

sycl::event project_nontiling(float *in1, float *in2, float *out, int num_items,
                              sycl::queue &q) {
  return q.parallel_for(num_items, [=](sycl::id<1> idx) {
    out[idx] = 2 * in1[idx] + 3 * in2[idx];
  });
}

/**
 * Main
 */
int main(int argc, char **argv) {
  int num_items = 1 << 28;
  int tiling = 1;
  int num_gpus = 1;
  int num_partitions = 1;
  int repetitions = 10;
  int c;
  while ((c = getopt(argc, argv, "n:t:g:p:r:")) != -1) {
    switch (c) {
    case 'n':
      num_items = atoi(optarg);
      break;
    case 't':
      tiling = atoi(optarg);
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

  ProbeData<float, float> prob;
  prob.n_probes = 2;
  prob.h_lo_data = new float *[prob.n_probes];
  prob.h_lo_data[0] = NULL;
  prob.h_lo_data[1] = NULL;
  prob.len_each_probe = num_items;
  prob.res_size = num_items;
  prob.res_array_cols = 1;
  prob.res_idx = 0;
  if (!tiling) {
    prob.probe_function = [&](float **probe_data, int partition_len, float **,
                              float *res, sycl::queue queue,
                              sycl::event &event) {
      event = project_nontiling(probe_data[0], probe_data[1], res,
                                partition_len, queue);
    };
  } else {
    prob.probe_function = [&](float **probe_data, int partition_len, float **,
                              float *res, sycl::queue queue,
                              sycl::event &event) {
      sycl::range<1> gws((partition_len + TILE_ITEMS - 1) / TILE_ITEMS *
                         N_BLOCK_THREADS);
      sycl::range<1> lws(N_BLOCK_THREADS);
      event = project_tiling<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
          probe_data[0], probe_data[1], res, partition_len, queue);
    };
  }

  generateUniformCPU(prob.h_lo_data[0], prob.h_lo_data[1], num_items,
                     cpu_queue);

  cout << "Query: project" << endl;
  exchange_operator_wrapper<float, float>(
      NULL, 0, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}
