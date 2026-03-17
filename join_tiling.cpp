#include <getopt.h>
#include <sycl/sycl.hpp>

#include "exchange.hpp"
#include "utils/generator.h"
#include "include/join_kernels.hpp"

using namespace std;

int main(int argc, char **argv) {
  int num_fact = 64 * 4 << 20; // probe table size
  int num_dim = 16 * 4 << 20;  // build table size
  int num_gpus = 1;
  int num_partitions = 1;
  int repetitions = 10;

  int c;
  while ((c = getopt(argc, argv, "f:d:g:p:r:")) != -1) {
    switch (c) {
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

  // Use default selector for data generation, typically hits the CPU or integrated GPU
  sycl::queue cpu_queue{
      sycl::default_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling()}};

  BuildData<int> build_tables[1];
  build_tables[0].h_filter_col = NULL;
  build_tables[0].h_dim_key = NULL;
  build_tables[0].h_dim_val = NULL;
  build_tables[0].num_tuples = num_dim;
  build_tables[0].num_slots = num_dim;

  // Capture only specific PODs to avoid capturing BuildData/std::function wrapper
  const int build_tuples = build_tables[0].num_tuples;
  const int build_slots = build_tables[0].num_slots;

  build_tables[0].build_function = [build_tuples, build_slots](int *filter, int *dim_key, int *dim_val,
                                       int *hash_table, sycl::queue queue,
                                       sycl::event &event) {
    const int num_entries = build_tuples;
    const int num_slots = build_slots;
    const int num_blocks = (num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
    
    bool *effective_filter = (bool*)filter;
    sycl::event prep_event;
    if (filter == NULL) {
        effective_filter = sycl::malloc_device<bool>(num_entries, queue);
        prep_event = queue.submit([num_blocks, effective_filter, num_entries](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_blocks * N_BLOCK_THREADS, N_BLOCK_THREADS), [effective_filter, num_entries](sycl::nd_item<1> item) {
                set_selection_flags_kernel<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(item, effective_filter, num_entries);
            });
        });
    } else {
        prep_event = queue.submit([num_blocks, effective_filter, num_entries](sycl::handler &cgh) {
             cgh.parallel_for(sycl::nd_range<1>(num_blocks * N_BLOCK_THREADS, N_BLOCK_THREADS), [effective_filter, num_entries](sycl::nd_item<1> item) {
                set_selection_flags_kernel<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(item, effective_filter, num_entries);
            });
        });
    }

    event = queue.submit([prep_event, num_blocks, dim_key, dim_val, effective_filter, num_entries, hash_table, num_slots](sycl::handler &cgh) {
      cgh.depends_on(prep_event);
      cgh.parallel_for(sycl::nd_range<1>(num_blocks * N_BLOCK_THREADS, N_BLOCK_THREADS), [dim_key, dim_val, effective_filter, num_entries, hash_table, num_slots](sycl::nd_item<1> item) {
        build_keys_vals_kernel<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
            dim_key, dim_val, effective_filter, num_entries, hash_table, num_slots, 0, item);
      });
    });

    if (filter == NULL) {
        event.wait();
        sycl::free(effective_filter, queue);
    }
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

  // Capture specific slot count for probe
  const int probe_slots = build_tables[0].num_slots;

  prob.probe_function = [probe_slots](int **probe_data, int partition_len,
                             int **hash_tables, unsigned long long *res,
                             sycl::queue queue, sycl::event &event) {
    const int num_blocks = (partition_len + TILE_ITEMS - 1) / TILE_ITEMS;
    const int num_slots = probe_slots;
    
    bool *sf = sycl::malloc_device<bool>(partition_len, queue);
    
    auto prep_event = queue.submit([num_blocks, sf, partition_len](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_blocks * N_BLOCK_THREADS, N_BLOCK_THREADS), [sf, partition_len](sycl::nd_item<1> item) {
            set_selection_flags_kernel<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(item, sf, partition_len);
        });
    });

    int *join_vals = sycl::malloc_device<int>(partition_len, queue);
    int *hash_table = hash_tables[0];
    int *probe_keys = probe_data[0];
    int *probe_vals = probe_data[1];

    auto probe_event = queue.submit([prep_event, num_blocks, probe_keys, join_vals, sf, partition_len, hash_table, num_slots](sycl::handler &cgh) {
      cgh.depends_on(prep_event);
      cgh.parallel_for(sycl::nd_range<1>(num_blocks * N_BLOCK_THREADS, N_BLOCK_THREADS), [probe_keys, join_vals, sf, partition_len, hash_table, num_slots](sycl::nd_item<1> item) {
        probe_keys_vals_kernel<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
            probe_keys, join_vals, sf, partition_len, hash_table, num_slots, 0, item);
      });
    });

    event = queue.submit([probe_event, num_blocks, join_vals, probe_vals, sf, partition_len, res](sycl::handler &cgh) {
      cgh.depends_on(probe_event);
      cgh.parallel_for(sycl::nd_range<1>(num_blocks * N_BLOCK_THREADS, N_BLOCK_THREADS), [join_vals, probe_vals, sf, partition_len, res](sycl::nd_item<1> item) {
          sum_kernel<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
              join_vals, probe_vals, sf, partition_len, res, kOpAsterisk, item);
      });
    });

    event.wait();
    sycl::free(join_vals, queue);
    sycl::free(sf, queue);
  };

  create_relation_fk(prob.h_lo_data[0], prob.h_lo_data[1], num_fact, num_dim,
                     cpu_queue);

  cout << "Query: join_tiling" << endl;
  exchange_operator_wrapper<int, unsigned long long>(
      build_tables, 1, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}
