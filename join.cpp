#include <getopt.h>
#include <sycl/sycl.hpp>

#include "exchange.hpp"
#include "utils/generator.h"

sycl::event build(int *dim_key, int *dim_val, int num_tuples, int *hash_table,
                  int num_slots, sycl::queue &q) {
  return q.parallel_for(num_tuples, [=](sycl::id<1> idx) {
    build_keys_vals_element(dim_key[idx], dim_val[idx], true, num_tuples,
                            hash_table, num_slots, 0);
  });
}

sycl::event probe(int *fact_fkey, int *fact_val, int num_tuples,
                  int *hash_table, int num_slots, unsigned long long *res,
                  sycl::queue &q) {
  return q.parallel_for(num_tuples, sycl::reduction(res, sycl::plus<>()),
                        [=](sycl::id<1> idx, auto &sum) {
                          int join_val = 0;
                          bool sf = true;
                          probe_keys_vals_element(fact_fkey[idx], join_val, sf,
                                                  num_tuples, hash_table,
                                                  num_slots, 0);
                          if (sf)
                            sum.combine(fact_val[idx] * join_val);
                        });
}

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

  sycl::queue cpu_queue{
      sycl::cpu_selector_v,
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
    event = build(dim_key, dim_val, build_tables[0].num_tuples, hash_table,
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
  prob.probe_function = [&](int **probe_data, int partition_len,
                            int **hash_tables, unsigned long long *res,
                            sycl::queue queue, sycl::event &event) {
    event = probe(probe_data[0], probe_data[1], partition_len, hash_tables[0],
                  build_tables[0].num_slots, res, queue);
  };

  create_relation_fk(prob.h_lo_data[0], prob.h_lo_data[1], num_fact, num_dim,
                     cpu_queue);

  cout << "Query: join" << endl;
  exchange_operator_wrapper<int, unsigned long long>(
      build_tables, 1, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}
