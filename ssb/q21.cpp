#include "../exchange.hpp"
#include "ssb_utils.hpp"

#include <getopt.h>
#include <sycl/sycl.hpp>

using namespace std;

void probe(int *lo_orderdate, int *lo_partkey, int *lo_suppkey, int *lo_revenue,
           int lo_len, int *ht_s, int s_len, int *ht_p, int p_len, int *ht_d,
           int d_len, int *res, sycl::id<1> idx) {
  int brand;
  int year;
  bool sf = true;
  probe_keys_vals_element(lo_partkey[idx], brand, sf, p_len, ht_p, p_len, 0);
  probe_keys_vals_element(lo_orderdate[idx], year, sf, d_len, ht_d, d_len,
                          19920101);
  probe_keys_element(lo_suppkey[idx], sf, s_len, ht_s, s_len, 0);
  if (sf) {
    int hash = (brand * 7 + (year - 1992)) % ((1998 - 1992 + 1) * (5 * 5 * 40));
    res[hash * 4] = year;
    res[hash * 4 + 1] = brand;
    auto sum_obj =
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::global_space>(
            *reinterpret_cast<unsigned long long *>(&res[hash * 4 + 2]));
    sum_obj.fetch_add((unsigned long long)(lo_revenue[idx]));
  }
}

void build_hashtable_s(int *filter_col, int *dim_key,
  int num_tuples, int *hash_table,
  int num_slots,
  sycl::id<1> idx) {
    bool sf = false;
    selection_element<int>(sf, filter_col[idx], NONE, EQ, 1);
    if (sf) {
      build_keys_element(dim_key[idx], sf, num_tuples, hash_table, num_slots, 0);
    }
}

void build_hashtable_p(int *filter_col, int *dim_key, int *dim_val,
                        int num_tuples, int *hash_table, int num_slots,
                        sycl::id<1> idx) {
  bool sf = false;
  selection_element<int>(sf, filter_col[idx], NONE, EQ, 1);
  if (sf) {
    build_keys_vals_element(dim_key[idx], dim_val[idx], sf, num_tuples,
                            hash_table, num_slots, 0);
  }
}

void build_hashtable_d(int *dim_key, int *dim_val, int num_tuples,
                        int *hash_table, int num_slots, int val_min,
                        sycl::id<1> idx) {
  build_keys_vals_element(dim_key[idx], dim_val[idx], true, num_tuples,
                          hash_table, num_slots, val_min);
}

int main(int argc, char **argv) {
  int num_partitions = 1;
  int num_gpus = 1;
  int repetitions = 10;
  int modes = 0;

  int c;
  while ((c = getopt(argc, argv, "p:g:r:m:")) != -1) {
    switch (c) {
    case 'p':
      num_partitions = atoi(optarg);
      break;
    case 'g':
      num_gpus = atoi(optarg);
      break;
    case 'r':
      repetitions = atoi(optarg);
      break;
    case 'm':
      modes = atoi(optarg);
      break;
    default:
      abort();
    }
  }

  sycl::queue cpu_queue{
      sycl::default_selector_v,
      sycl::property_list{
          sycl::property::queue::enable_profiling(),
          sycl::ext::codeplay::experimental::property::queue::enable_fusion()}};

  BuildData<int> build_tables[3];

  build_tables[0].h_filter_col = loadColumn<int>("s_region", S_LEN, cpu_queue);
  build_tables[0].h_dim_key = loadColumn<int>("s_suppkey", S_LEN, cpu_queue);
  build_tables[0].h_dim_val = NULL;
  build_tables[0].num_tuples = S_LEN;
  build_tables[0].num_slots = S_LEN;
  build_tables[0].build_function = [&](int *filter_col, int *dim_key, int *,
                                       int *hash_table, sycl::queue queue,
                                       sycl::event &event) {
    event = queue.submit([&](sycl::handler &cgh) {
      int build_tables_num_tuples_ct2 = build_tables[0].num_tuples;
      int build_tables_num_slots_ct4 = build_tables[0].num_slots;
      cgh.parallel_for(build_tables_num_tuples_ct2, [=](sycl::id<1> idx) {
        build_hashtable_s(filter_col, dim_key, build_tables_num_tuples_ct2,
                          hash_table, build_tables_num_slots_ct4, idx);
      });
    });
  };

  build_tables[1].h_filter_col =
      loadColumn<int>("p_category", P_LEN, cpu_queue);
  build_tables[1].h_dim_key = loadColumn<int>("p_partkey", P_LEN, cpu_queue);
  build_tables[1].h_dim_val = loadColumn<int>("p_brand1", P_LEN, cpu_queue);
  build_tables[1].num_tuples = P_LEN;
  build_tables[1].num_slots = P_LEN;
  build_tables[1].build_function = [&](int *filter_col, int *dim_key,
                                       int *dim_val, int *hash_table,
                                       sycl::queue queue, sycl::event &event) {
    event = queue.submit([&](sycl::handler &cgh) {
      int build_tables_num_tuples_ct3 = build_tables[1].num_tuples;
      int build_tables_num_slots_ct5 = build_tables[1].num_slots;
      cgh.parallel_for(build_tables_num_tuples_ct3, [=](sycl::id<1> idx) {
        build_hashtable_p(filter_col, dim_key, dim_val, build_tables_num_tuples_ct3,
                          hash_table, build_tables_num_slots_ct5, idx);
      });
    });
  };

  build_tables[2].h_filter_col = NULL;
  build_tables[2].h_dim_key = loadColumn<int>("d_datekey", D_LEN, cpu_queue);
  build_tables[2].h_dim_val = loadColumn<int>("d_year", D_LEN, cpu_queue);
  build_tables[2].num_tuples = D_LEN;
  build_tables[2].num_slots = 19981230 - 19920101 + 1;
  build_tables[2].build_function = [&](int *, int *dim_key, int *dim_val,
                                       int *hash_table, sycl::queue queue,
                                       sycl::event &event) {
    event = queue.submit([&](sycl::handler &cgh) {
      int build_tables_num_tuples_ct2 = build_tables[2].num_tuples;
      int build_tables_num_slots_ct4 = build_tables[2].num_slots;
      cgh.parallel_for(build_tables_num_tuples_ct2, [=](sycl::id<1> idx) {
        build_hashtable_d(dim_key, dim_val, build_tables_num_tuples_ct2,
                          hash_table, build_tables_num_slots_ct4, 19920101,
                          idx);
      });
    });
  };

  ProbeData<int, int> prob;
  prob.n_probes = 4;
  prob.h_lo_data = new int *[prob.n_probes];
  prob.h_lo_data[0] = loadColumn<int>("lo_orderdate", LO_LEN, cpu_queue);
  prob.h_lo_data[1] = loadColumn<int>("lo_partkey", LO_LEN, cpu_queue);
  prob.h_lo_data[2] = loadColumn<int>("lo_suppkey", LO_LEN, cpu_queue);
  prob.h_lo_data[3] = loadColumn<int>("lo_revenue", LO_LEN, cpu_queue);
  prob.len_each_probe = LO_LEN;
  prob.res_size = ((1998 - 1992 + 1) * (5 * 5 * 40));
  prob.res_array_cols = 4;
  prob.res_idx = 2;
  prob.probe_function = [&](int **probe_data, int partition_len,
                            int **hash_tables, int *res, sycl::queue queue,
                            sycl::event &event) {
    if (modes == 0) {
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct0 = probe_data[0];
        int *probe_data_ct1 = probe_data[1];
        int *probe_data_ct2 = probe_data[2];
        int *probe_data_ct3 = probe_data[3];
        int *hash_tables_ct5 = hash_tables[0];
        int build_tables_num_slots_ct6 = build_tables[0].num_slots;
        int *hash_tables_ct7 = hash_tables[1];
        int build_tables_num_slots_ct8 = build_tables[1].num_slots;
        int *hash_tables_ct9 = hash_tables[2];
        int build_tables_num_slots_ct10 = build_tables[2].num_slots;
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          probe(probe_data_ct0, probe_data_ct1, probe_data_ct2,
                probe_data_ct3, partition_len, hash_tables_ct5,
                build_tables_num_slots_ct6, hash_tables_ct7,
                build_tables_num_slots_ct8, hash_tables_ct9,
                build_tables_num_slots_ct10, res, idx);
        });
      });
    }

    if (modes == 1 || modes == 2) {
      int *brand = sycl::malloc_shared<int>(partition_len, queue);
      int *year = sycl::malloc_shared<int>(partition_len, queue);
      bool *selection_flags = sycl::malloc_shared<bool>(partition_len, queue);

      queue.parallel_for(partition_len, [=](sycl::id<1> idx) {
        selection_flags[idx] = 1;
        brand[idx] = 0;
        year[idx] = 0;
      });
      queue.wait();

      sycl::ext::codeplay::experimental::fusion_wrapper fw{queue};
      float total_time = 0;
      auto start = std::chrono::high_resolution_clock::now();
      if (modes == 1)
        fw.start_fusion();
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct1 = probe_data[1];
        int *hash_tables_ct7 = hash_tables[1];
        int build_tables_num_slots_ct8 = build_tables[1].num_slots;
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          probe_keys_vals_element(probe_data_ct1[idx], brand[idx],
                                  selection_flags[idx],
                                  build_tables_num_slots_ct8, hash_tables_ct7,
                                  build_tables_num_slots_ct8, 0);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct0 = probe_data[0];
        int *hash_tables_ct9 = hash_tables[2];
        int build_tables_num_slots_ct10 = build_tables[2].num_slots;
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          probe_keys_vals_element(probe_data_ct0[idx], year[idx],
                                  selection_flags[idx],
                                  build_tables_num_slots_ct10, hash_tables_ct9,
                                  build_tables_num_slots_ct10, 19920101);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct2 = probe_data[2];
        int *hash_tables_ct5 = hash_tables[0];
        int build_tables_num_slots_ct6 = build_tables[0].num_slots;
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          probe_keys_element(probe_data_ct2[idx], selection_flags[idx],
                             build_tables_num_slots_ct6, hash_tables_ct5,
                             build_tables_num_slots_ct6, 0);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct3 = probe_data[3];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          if (selection_flags[idx]) {
            int hash = (brand[idx] * 7 + (year[idx] - 1992)) %
                       ((1998 - 1992 + 1) * (5 * 5 * 40));
            res[hash * 4] = year[idx];
            res[hash * 4 + 1] = brand[idx];
            auto sum_obj = sycl::atomic_ref<
                unsigned long long, sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::global_space>(
                *reinterpret_cast<unsigned long long *>(&res[hash * 4 + 2]));
            sum_obj.fetch_add((unsigned long long)(probe_data_ct3[idx]));
          }
        });
      });
      if (modes == 1) {
        event = fw.complete_fusion(
            {sycl::ext::codeplay::experimental::property::no_barriers{}});
        event.wait();
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      event.wait();
      wait_and_add_time(event, total_time);
      unsigned long long selectivity = 0;
      for (int i = 0; i < partition_len; i++)
        if (selection_flags[i])
          selectivity++;
      std::cout << "%Selected " << (float)selectivity/(float)partition_len * 100 << std::endl;
      std::cout << "Q21 >>Internal total timer reported " << total_time
                << " ms.\n";
      std::cout << "Q21 >>External total timer reported "
                << elapsed.count() * 1000 << " ms.\n";
    }
  };

  cout << "Query: q21" << endl;

  exchange_operator_wrapper<int, int, unsigned long long>(
      build_tables, 3, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}