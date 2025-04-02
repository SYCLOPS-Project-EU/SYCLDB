#include "../exchange.hpp"
#include "ssb_utils.hpp"

#include <getopt.h>
#include <sycl/sycl.hpp>

using namespace std;

void probe(int *lo_orderdate, int *lo_custkey, int *lo_suppkey, int *lo_revenue,
           int lo_len, int *ht_s, int s_len, int *ht_c, int c_len, int *ht_d,
           int d_len, int *res, sycl::id<1> idx) {
  int s_nation;
  int c_nation;
  int year;
  bool sf = true;
  probe_keys_vals_element(lo_suppkey[idx], s_nation, sf, s_len, ht_s, s_len, 0);
  probe_keys_vals_element(lo_custkey[idx], c_nation, sf, c_len, ht_c, c_len, 0);
  probe_keys_vals_element(lo_orderdate[idx], year, sf, d_len, ht_d, d_len,
                          19920101);
  if (sf) {
    int hash = (s_nation * 250 * 7 + c_nation * 7 + (year - 1992)) %
               ((1998 - 1992 + 1) * 250 * 250);
    res[hash * 6] = year;
    res[hash * 6 + 1] = c_nation;
    res[hash * 6 + 2] = s_nation;
    auto sum_obj =
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::global_space>(
            *reinterpret_cast<unsigned long long *>(&res[hash * 6 + 4]));
    sum_obj.fetch_add((unsigned long long)(lo_revenue[idx]));
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
SYCL_EXTERNAL void build_hashtable_s(int *filter_col, int *dim_key,
                                     int *dim_val, int num_tuples,
                                     int *hash_table, int num_slots,
                                     const sycl::nd_item<1> &item_ct1) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
  int num_tiles = (num_tuples + TILE_ITEMS - 1) / TILE_ITEMS;
  int num_tile_items = TILE_ITEMS;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      filter_col + tile_offset, items, num_tile_items, item_ct1);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 24, selection_flags,
                                                    num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items, item_ct1);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items, item_ct1);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, 0, num_tile_items,
      item_ct1);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
SYCL_EXTERNAL void build_hashtable_c(int *filter_col, int *dim_key,
                                     int *dim_val, int num_tuples,
                                     int *hash_table, int num_slots,
                                     const sycl::nd_item<1> &item_ct1) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
  int num_tiles = (num_tuples + TILE_ITEMS - 1) / TILE_ITEMS;
  int num_tile_items = TILE_ITEMS;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      filter_col + tile_offset, items, num_tile_items, item_ct1);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 24, selection_flags,
                                                    num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items, item_ct1);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items, item_ct1);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, 0, num_tile_items,
      item_ct1);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
SYCL_EXTERNAL void build_hashtable_d(int *dim_key, int *dim_val, int num_tuples,
                                     int *hash_table, int num_slots,
                                     int val_min,
                                     const sycl::nd_item<1> &item_ct1) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
  int num_tiles = (num_tuples + TILE_ITEMS - 1) / TILE_ITEMS;
  int num_tile_items = TILE_ITEMS;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items, item_ct1);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1992, selection_flags, num_tile_items, item_ct1);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1997, selection_flags, num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items, item_ct1);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, 19920101,
      num_tile_items, item_ct1);
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
  std::cout << "MODE " << modes << std::endl;

  sycl::queue cpu_queue{
      sycl::default_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling(),
      sycl::ext::codeplay::experimental::property::queue::enable_fusion()}};

  BuildData<int> build_tables[3];

  build_tables[0].h_filter_col = loadColumn<int>("s_nation", S_LEN, cpu_queue);
  build_tables[0].h_dim_key = loadColumn<int>("s_suppkey", S_LEN, cpu_queue);
  build_tables[0].h_dim_val = loadColumn<int>("s_city", S_LEN, cpu_queue);
  build_tables[0].num_tuples = S_LEN;
  build_tables[0].num_slots = S_LEN;
  build_tables[0].build_function = [&](int *filter_col, int *dim_key,
                                       int *dim_val, int *hash_table,
                                       sycl::queue queue, sycl::event &event) {
    sycl::range<1> gws((build_tables[0].num_tuples + TILE_ITEMS - 1) /
                       TILE_ITEMS * N_BLOCK_THREADS);
    sycl::range<1> lws(N_BLOCK_THREADS);
    event = queue.submit([&](sycl::handler &cgh) {
      int build_tables_num_tuples_ct3 = build_tables[0].num_tuples;
      int build_tables_num_slots_ct5 = build_tables[0].num_slots;

      cgh.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item_ct1) {
            build_hashtable_s<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
                filter_col, dim_key, dim_val, build_tables_num_tuples_ct3,
                hash_table, build_tables_num_slots_ct5, item_ct1);
          });
    });
  };

  build_tables[1].h_filter_col = loadColumn<int>("c_nation", C_LEN, cpu_queue);
  build_tables[1].h_dim_key = loadColumn<int>("c_custkey", C_LEN, cpu_queue);
  build_tables[1].h_dim_val = loadColumn<int>("c_city", C_LEN, cpu_queue);
  build_tables[1].num_tuples = C_LEN;
  build_tables[1].num_slots = C_LEN;
  build_tables[1].build_function = [&](int *filter_col, int *dim_key,
                                       int *dim_val, int *hash_table,
                                       sycl::queue queue, sycl::event &event) {
    sycl::range<1> gws((build_tables[1].num_tuples + TILE_ITEMS - 1) /
                       TILE_ITEMS * N_BLOCK_THREADS);
    sycl::range<1> lws(N_BLOCK_THREADS);
    event = queue.submit([&](sycl::handler &cgh) {
      int build_tables_num_tuples_ct3 = build_tables[1].num_tuples;
      int build_tables_num_slots_ct5 = build_tables[1].num_slots;

      cgh.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item_ct1) {
            build_hashtable_c<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
                filter_col, dim_key, dim_val, build_tables_num_tuples_ct3,
                hash_table, build_tables_num_slots_ct5, item_ct1);
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
    sycl::range<1> gws((build_tables[2].num_tuples + TILE_ITEMS - 1) /
                       TILE_ITEMS * N_BLOCK_THREADS);
    sycl::range<1> lws(N_BLOCK_THREADS);
    event = queue.submit([&](sycl::handler &cgh) {
      int build_tables_num_tuples_ct2 = build_tables[2].num_tuples;
      int build_tables_num_slots_ct4 = build_tables[2].num_slots;

      cgh.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item_ct1) {
            build_hashtable_d<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
                dim_key, dim_val, build_tables_num_tuples_ct2, hash_table,
                build_tables_num_slots_ct4, 19920101, item_ct1);
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
  prob.res_size = ((1998 - 1992 + 1) * 250 * 250);
  prob.res_array_cols = 6;
  prob.res_idx = 4;
  prob.probe_function = [&](int **probe_data, int partition_len,
                            int **hash_tables, int *res, sycl::queue queue,
                            sycl::event &event) {
    sycl::range<1> gws((partition_len + TILE_ITEMS - 1) / TILE_ITEMS *
                       N_BLOCK_THREADS);
    sycl::range<1> lws(N_BLOCK_THREADS);
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
        cgh.parallel_for(partition_len, [=](sycl::id<1> id) {
            probe(probe_data_ct0, probe_data_ct1, probe_data_ct2,
                  probe_data_ct3, partition_len, hash_tables_ct5,
                  build_tables_num_slots_ct6, hash_tables_ct7,
                  build_tables_num_slots_ct8, hash_tables_ct9,
                  build_tables_num_slots_ct10, res, id);
          });
      });
    }

    if (modes == 1 || modes == 2) {
      int *s_nation = sycl::malloc_shared<int>(partition_len, queue);
      int *c_nation = sycl::malloc_shared<int>(partition_len, queue);
      int *year = sycl::malloc_shared<int>(partition_len, queue);
      bool *selection_flags = sycl::malloc_shared<bool>(partition_len, queue);

      queue.parallel_for(partition_len, [=](sycl::id<1> idx) {
        selection_flags[idx] = 1;
        s_nation[idx] = 0;
        c_nation[idx] = 0;
        year[idx] = 0;
      });
      queue.wait();

      sycl::ext::codeplay::experimental::fusion_wrapper fw{queue};
      float total_time = 0;
      auto start = std::chrono::high_resolution_clock::now();
      if (modes == 1)
        fw.start_fusion();
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct2 = probe_data[2];
        int *hash_tables_ct5 = hash_tables[0];
        int build_tables_num_slots_ct6 = build_tables[0].num_slots;
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          probe_keys_vals_element(probe_data_ct2[idx], s_nation[idx],
                                  selection_flags[idx],
                                  build_tables_num_slots_ct6, hash_tables_ct5,
                                  build_tables_num_slots_ct6, 0);
        });
      });
      //if (modes == 2)
      //  wait_and_add_time(event, total_time);
      if (modes == 2) {
        wait_and_add_time(event, total_time);
        unsigned long long selectivity = 0;
        for (int i = 0; i < partition_len; i++)
          if (selection_flags[i])
            selectivity++;
        std::cout << "%Selected " << (float)selectivity/(float)partition_len * 100 << std::endl;
      }
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct1 = probe_data[1];
        int *hash_tables_ct7 = hash_tables[1];
        int build_tables_num_slots_ct8 = build_tables[1].num_slots;
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          probe_keys_vals_element(probe_data_ct1[idx], c_nation[idx],
                                  selection_flags[idx],
                                  build_tables_num_slots_ct8, hash_tables_ct7,
                                  build_tables_num_slots_ct8, 0);
        });
      });
      //if (modes == 2)
      //  wait_and_add_time(event, total_time);
      if (modes == 2) {
        wait_and_add_time(event, total_time);
        unsigned long long selectivity = 0;
        for (int i = 0; i < partition_len; i++)
          if (selection_flags[i])
            selectivity++;
        std::cout << "%Selected " << (float)selectivity/(float)partition_len * 100 << std::endl;
      }
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
      //if (modes == 2)
      //  wait_and_add_time(event, total_time);
      if (modes == 2) {
        wait_and_add_time(event, total_time);
        unsigned long long selectivity = 0;
        for (int i = 0; i < partition_len; i++)
          if (selection_flags[i])
            selectivity++;
        std::cout << "%Selected " << (float)selectivity/(float)partition_len * 100 << std::endl;
      }
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct3 = probe_data[3];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          if (selection_flags[idx]) {
            int hash = (s_nation[idx] * 250 * 7 + c_nation[idx] * 7 +
                        (year[idx] - 1992)) %
                       ((1998 - 1992 + 1) * 250 * 250);
            res[hash * 6] = year[idx];
            res[hash * 6 + 1] = c_nation[idx];
            res[hash * 6 + 2] = s_nation[idx];
            auto sum_obj = sycl::atomic_ref<
                unsigned long long, sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::global_space>(
                *reinterpret_cast<unsigned long long *>(&res[hash * 6 + 4]));
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
      std::cout << "Q32 >>Internal total timer reported " << total_time
                << " ms.\n";
      std::cout << "Q32 >>External total timer reported "
                << elapsed.count() * 1000 << " ms.\n";
    }
  };

  cout << "Query: q32" << endl;
  exchange_operator_wrapper<int, int, unsigned long long>(
      build_tables, 3, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}