#include "../exchange.hpp"
#include "ssb_utils.hpp"

#include <getopt.h>
#include <sycl/sycl.hpp>

using namespace std;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
SYCL_EXTERNAL void
DeviceSelectIf(int *lo_orderdate, int *lo_discount, int *lo_quantity,
               int *lo_extendedprice, int lo_num_entries,
               unsigned long long *revenue, const sycl::nd_item<1> &item_ct1,
               long long *buffer) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  unsigned long long sum = 0;

  int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
  int num_tiles = (lo_num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
  int num_tile_items = TILE_ITEMS;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, item_ct1);
  BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940101, selection_flags, num_tile_items, item_ct1);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940131, selection_flags, num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_quantity + tile_offset, items, num_tile_items, item_ct1);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 26, selection_flags, num_tile_items, item_ct1);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 35, selection_flags, num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_discount + tile_offset, items, num_tile_items, item_ct1);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 4, selection_flags, num_tile_items, item_ct1);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 6, selection_flags, num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_extendedprice + tile_offset, items2, num_tile_items, item_ct1);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (item_ct1.get_local_id(0) + (BLOCK_THREADS * ITEM) < num_tile_items)
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

#ifndef COMPILER_IS_ACPP
  unsigned long long aggregate =
      BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(
          sum, (long long *)buffer, item_ct1);
  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (item_ct1.get_local_id(0) == 0) {
    // dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
    //     revenue, aggregate);
    auto sum_obj =
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::global_space>(revenue[0]);
    sum_obj.fetch_add(aggregate);
  }
#else
  auto sum_obj =
      sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                       sycl::memory_scope::work_group,
                       sycl::access::address_space::global_space>(revenue[0]);
  sum_obj.fetch_add(sum);
#endif
}

/**
 * Main
 */
int main(int argc, char **argv) {
  int num_partitions = 1;
  int num_gpus = 1;
  int repetitions = 10;

  int c;
  while ((c = getopt(argc, argv, "p:g:r:")) != -1) {
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
    default:
      abort();
    }
  }

  sycl::queue cpu_queue{
      sycl::default_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling()}};

  ProbeData<int, unsigned long long> prob;
  prob.n_probes = 4;
  prob.h_lo_data = new int *[prob.n_probes];
  prob.h_lo_data[0] = loadColumn<int>("lo_orderdate", LO_LEN, cpu_queue);
  prob.h_lo_data[1] = loadColumn<int>("lo_discount", LO_LEN, cpu_queue);
  prob.h_lo_data[2] = loadColumn<int>("lo_quantity", LO_LEN, cpu_queue);
  prob.h_lo_data[3] = loadColumn<int>("lo_extendedprice", LO_LEN, cpu_queue);
  prob.len_each_probe = LO_LEN;
  prob.res_size = 1;
  prob.res_array_cols = 1;
  prob.res_idx = 0;
  prob.probe_function = [&](int **probe_data, int partition_len, int **,
                            unsigned long long *res, sycl::queue queue,
                            sycl::event &event) {
    sycl::range<1> gws((partition_len + TILE_ITEMS - 1) / TILE_ITEMS *
                       N_BLOCK_THREADS);
    sycl::range<1> lws(N_BLOCK_THREADS);
    event = queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<long long, 1> buffer_acc_ct1(sycl::range<1>(32),
                                                        cgh);

      int *probe_data_ct0 = probe_data[0];
      int *probe_data_ct1 = probe_data[1];
      int *probe_data_ct2 = probe_data[2];
      int *probe_data_ct3 = probe_data[3];

      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1>
                                                            item_ct1) {
#ifndef COMPILER_IS_ACPP
        DeviceSelectIf<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
            probe_data_ct0, probe_data_ct1, probe_data_ct2, probe_data_ct3,
            partition_len, res, item_ct1,
            buffer_acc_ct1.get_multi_ptr<sycl::access::decorated::yes>().get());
#else
            DeviceSelectIf<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
                probe_data_ct0, probe_data_ct1, probe_data_ct2, probe_data_ct3,
                partition_len, res, item_ct1,
                buffer_acc_ct1.get_pointer()
            );
#endif
      });
    });
  };

  cout << "Query: q12" << endl;
  exchange_operator_wrapper<int, unsigned long long>(
      NULL, 0, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}
