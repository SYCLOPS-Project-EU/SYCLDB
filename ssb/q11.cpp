#include "../exchange.hpp"
#include "ssb_utils.hpp"

#include <getopt.h>
#include <sycl/sycl.hpp>

using namespace std;

template <typename R>
using sum_reduction_t = sycl::reducer<
    R, std::plus<R>, 0, 1UL,
    sycl::detail::ReductionIdentityContainer<R, std::plus<R>, false>>;

void QueryKernel(int *lo_orderdate, int *lo_discount, int *lo_quantity,
                 int *lo_extendedprice, int lo_num_entries,
                 sum_reduction_t<unsigned long long> &sum, sycl::id<1> idx) {
  bool sf = false;
  
  
  //selection_element<int>(sf, tmp_lo_orderdate, NONE, GT, 19930000);
  //selection_element<int>(sf, tmp_lo_orderdate, AND, LT, 19940000);
  //selection_element<int>(sf, tmp_lo_quantity, AND, LT, 25);
  //selection_element<int>(sf, tmp_lo_discount, AND, GE, 1);
  //selection_element<int>(sf, tmp_lo_discount, AND, LE, 3);
  //if (sf)
  //  sum.combine(tmp_lo_discount * tmp_lo_extendedprice);

  selection_element<int>(sf, lo_orderdate[idx], NONE, GT, 19930000);
  selection_element<int>(sf, lo_orderdate[idx], AND, LT, 19940000);
  selection_element<int>(sf, lo_quantity[idx], AND, LT, 25);
  selection_element<int>(sf, lo_discount[idx], AND, GE, 1);
  selection_element<int>(sf, lo_discount[idx], AND, LE, 3);
  if (sf)
    sum.combine(lo_discount[idx] * lo_extendedprice[idx]);
}


template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void QueryKernel(int *lo_orderdate, int *lo_discount, int *lo_quantity,
                 int *lo_extendedprice, int lo_num_entries,
                 sum_reduction_t<unsigned long long> &sum,
                 //unsigned long long *revenue,
                 const sycl::nd_item<1> &item_ct1
                 //long long *buffer
                 ) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  //unsigned long long sum = 0;

  int tile_offset = item_ct1.get_group(0) * TILE_ITEMS;
  int num_tiles = (lo_num_entries + TILE_ITEMS - 1) / TILE_ITEMS;
  int num_tile_items = TILE_ITEMS;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items, num_tile_items, item_ct1);
  BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19930000, selection_flags, num_tile_items, item_ct1);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 19940000, selection_flags, num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_quantity + tile_offset, items, num_tile_items, item_ct1);
  BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 25, selection_flags, num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_discount + tile_offset, items, num_tile_items, item_ct1);
  BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 1, selection_flags, num_tile_items, item_ct1);
  BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 3, selection_flags, num_tile_items, item_ct1);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_extendedprice + tile_offset, items2, num_tile_items, item_ct1);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if ((item_ct1.get_local_id(0) + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        //sum += items[ITEM] * items2[ITEM];
        sum.combine(items[ITEM] * items2[ITEM]);
  }
}

/**
 * Main
 */
int main(int argc, char **argv) {
  int num_partitions = 1;
  int num_gpus = 0;
  int repetitions = 10;
  bool tiles = 0;

  int c;
  while ((c = getopt(argc, argv, "p:g:r:t:")) != -1) {
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
    case 't':
      tiles = atoi(optarg);
      break;
    default:
      abort();
    }
  }
  std::cout << "Tiles? " << tiles << std::endl;

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
      //sycl::local_accessor<long long, 1> buffer_acc_ct1(sycl::range<1>(32),
      //                                                  cgh);

      int *probe_data_ct0 = probe_data[0];
      int *probe_data_ct1 = probe_data[1];
      int *probe_data_ct2 = probe_data[2];
      int *probe_data_ct3 = probe_data[3];

      auto sum_reduction = sycl::reduction(res, std::plus<unsigned long long>());
      if (tiles) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), sum_reduction,
                       [=](sycl::nd_item<1> item_ct1, sum_reduction_t<unsigned long long> &res) {
            QueryKernel<N_BLOCK_THREADS, N_ITEMS_PER_THREAD>(
                probe_data_ct0, probe_data_ct1, probe_data_ct2, probe_data_ct3,
                partition_len, res, item_ct1//,
                //buffer_acc_ct1.get_pointer()
            );
      });
      }
      else {
        cgh.parallel_for(partition_len, sum_reduction,
                         [=](sycl::id<1> idx, sum_reduction_t<unsigned long long> &res) {
          QueryKernel(probe_data_ct0, probe_data_ct1, probe_data_ct2, probe_data_ct3,
                      partition_len, res, idx);
        });
      }
    });
  };

  cout << "Query: q11" << endl;
  exchange_operator_wrapper<int, unsigned long long>(
      NULL, 0, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}
