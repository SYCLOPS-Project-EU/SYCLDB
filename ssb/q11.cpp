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
                 unsigned long long *revenue,
                 // sum_reduction_t<unsigned long long> &sum,
                 sycl::id<1> idx) {
  bool sf = false;
  selection_element<int>(sf, lo_orderdate[idx], NONE, GT, 19930000);
  selection_element<int>(sf, lo_orderdate[idx], AND, LT, 19940000);
  selection_element<int>(sf, lo_quantity[idx], AND, LT, 25);
  selection_element<int>(sf, lo_discount[idx], AND, GE, 1);
  selection_element<int>(sf, lo_discount[idx], AND, LE, 3);
  if (sf) {
    auto sum_obj =
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::global_space>(
            *reinterpret_cast<unsigned long long *>(&revenue[0]));
    sum_obj.fetch_add(
        (unsigned long long)(lo_discount[idx] * lo_extendedprice[idx]));
  }
}

int main(int argc, char **argv) {
  int num_partitions = 1;
  int num_gpus = 0;
  int repetitions = 10;
  int modes = 1;

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
    if (modes == 0) {
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct0 = probe_data[0];
        int *probe_data_ct1 = probe_data[1];
        int *probe_data_ct2 = probe_data[2];
        int *probe_data_ct3 = probe_data[3];

        auto sum_reduction =
            sycl::reduction(res, std::plus<unsigned long long>());
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          QueryKernel(probe_data_ct0, probe_data_ct1, probe_data_ct2,
                      probe_data_ct3, partition_len, res, idx);
        });
      });
    }
    if (modes == 1 || modes == 2) {
      bool *selection_flags = sycl::malloc_shared<bool>(partition_len, queue);
      queue.memset(selection_flags, 0, partition_len * sizeof(bool));
      queue.wait();
      sycl::ext::codeplay::experimental::fusion_wrapper fw{queue};
      float total_time = 0;
      auto start = std::chrono::high_resolution_clock::now();
      if (modes == 1)
        fw.start_fusion();
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct0 = probe_data[0];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct0[idx],
                                 NONE, GT, 19930000);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct0 = probe_data[0];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct0[idx], AND,
                                 LT, 19940000);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct2 = probe_data[2];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct2[idx], AND,
                                 LT, 25);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct1 = probe_data[1];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct1[idx], AND,
                                 GE, 1);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct1 = probe_data[1];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct1[idx], AND,
                                 LE, 3);
        });
      });
      if (modes == 2)
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct1 = probe_data[1];
        int *probe_data_ct3 = probe_data[3];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          if (selection_flags[idx]) {
            auto sum_obj =
                sycl::atomic_ref<unsigned long long,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::global_space>(
                    *reinterpret_cast<unsigned long long *>(&res[0]));
            sum_obj.fetch_add((unsigned long long)(probe_data_ct1[idx] *
                                                   probe_data_ct3[idx]));
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
      std::cout << "Q11 >>Internal total timer reported " << total_time
                << " ms.\n";
      std::cout << "Q11 >>External total timer reported "
                << elapsed.count() * 1000 << " ms.\n";
    }
  };

  cout << "Query: q11" << endl;
  exchange_operator_wrapper<int, unsigned long long>(
      NULL, 0, prob, num_gpus, num_partitions, repetitions, cpu_queue);

  return 0;
}
