#include "../runner.hpp"
#include "ssb_utils.hpp"

#include <getopt.h>
#include <sycl/sycl.hpp>

using namespace std;



void QueryKernel(int *lo_orderdate, int *lo_discount, int *lo_quantity,
                 int *lo_extendedprice, int lo_num_entries,
                 unsigned long long *revenue, bool use_sharding, sycl::id<1> idx) {
  bool sf = false;
  selection_element<int>(sf, lo_orderdate[idx], NONE, GE, 19940101);
  selection_element<int>(sf, lo_orderdate[idx], AND, LE, 19940131);
  selection_element<int>(sf, lo_quantity[idx], AND, GE, 26);
  selection_element<int>(sf, lo_quantity[idx], AND, LE, 35);
  selection_element<int>(sf, lo_discount[idx], AND, GE, 4);
  selection_element<int>(sf, lo_discount[idx], AND, LE, 6);
  if (sf) {
    auto sum_obj =
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::global_space>(
            *reinterpret_cast<unsigned long long *>(&revenue[use_sharding ? (idx[0]/1024)%1024 : 0]));
    sum_obj.fetch_add(
        (unsigned long long)(lo_discount[idx] * lo_extendedprice[idx]));
  }
}

int main(int argc, char **argv) {
  int target_device = 1;
  
  int repetitions = 10;
  int modes = 0;
  int optimize = 0;

  int c;
  while ((c = getopt(argc, argv, "t:r:m:O:")) != -1) {
    switch (c) {
    case 't':
      target_device = atoi(optarg);
      break;
    case 'r':
      repetitions = atoi(optarg);
      break;
    case 'm':
      modes = atoi(optarg);
      break;
    case 'O': optimize = atoi(optarg); break;
    default:
      abort();
    }
  }

  

  sycl::queue q;
  if (target_device == 1) {
    q = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::enable_profiling{});
  } else if (target_device == 2) {
    q = sycl::queue([](const sycl::device& d) { return d.is_gpu() && d.get_info<sycl::info::device::vendor>().find("NVIDIA") != std::string::npos ? 1 : -1; }, sycl::property::queue::enable_profiling{});
  } else if (target_device == 3) {
    q = sycl::queue([](const sycl::device& d) { return d.is_gpu() && d.get_info<sycl::info::device::vendor>().find("AMD") != std::string::npos ? 1 : -1; }, sycl::property::queue::enable_profiling{});
  } else if (target_device == 4) {
    q = sycl::queue([](const sycl::device& d) { return d.is_gpu() && d.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos ? 1 : -1; }, sycl::property::queue::enable_profiling{});
  } else {
    q = sycl::queue(sycl::default_selector_v, sycl::property::queue::enable_profiling{});
  }
  std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  sycl::queue cpu_queue{sycl::cpu_selector_v, sycl::property::queue::enable_profiling{}};
  
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
    bool use_sharding = queue.get_device().is_cpu() && (optimize == 1);
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
                      probe_data_ct3, partition_len, res, use_sharding, idx);
        });
      });
    }
    if (modes == 1) {
      bool *selection_flags = sycl::malloc_shared<bool>(partition_len, queue);
      queue.memset(selection_flags, 0, partition_len * sizeof(bool));
      queue.wait();
      
      float total_time = 0;
      auto start = std::chrono::high_resolution_clock::now();
      
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct0 = probe_data[0];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct0[idx],
                                 NONE, GE, 19940101);
        });
      });
      
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct0 = probe_data[0];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct0[idx], AND,
                                 LE, 19940131);
        });
      });
      
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct2 = probe_data[2];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct2[idx], AND,
                                 GE, 26);
        });
      });
      
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct2 = probe_data[2];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct2[idx], AND,
                                 LE, 35);
        });
      });
      
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct1 = probe_data[1];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct1[idx], AND,
                                 GE, 4);
        });
      });
      
        wait_and_add_time(event, total_time);
      event = queue.submit([&](sycl::handler &cgh) {
        int *probe_data_ct1 = probe_data[1];
        cgh.parallel_for(partition_len, [=](sycl::id<1> idx) {
          selection_element<int>(selection_flags[idx], probe_data_ct1[idx], AND,
                                 LE, 6);
        });
      });
      
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
                    *reinterpret_cast<unsigned long long *>(&res[use_sharding ? (idx[0]/1024)%1024 : 0]));
            sum_obj.fetch_add((unsigned long long)(probe_data_ct1[idx] *
                                                   probe_data_ct3[idx]));
          }
        });
      });

      
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      event.wait();
      wait_and_add_time(event, total_time);
      unsigned long long selectivity = 0;
      for (int i = 0; i < partition_len; i++)
        if (selection_flags[i])
          selectivity++;
      std::cout << "%Selected " << (float)selectivity/(float)partition_len * 100 << std::endl;
      std::cout << "Q12 >>Internal total timer reported " << total_time
                << " ms.\n";
      std::cout << "Q12 >>External total timer reported "
                << elapsed.count() * 1000 << " ms.\n";
    }
  };

  cout << "Query: q12" << endl;
  run_benchmark((BuildData<int>*)nullptr, 0, prob, q, repetitions, cpu_queue, optimize == 1);

  return 0;
}
