#include <getopt.h>
#include <sycl/sycl.hpp>

#include "include/kernels.hpp"
#include "utils/generator.h"

using namespace std;

#include <chrono>

void wait_and_measure_time(sycl::event &event, const std::string &name, float &elapsed) {
  auto start = std::chrono::high_resolution_clock::now();
  event.wait();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  elapsed += duration.count();
}

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
  int target_device = 1;
  int repetitions = 10;

  int c;
  while ((c = getopt(argc, argv, "f:d:t:r:")) != -1) {
    switch (c) {
    case 'f':
      num_fact = atoi(optarg);
      break;
    case 'd':
      num_dim = atoi(optarg);
      break;
    case 't':
      target_device = atoi(optarg);
      break;
    case 'r':
      repetitions = atoi(optarg);
      break;
    default:
      abort();
    }
  }

  sycl::queue q;
  if (target_device == 1) { // CPU
    q = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::enable_profiling{});
  } else if (target_device == 2) { // NVIDIA
    q = sycl::queue([](const sycl::device& d) {
      return d.is_gpu() && d.get_info<sycl::info::device::vendor>().find("NVIDIA") != std::string::npos ? 1 : -1;
    }, sycl::property::queue::enable_profiling{});
  } else if (target_device == 3) { // AMD
    q = sycl::queue([](const sycl::device& d) {
      return d.is_gpu() && d.get_info<sycl::info::device::vendor>().find("AMD") != std::string::npos ? 1 : -1;
    }, sycl::property::queue::enable_profiling{});
  } else if (target_device == 4) { // Intel
    q = sycl::queue([](const sycl::device& d) {
      return d.is_gpu() && d.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos ? 1 : -1;
    }, sycl::property::queue::enable_profiling{});
  } else {
    q = sycl::queue(sycl::default_selector_v, sycl::property::queue::enable_profiling{});
  }

  cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  sycl::queue cpu_queue{
      sycl::cpu_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling()}};

  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  create_relation_pk(h_dim_key, h_dim_val, num_dim, cpu_queue);

  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim, cpu_queue);

  int *d_dim_key = sycl::malloc_device<int>(num_dim, q);
  int *d_dim_val = sycl::malloc_device<int>(num_dim, q);
  int *d_hash_table = sycl::malloc_device<int>(num_dim * 2, q);
  int *d_fact_fkey = sycl::malloc_device<int>(num_fact, q);
  int *d_fact_val = sycl::malloc_device<int>(num_fact, q);
  unsigned long long *d_res = sycl::malloc_device<unsigned long long>(1, q);

  q.memcpy(d_dim_key, h_dim_key, num_dim * sizeof(int)).wait();
  q.memcpy(d_dim_val, h_dim_val, num_dim * sizeof(int)).wait();
  q.memcpy(d_fact_fkey, h_fact_fkey, num_fact * sizeof(int)).wait();
  q.memcpy(d_fact_val, h_fact_val, num_fact * sizeof(int)).wait();

  cout << "Query: join (standalone)" << endl;
  
  for (int r = 0; r < repetitions; r++) {
    q.memset(d_res, 0, sizeof(unsigned long long)).wait();
    q.memset(d_hash_table, 0, num_dim * 2 * sizeof(int)).wait();
    
    float elapsed_build = 0;
    sycl::event ev_build = build(d_dim_key, d_dim_val, num_dim, d_hash_table, num_dim, q);
    wait_and_measure_time(ev_build, "Build", elapsed_build);
    
    float elapsed_probe = 0;
    sycl::event ev_probe = probe(d_fact_fkey, d_fact_val, num_fact, d_hash_table, num_dim, d_res, q);
    wait_and_measure_time(ev_probe, "Probe", elapsed_probe);
    
    unsigned long long h_res = 0;
    q.memcpy(&h_res, d_res, sizeof(unsigned long long)).wait();
    
    cout << "Repetition " << r << " Result: " << h_res 
         << " | Build time: " << elapsed_build << " ms"
         << " | Probe time: " << elapsed_probe << " ms" << endl;
  }

  sycl::free(d_dim_key, q);
  sycl::free(d_dim_val, q);
  sycl::free(d_hash_table, q);
  sycl::free(d_fact_fkey, q);
  sycl::free(d_fact_val, q);
  sycl::free(d_res, q);

  sycl::free(h_dim_key, cpu_queue);
  sycl::free(h_dim_val, cpu_queue);
  sycl::free(h_fact_fkey, cpu_queue);
  sycl::free(h_fact_val, cpu_queue);

  return 0;
}
