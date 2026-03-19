#pragma once
#include "include/kernels.hpp"
#include <sycl/sycl.hpp>
#include <string>
#include <iostream>
#include <functional>
#include <vector>

using namespace std;

template <typename T> struct BuildData {
  T *h_filter_col;
  T *h_dim_key;
  T *h_dim_val;
  int num_tuples;
  int num_slots;
  std::function<void(T *, T *, T *, T *, sycl::queue, sycl::event &)>
      build_function;
};

template <typename T, typename R> struct ProbeData {
  int n_probes;
  T **h_lo_data;
  int len_each_probe;
  int res_size;
  int res_array_cols;
  int res_idx;
  std::function<void(T **, int, T **, R *, sycl::queue, sycl::event &)>
      probe_function;
};

template<typename T, typename R>
void run_benchmark(BuildData<T>* builds, int n_builds, ProbeData<T, R>& probe, sycl::queue& q, int repetitions, sycl::queue& cpu_queue) {
    // 1. Setup Build Device Memory
    std::vector<T*> d_filter_cols(n_builds, nullptr);
    std::vector<T*> d_dim_keys(n_builds, nullptr);
    std::vector<T*> d_dim_vals(n_builds, nullptr);
    std::vector<T*> d_hash_tables(n_builds, nullptr);

    for (int i = 0; i < n_builds; i++) {
        if (builds[i].h_filter_col != nullptr) {
            d_filter_cols[i] = sycl::malloc_device<T>(builds[i].num_tuples, q);
            q.memcpy(d_filter_cols[i], builds[i].h_filter_col, builds[i].num_tuples * sizeof(T)).wait();
        }
        if (builds[i].h_dim_key != nullptr) {
            d_dim_keys[i] = sycl::malloc_device<T>(builds[i].num_tuples, q);
            q.memcpy(d_dim_keys[i], builds[i].h_dim_key, builds[i].num_tuples * sizeof(T)).wait();
        }
        if (builds[i].h_dim_val != nullptr) {
            d_dim_vals[i] = sycl::malloc_device<T>(builds[i].num_tuples, q);
            q.memcpy(d_dim_vals[i], builds[i].h_dim_val, builds[i].num_tuples * sizeof(T)).wait();
        }
        d_hash_tables[i] = sycl::malloc_device<T>(2 * builds[i].num_slots, q); // The exchange wrapper used 2*num_slots
    }

    // 2. Setup Probe Device Memory
    T** d_probe_data = new T*[probe.n_probes];
    for(int i = 0; i < probe.n_probes; i++) {
        d_probe_data[i] = sycl::malloc_device<T>(probe.len_each_probe, q);
        q.memcpy(d_probe_data[i], probe.h_lo_data[i], probe.len_each_probe * sizeof(T)).wait();
    }
    
    R* d_res = sycl::malloc_device<R>(probe.res_size * probe.res_array_cols, q);

    for (int r = 0; r < repetitions; r++) {
        q.memset(d_res, 0, probe.res_size * probe.res_array_cols * sizeof(R)).wait();
        for (int i = 0; i < n_builds; i++) {
            q.memset(d_hash_tables[i], 0, 2 * builds[i].num_slots * sizeof(T)).wait();
            sycl::event e;
            builds[i].build_function(d_filter_cols[i], d_dim_keys[i], d_dim_vals[i], d_hash_tables[i], q, e);
            e.wait();
        }

        sycl::event e;
        auto start = std::chrono::high_resolution_clock::now();
        probe.probe_function(d_probe_data, probe.len_each_probe, d_hash_tables.data(), d_res, q, e);
        e.wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Repetition " << r << " Kernel Time: " << elapsed.count() * 1000 << " ms" << std::endl;
    }

    // 3. Cleanup Resource
    for (int i = 0; i < n_builds; i++) {
        if (d_filter_cols[i]) sycl::free(d_filter_cols[i], q);
        if (d_dim_keys[i]) sycl::free(d_dim_keys[i], q);
        if (d_dim_vals[i]) sycl::free(d_dim_vals[i], q);
        sycl::free(d_hash_tables[i], q);
    }
    for(int i = 0; i < probe.n_probes; i++) {
        sycl::free(d_probe_data[i], q);
    }
    delete[] d_probe_data;
    sycl::free(d_res, q);
}
