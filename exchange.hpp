#include <functional>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

#include "include/join.hpp"
#include "include/load.hpp"
#include "include/pred.hpp"
#ifndef COMPILER_IS_ACPP
#include "include/reduce.hpp"
#endif
#include "include/store.hpp"
#include <cmath>

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

struct perf_times {
  float transfer_kernels_time;
  float probe_kernels_time;
  float total_exec_time;
};

// https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/DPCXX.md
void wait_and_measure_time(sycl::event e, std::string name, float &total_time) {
  e.wait();
  const auto start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
  float time = (end - start) / 1e6;
  std::cout << name << " took: " << time << " ms" << std::endl;
  total_time += time;
}

template <typename T, typename R, typename S>
perf_times exchange_operator(const BuildData<T> *builds, int n_builds,
                             const ProbeData<T, R> &probe, int n_gpus,
                             int n_partitions) {
  int partition_size = (probe.len_each_probe + n_partitions - 1) / n_partitions;
  int last_partition_size =
      probe.len_each_probe - (n_partitions - 1) * partition_size;
  // since we compute 1 partition at a time, it suffices to keep at most 2
  // while asynchronously transferring the i-th partition, compute the (i-1)-th
  int n_min_partitions_per_gpu = std::min(n_partitions, 2);

  // https://github.com/ENCCS/sycl-workshop/blob/main/content/device-discovery.rst
  // need to "discover" the GPUs in advance
  std::vector<sycl::queue> general_queues;
  std::vector<sycl::queue> kernel_queues;
  auto platforms = sycl::platform::get_platforms();
  for (auto &platform : platforms) {
    auto devices = platform.get_devices();
    for (auto &d : devices) {
      if (d.is_gpu() && general_queues.size() < n_gpus) {
        general_queues.push_back(sycl::queue(
            d, // sycl::async_handler{},
            sycl::property_list{sycl::property::queue::enable_profiling()}));
        kernel_queues.push_back(sycl::queue(
            general_queues[general_queues.size() - 1].get_context(),
            d, // sycl::async_handler{},
            sycl::property_list{sycl::property::queue::enable_profiling()}));
      }
    }
  }

  if (general_queues.size() < n_gpus) {
    std::cout << "Not enough GPUs available" << std::endl;
    std::cout << "Available GPUs: " << general_queues.size() << std::endl;
    exit(1);
  }

  R **h_res = sycl::malloc_host<R *>(n_gpus, general_queues[0]);

  std::vector<std::thread> threads;
  std::atomic<int> interval_idx(0);
  float transfer_kernels_time = 0;
  float probe_kernels_time = 0;

  auto itime = std::chrono::steady_clock::now();

  for (int gpu_id = 0; gpu_id < n_gpus; gpu_id++) {
    threads.push_back(std::thread([&, gpu_id]() {
      sycl::event e{};
      sycl::event et{};
      // with a shared output, no need to manually transfer the result back
      h_res[gpu_id] = sycl::malloc_shared<R>(
          probe.res_size * probe.res_array_cols, general_queues[gpu_id]);
      T **d_filter_cols =
          sycl::malloc_host<T *>(n_builds, general_queues[gpu_id]);
      T **d_dim_keys = sycl::malloc_host<T *>(n_builds, general_queues[gpu_id]);
      T **d_dim_vals = sycl::malloc_host<T *>(n_builds, general_queues[gpu_id]);
      T **d_hash_tables =
          sycl::malloc_host<T *>(n_builds, general_queues[gpu_id]);
      general_queues[gpu_id].wait();
      for (int i = 0; i < n_builds; i++) {
        if (builds[i].h_filter_col != NULL) {
          d_filter_cols[i] = sycl::malloc_device<T>(builds[i].num_tuples,
                                                    general_queues[gpu_id]);
          general_queues[gpu_id].memcpy(d_filter_cols[i],
                                        builds[i].h_filter_col,
                                        builds[i].num_tuples * sizeof(T));
        }
        if (builds[i].h_dim_key != NULL) {
          d_dim_keys[i] = sycl::malloc_device<T>(builds[i].num_tuples,
                                                 general_queues[gpu_id]);
          general_queues[gpu_id].memcpy(d_dim_keys[i], builds[i].h_dim_key,
                                        builds[i].num_tuples * sizeof(T));
        }
        if (builds[i].h_dim_val != NULL) {
          d_dim_vals[i] = sycl::malloc_device<T>(builds[i].num_tuples,
                                                 general_queues[gpu_id]);
          general_queues[gpu_id].memcpy(d_dim_vals[i], builds[i].h_dim_val,
                                        builds[i].num_tuples * sizeof(T));
        }
        d_hash_tables[i] = sycl::malloc_device<T>(2 * builds[i].num_slots,
                                                  general_queues[gpu_id]);
        general_queues[gpu_id].memset(d_hash_tables[i], 0,
                                      2 * builds[i].num_slots * sizeof(T));
      }
      general_queues[gpu_id].wait();
      for (int i = 0; i < n_builds; i++) {
        builds[i].build_function(d_filter_cols[i], d_dim_keys[i], d_dim_vals[i],
                                 d_hash_tables[i], kernel_queues[gpu_id], e);
        std::string build_name =
            "Build " + std::to_string(i) + " on GPU " + std::to_string(gpu_id);
        // wait_and_measure_time(e, build_name, build_kernels_time);
        e.wait();
      }

      T ***d_probe_data = sycl::malloc_host<T **>(n_min_partitions_per_gpu,
                                                  general_queues[gpu_id]);
      for (int i = 0; i < n_min_partitions_per_gpu; i++) {
        d_probe_data[i] =
            sycl::malloc_host<T *>(probe.n_probes, general_queues[gpu_id]);
        for (int j = 0; j < probe.n_probes; j++) {
          d_probe_data[i][j] =
              sycl::malloc_device<T>(partition_size, general_queues[gpu_id]);
        }
      }
      general_queues[gpu_id].wait();
      kernel_queues[gpu_id].wait();
      bool first_kernel_on_this_device = true;
      // exchange operation
      while (interval_idx < n_partitions) {
        for (int i = 0; i < n_min_partitions_per_gpu; i++) {
          int this_idx = interval_idx++;
          if (this_idx >= n_partitions)
            break;
          int this_partition_size = (this_idx == n_partitions - 1)
                                        ? last_partition_size
                                        : partition_size;
          for (int j = 0; j < probe.n_probes; j++) {
            et = general_queues[gpu_id].memcpy(d_probe_data[i][j],
                                               probe.h_lo_data[j] +
                                                   this_idx * partition_size,
                                               this_partition_size * sizeof(T));
          }
          general_queues[gpu_id].wait(); // i-th transfer is done before
                                         // computing the i-th kernel
          kernel_queues[gpu_id].wait();  // (i-1)-th kernel is done before
                                         // computing the i-th kernel
          wait_and_measure_time(
              et, "Probe data transfer on GPU " + std::to_string(gpu_id),
              transfer_kernels_time);
          if (!first_kernel_on_this_device) {
            wait_and_measure_time(
                e, "Probe kernel on GPU " + std::to_string(gpu_id),
                probe_kernels_time);
          }
          probe.probe_function(d_probe_data[i], this_partition_size,
                               d_hash_tables, h_res[gpu_id],
                               kernel_queues[gpu_id], e);
          if (first_kernel_on_this_device) {
            first_kernel_on_this_device = false;
          }
        }
      }
      // wait for the last kernel to finish
      wait_and_measure_time(e, "Probe on GPU " + std::to_string(gpu_id),
                            probe_kernels_time);
      for (int i = 0; i < n_builds; i++) {
        if (builds[i].h_filter_col != NULL) {
          sycl::free(d_filter_cols[i], general_queues[gpu_id]);
        }
        if (builds[i].h_dim_key != NULL) {
          sycl::free(d_dim_keys[i], general_queues[gpu_id]);
        }
        if (builds[i].h_dim_val != NULL) {
          sycl::free(d_dim_vals[i], general_queues[gpu_id]);
        }
        sycl::free(d_hash_tables[i], general_queues[gpu_id]);
      }
      for (int i = 0; i < n_min_partitions_per_gpu; i++) {
        for (int j = 0; j < probe.n_probes; j++) {
          sycl::free(d_probe_data[i][j], general_queues[gpu_id]);
        }
        sycl::free(d_probe_data[i], general_queues[gpu_id]);
      }
      if (n_builds) {
        sycl::free(d_filter_cols, general_queues[gpu_id]);
        sycl::free(d_dim_keys, general_queues[gpu_id]);
        sycl::free(d_dim_vals, general_queues[gpu_id]);
        sycl::free(d_hash_tables, general_queues[gpu_id]);
      }
      sycl::free(d_probe_data, general_queues[gpu_id]);
    }));
  }
  for (auto &t : threads) {
    t.join();
  }

  auto ftime = std::chrono::steady_clock::now();
  float exec_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(ftime - itime)
          .count();

  int res_count = 0;
  S total_revenue = 0;
  for (int i = 0; i < probe.res_size; i++) {
    for (int j = 0; j < n_gpus; j++) {
      if (h_res[j][probe.res_array_cols * i] != 0) {
        res_count += 1;
        total_revenue += h_res[j][probe.res_array_cols * i + probe.res_idx];
      }
    }
  }
  cout << "Res Count: " << res_count << endl;
  cout << "Total Revenue: " << total_revenue << endl;

  for (int i = 0; i < n_gpus; i++) {
    sycl::free(h_res[i], general_queues[i]);
    general_queues[i].wait_and_throw();
    kernel_queues[i].wait_and_throw();
  }
  sycl::free(h_res, general_queues[0]);

  return {transfer_kernels_time, probe_kernels_time, exec_time};
}

pair<float, float> hot_output_stat(float *times, int repetitions) {
  float mean_time = 0;
  float std_time = 0;

  for (int i = 1; i < repetitions; i++) {
    mean_time += times[i];
  }
  mean_time /= (repetitions - 1);

  for (int i = 1; i < repetitions; i++) {
    std_time += (times[i] - mean_time) * (times[i] - mean_time);
  }
  std_time = sqrt(std_time / (repetitions - 1));

  float percentage_time = 100 * std_time / mean_time;

  return make_pair(mean_time, percentage_time);
}

void print_output_stats(float *transfer_kernels_times,
                        float *probe_kernels_times, float *total_times,
                        int repetitions) {
  cout << "Cold run probe transfers time: " << transfer_kernels_times[0]
       << " ms" << endl;
  cout << "Cold run probe kernels time: " << probe_kernels_times[0] << " ms"
       << endl;
  cout << "Cold run total (device alloc + transfer + kernels) time: "
       << total_times[0] << " ms" << endl;

  if (repetitions > 1) {
    cout << "Hot run:" << endl;
    pair<float, float> transfer_stats =
        hot_output_stat(transfer_kernels_times, repetitions);
    pair<float, float> probe_stats =
        hot_output_stat(probe_kernels_times, repetitions);
    pair<float, float> total_stats = hot_output_stat(total_times, repetitions);

    cout << "Hot runs: average probe transfers time (" << repetitions - 1
         << " runs): " << transfer_stats.first << " ms (+-"
         << transfer_stats.second << "%)" << endl;
    cout << "Hot runs: average probe kernels time (" << repetitions - 1
         << " runs): " << probe_stats.first << " ms (+-" << probe_stats.second
         << "%)" << endl;
    cout << "Hot runs: average total (transfer + kernels) time ("
         << repetitions - 1 << " runs): " << total_stats.first << " ms (+-"
         << total_stats.second << "%)" << endl;
  }
}

void print_host_output_stats(float *probe_kernels_times, int repetitions) {
  cout << "Cold run probe kernels time: " << probe_kernels_times[0] << " ms"
       << endl;

  if (repetitions > 1) {
    cout << "Hot run:" << endl;
    pair<float, float> probe_stats =
        hot_output_stat(probe_kernels_times, repetitions);
    cout << "Hot runs: average probe kernels time (" << repetitions - 1
         << " runs): " << probe_stats.first << " ms (+-" << probe_stats.second
         << "%)" << endl;
  }
}

template <typename T, typename R, typename S = R>
perf_times run_on_host(const BuildData<T> *builds, int n_builds,
                       const ProbeData<T, R> &probe, sycl::queue queue) {
  auto itime = std::chrono::steady_clock::now();
  R *h_res = sycl::malloc_host<R>(probe.res_size * probe.res_array_cols, queue);
  queue.memset(h_res, 0, probe.res_size * probe.res_array_cols * sizeof(R));
  queue.wait();
  T **h_build_tables = sycl::malloc_host<T *>(n_builds, queue);
  sycl::event e{};
  float probe_time = 0;
  for (int i = 0; i < n_builds; i++) {
    h_build_tables[i] = sycl::malloc_host<T>(2 * builds[i].num_slots, queue);
    queue.memset(h_build_tables[i], 0, 2 * builds[i].num_slots * sizeof(T));
    queue.wait();
    builds[i].build_function(builds[i].h_filter_col, builds[i].h_dim_key,
                             builds[i].h_dim_val, h_build_tables[i], queue, e);
    queue.wait();
  }
  probe.probe_function(probe.h_lo_data, probe.len_each_probe, h_build_tables,
                       h_res, queue, e);
  queue.wait();
  wait_and_measure_time(e, "Probe", probe_time);
  S total_revenue = 0;
  int res_count = 0;
  for (int i = 0; i < probe.res_size; i++) {
    if (h_res[probe.res_array_cols * i] != 0) {
      res_count += 1;
      total_revenue += h_res[probe.res_array_cols * i + probe.res_idx];
    }
  }
  cout << "Res Count: " << res_count << endl;
  cout << "Total Revenue: " << total_revenue << endl;
  cout << "Probe time: " << probe_time << " ms" << endl;
  for (int i = 0; i < n_builds; i++) {
    sycl::free(h_build_tables[i], queue);
  }
  sycl::free(h_build_tables, queue);
  sycl::free(h_res, queue);
  auto ftime = std::chrono::steady_clock::now();
  float exec_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(ftime - itime)
          .count();
  return {0, probe_time, exec_time};
}

template <typename T, typename R, typename S = R>
void exchange_operator_wrapper(const BuildData<T> *builds, int n_builds,
                               const ProbeData<T, R> &probe, int n_gpus,
                               int n_partitions, int repetitions,
                               sycl::queue cpu_queue) {
  unsigned long long build_items = 0;
  for (int i = 0; i < n_builds; i++) {
    build_items += builds[i].num_tuples * 2;
  }
  cout << "Build Size: " << build_items * sizeof(T) / 1024 / 1024 << " MB"
       << endl;
  cout << "Probe Size: "
       << (unsigned int)probe.len_each_probe * probe.n_probes / 1024 / 1024 *
              sizeof(T)
       << " MB" << endl;

  cout << "Repetitions: " << repetitions << endl;
  cout << "Number of GPUs: " << n_gpus << endl;
  cout << "Number of Partitions: " << n_partitions << endl;

  float *transfer_kernels_times = new float[repetitions];
  float *probe_kernels_times = new float[repetitions];
  float *total_times = new float[repetitions];

  for (int i = 0; i < repetitions; i++) {
    perf_times times;
    if (n_gpus > 0)
      times = exchange_operator<T, R, S>(builds, n_builds, probe, n_gpus,
                                         n_partitions);
    else
      times = run_on_host<T, R, S>(builds, n_builds, probe, cpu_queue);
    transfer_kernels_times[i] = times.transfer_kernels_time;
    probe_kernels_times[i] = times.probe_kernels_time;
    total_times[i] = times.total_exec_time;
  }
  if (n_gpus > 0)
    print_output_stats(transfer_kernels_times, probe_kernels_times, total_times,
                       repetitions);
  else
    print_host_output_stats(probe_kernels_times, repetitions);
}
