#include <getopt.h>
#include <sycl/sycl.hpp>

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

sycl::event project(float *in1, float *in2, float *out, int num_items,
                    sycl::queue &q, int mode) {
  if (q.get_device().is_cpu() && mode == 1) {
    return q.parallel_for(sycl::range<1>(num_items / 16), [=](sycl::id<1> idx) {
      size_t offset = idx[0] * 16;
      sycl::vec<float, 16> v1, v2;
      v1.load(0, sycl::global_ptr<const float>(static_cast<const float*>(in1 + offset)));
      v2.load(0, sycl::global_ptr<const float>(static_cast<const float*>(in2 + offset)));
      v1 *= 2.0f;
      v2 *= 3.0f;
      sycl::vec<float, 16> res = v1 + v2;
      res.store(0, sycl::global_ptr<float>(out + offset));
    });
  } else {
    return q.parallel_for(num_items, [=](sycl::id<1> idx) {
      out[idx] = 2 * in1[idx] + 3 * in2[idx];
    });
  }
}

int main(int argc, char **argv) {
  int num_items = 1 << 22;
  int target_device = 1;
  int repetitions = 10;
  int mode = 0;
  
  int c;
  while ((c = getopt(argc, argv, "n:t:r:m:")) != -1) {
    switch (c) {
    case 'n': num_items = atoi(optarg); break;
    case 't': target_device = atoi(optarg); break;
    case 'r': repetitions = atoi(optarg); break;
    case 'm': mode = atoi(optarg); break;
    default: abort();
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

  cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << " (Mode: " << (mode == 1 ? "New/CPU-Opt" : "Old/Default") << ")\n";

  sycl::queue cpu_queue{
      sycl::cpu_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling()}};

  float *h_in1 = NULL;
  float *h_in2 = NULL;
  generateUniformCPU(h_in1, h_in2, num_items, cpu_queue);

  float *d_in1 = sycl::malloc_device<float>(num_items, q);
  float *d_in2 = sycl::malloc_device<float>(num_items, q);
  float *d_out = sycl::malloc_device<float>(num_items, q);

  q.memcpy(d_in1, h_in1, num_items * sizeof(float)).wait();
  q.memcpy(d_in2, h_in2, num_items * sizeof(float)).wait();
  
  cout << "Query: project (standalone)" << endl;
  
  for (int r = 0; r < repetitions; r++) {
    float elapsed_project = 0;
    sycl::event ev_project = project(d_in1, d_in2, d_out, num_items, q, mode);
    wait_and_measure_time(ev_project, "Project", elapsed_project);
    
    cout << "Repetition " << r << " | Project time: " << elapsed_project << " ms" << endl;
  }

  sycl::free(d_in1, q);
  sycl::free(d_in2, q);
  sycl::free(d_out, q);

  sycl::free(h_in1, cpu_queue);
  h_in1 = nullptr;
  sycl::free(h_in2, cpu_queue);
  h_in2 = nullptr;

  return 0;
}
