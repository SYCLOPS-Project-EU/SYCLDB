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
                    sycl::queue &q) {
  return q.parallel_for(num_items, [=](sycl::id<1> idx) {
    out[idx] = 2 * in1[idx] + 3 * in2[idx];
  });
}

sycl::event project(sycl::buffer<float, 1> &in1, sycl::buffer<float, 1> &in2,
                    sycl::buffer<float, 1> &out, int num_items,
                    sycl::queue &q) {
  return q.submit([&](sycl::handler &cgh) {
    auto in1_acc = in1.get_access<sycl::access::mode::read>(cgh);
    auto in2_acc = in2.get_access<sycl::access::mode::read>(cgh);
    auto out_acc = out.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(num_items, [=](sycl::id<1> idx) {
      out_acc[idx] = 2 * in1_acc[idx] + 3 * in2_acc[idx];
    });
  });
}

int main(int argc, char **argv) {
  int num_items = 1 << 22;
  int target_device = 1;
  int repetitions = 10;
  
  int c;
  while ((c = getopt(argc, argv, "n:t:r:")) != -1) {
    switch (c) {
    case 'n':
      num_items = atoi(optarg);
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
    sycl::event ev_project = project(d_in1, d_in2, d_out, num_items, q);
    wait_and_measure_time(ev_project, "Project", elapsed_project);
    
    cout << "Repetition " << r << " | Project time: " << elapsed_project << " ms" << endl;
  }

  sycl::free(d_in1, q);
  sycl::free(d_in2, q);
  sycl::free(d_out, q);

  sycl::free(h_in1, cpu_queue);
  sycl::free(h_in2, cpu_queue);

  return 0;
}
