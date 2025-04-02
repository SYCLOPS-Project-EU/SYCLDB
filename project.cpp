#include <getopt.h>
#include <sycl/sycl.hpp>

#include "exchange.hpp"
#include "utils/generator.h"

using namespace std;

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
  int buffers = 0;
  int num_gpus = 1;
  int num_partitions = 1;
  int repetitions = 10;
  int c;
  while ((c = getopt(argc, argv, "n:b:g:p:r:")) != -1) {
    switch (c) {
    case 'n':
      num_items = atoi(optarg);
      break;
    case 'b':
      buffers = atoi(optarg);
      break;
    case 'g':
      num_gpus = atoi(optarg);
      break;
    case 'p':
      num_partitions = atoi(optarg);
      break;
    case 'r':
      repetitions = atoi(optarg);
      break;
    default:
      abort();
    }
  }

  sycl::queue cpu_queue{
      sycl::cpu_selector_v,
      sycl::property_list{sycl::property::queue::enable_profiling()}};

  ProbeData<float, float> prob;
  prob.n_probes = 2;
  prob.h_lo_data = new float *[prob.n_probes];
  prob.h_lo_data[0] = NULL;
  prob.h_lo_data[1] = NULL;
  generateUniformCPU(prob.h_lo_data[0], prob.h_lo_data[1], num_items,
                     cpu_queue);
  prob.len_each_probe = num_items;
  prob.res_size = num_items;
  prob.res_array_cols = 1;
  prob.res_idx = 0;
  if (!buffers) {
    prob.probe_function = [&](float **probe_data, int partition_len, float **,
                              float *res, sycl::queue queue,
                              sycl::event &event) {
      event = project(probe_data[0], probe_data[1], res, partition_len, queue);
    };
    exchange_operator_wrapper<float, float>(
        NULL, 0, prob, num_gpus, num_partitions, repetitions, cpu_queue);
  } else {
    sycl::buffer<float, 1> buf_in1{prob.h_lo_data[0], num_items};
    sycl::buffer<float, 1> buf_in2{prob.h_lo_data[1], num_items};
    sycl::buffer<float, 1> buf_out{num_items};
    sycl::queue q =
        (num_gpus >= 1)
            ? sycl::queue{sycl::gpu_selector_v,
                          sycl::property_list{
                              sycl::property::queue::enable_profiling()}}
            : cpu_queue;
    for (int i = 0; i < repetitions; i++) {
      sycl::event event = project(buf_in1, buf_in2, buf_out, num_items, q);
      float elapsed = 0;
      wait_and_measure_time(event, "Project", elapsed);
    }
  }

  return 0;
}
