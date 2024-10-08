#include <sycl/sycl.hpp>

int N = 1 << 24;

void wait_and_measure_time(sycl::event e, std::string name) {
  e.wait();
  const auto start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
  float time = (end - start) / 1e6;
  std::cout << name << " took: " << time << " ms" << std::endl;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling()}};
    int *coordinates = sycl::malloc_shared<int>(N, q);
    int *profits = sycl::malloc_shared<int>(N, q);
    for (int i = 0; i < N; i++) {
        coordinates[i] = i % 100 + 100;
        profits[i] = i;
    }
    // each result contains a unique coordinate
    // and the sum of the profits in this coordinate
    // the sum is (potentially) occupying 2 slots
    int *res = sycl::malloc_shared<int>(4*100, q);
    q.memset(res, 0, 4*100*sizeof(int)).wait();
    sycl::event e;
    e = q.parallel_for(N, [=](sycl::id<1> idx) {
        int hash = coordinates[idx] % 100;
        res[hash*4] = coordinates[idx];
        auto sum_obj =
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::global_space>(
                *reinterpret_cast<unsigned long long *>(&res[hash * 4 + 2]));
        sum_obj.fetch_add((unsigned long long)profits[idx]);
    });
    wait_and_measure_time(e, "Kernel1");
}