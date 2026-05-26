#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <numeric>
#include <sycl/sycl.hpp>

#undef DATA_DIR
#define DATA_DIR "/media/ssb/s100_columnar/"
#define SF 100
#include "SYCLDB/ssb/ssb_utils.hpp"
#include "kernels/selection.hpp"
#include "kernels/join.hpp"
#include "kernels/aggregation.hpp"

using namespace std;

#define REPETITIONS 10

// --- Performance Timer ---
struct PerfTimer {
    chrono::high_resolution_clock::time_point start_time;
    void start() { start_time = chrono::high_resolution_clock::now(); }
    double stop() {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

double print_step_time(const std::string& name, sycl::event e) {
    try {
        e.wait();
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        double ms = (double)(end - start) / 1000000.0;
        std::cout << "  [Kernel] " << name << ": " << ms << " ms" << endl;
        return ms;
    } catch (...) {
        return 0.0;
    }
}

template <typename F>
sycl::event time_step(const std::string& name, sycl::queue& q, F&& func, bool is_intel) {
    if (is_intel) {
        q.wait();
        auto start = std::chrono::high_resolution_clock::now();
        auto e = func();
        e.wait();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  [Kernel] " << name << ": " << ms << " ms (Host Timer)" << endl;
        return e;
    } else {
        auto e = func();
        print_step_time(name, e);
        return e;
    }
}

double q11_modular(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_rev, int* d_date, int* d_year, bool* flags, int* temp_rev, uint64_t* result, int rep, bool is_intel) {
    cout << "\n--- Repetition " << rep << " Step-by-Step Breakdown ---" << endl;
    PerfTimer t; t.start();
    
    // Step 1: Init flags
    time_step("Init Flags (LO_LEN)", q, [&]() { return q.fill(flags, true, LO_LEN); }, is_intel);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    
    time_step("Fill D_Flags (D_LEN)", q, [&]() { return q.fill(d_flags, true, D_LEN); }, is_intel);
    
    // Step 2: Selection on Date
    time_step("Select Date (year=1993)", q, [&]() {
        SelectionKernelLiteral kernel1(EQ, NONE, d_flags, d_flags, d_year, 1993, D_LEN);
        return q.parallel_for(D_LEN, kernel1);
    }, is_intel);
    
    // Step 3: Build HT Date
    time_step("Fill Date HT", q, [&]() { return q.fill(ht_d, false, D_LEN); }, is_intel);
    time_step("Build Date HT", q, [&]() {
        return q.parallel_for(D_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (d_flags[j]) {
                ht_d[HASH(d_date[j], (int)D_LEN, 0)] = true;
            }
        });
    }, is_intel);
    
    // Step 4: Probe Date
    time_step("Probe Date", q, [&]() {
        return q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int val = lo_date[j];
                if (val >= 0 && val <= 10000000) {
                    flags[j] = ht_d[HASH(val, (int)D_LEN, 0)];
                } else {
                    flags[j] = false;
                }
            }
        });
    }, is_intel);
    
    // Step 5: Select Quantity
    time_step("Select Quantity (< 25)", q, [&]() {
        SelectionKernelLiteral kernel2(LT, AND, flags, flags, lo_quant, 25, LO_LEN);
        return q.parallel_for(LO_LEN, kernel2);
    }, is_intel);
    
    // Step 6: Select Discount (>= 1)
    time_step("Select Discount (>= 1)", q, [&]() {
        SelectionKernelLiteral kernel3(GE, AND, flags, flags, lo_disc, 1, LO_LEN);
        return q.parallel_for(LO_LEN, kernel3);
    }, is_intel);
    
    // Step 7: Select Discount (<= 3)
    time_step("Select Discount (<= 3)", q, [&]() {
        SelectionKernelLiteral kernel4(LE, AND, flags, flags, lo_disc, 3, LO_LEN);
        return q.parallel_for(LO_LEN, kernel4);
    }, is_intel);
    
    // Step 8: Project Mul
    time_step("Project Mul (revenue * discount)", q, [&]() {
        return q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) temp_rev[j] = lo_rev[j] * lo_disc[j];
        });
    }, is_intel);
    
    // Step 9: Aggregate
    time_step("Fill Result Buffer", q, [&]() { return q.fill(result, 0ULL, 1); }, is_intel);
    time_step("Aggregate Profit", q, [&]() {
        return q.parallel_for(sycl::range<1>(LO_LEN), sycl::reduction(result, sycl::plus<uint64_t>()), [=](sycl::id<1> idx, auto& sum) {
            uint64_t j = idx[0];
            if (flags[j]) {
                sum.combine((uint64_t)temp_rev[j]);
            }
        });
    }, is_intel);
    
    sycl::free(ht_d, q); sycl::free(d_flags, q);
    double elapsed = t.stop();
    cout << "Total Wall-Clock Time: " << elapsed << " ms" << endl;
    return elapsed;
}

int main() {
    char* target_env = getenv("TARGET_DEVICE");
    sycl::device selected_dev;
    bool found = false;
    bool is_intel = false;

    if (target_env) {
        std::string tgt(target_env);
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        for (const auto& dev : devices) {
            auto name = dev.get_info<sycl::info::device::name>();
            if (tgt == "nvidia" && name.find("L40S") != std::string::npos) {
                selected_dev = dev; found = true; break;
            } else if (tgt == "amd" && (name.find("MI210") != std::string::npos || name.find("Instinct") != std::string::npos)) {
                selected_dev = dev; found = true; break;
            } else if (tgt == "intel" && (name.find("Flex") != std::string::npos || name.find("Data Center") != std::string::npos || name.find("Max") != std::string::npos)) {
                selected_dev = dev; found = true; is_intel = true;
                g_data_dir = "/media/ssb/s40_columnar/";
                g_lo_len = 240014460;
                g_p_len = 1200000;
                g_s_len = 80000;
                g_c_len = 1200000;
                g_d_len = 2556;
                break;
            }
        }
    }

    auto async_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception const& ex) {
                // Ignore profiling exceptions
            }
        }
    };

    sycl::queue q;
    if (found) {
        if (is_intel) {
            q = sycl::queue(selected_dev, async_handler);
        } else {
            q = sycl::queue(selected_dev, async_handler, sycl::property::queue::enable_profiling());
        }
    } else {
        q = sycl::queue(sycl::gpu_selector_v, async_handler, sycl::property::queue::enable_profiling());
    }

    cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << endl;

    // Load necessary columns only
    int* h_lo_date = loadColumn<int>("lo_orderdate", LO_LEN, q);
    int* h_lo_disc = loadColumn<int>("lo_discount", LO_LEN, q);
    int* h_lo_quant = loadColumn<int>("lo_quantity", LO_LEN, q);
    int* h_lo_rev = loadColumn<int>("lo_revenue", LO_LEN, q);
    int* h_d_date = loadColumn<int>("d_datekey", D_LEN, q);
    int* h_d_year = loadColumn<int>("d_year", D_LEN, q);

    if (!h_lo_date || !h_lo_disc || !h_lo_quant || !h_lo_rev || !h_d_date || !h_d_year) {
        cerr << "Error loading columns!" << endl;
        return 1;
    }

    int* lo_date = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_disc = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_quant = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_rev = sycl::malloc_device<int>(LO_LEN, q);
    int* d_date = sycl::malloc_device<int>(D_LEN, q);
    int* d_year = sycl::malloc_device<int>(D_LEN, q);

    q.memcpy(lo_date, h_lo_date, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_disc, h_lo_disc, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_quant, h_lo_quant, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_rev, h_lo_rev, (size_t)LO_LEN * sizeof(int));
    q.memcpy(d_date, h_d_date, (size_t)D_LEN * sizeof(int));
    q.memcpy(d_year, h_d_year, (size_t)D_LEN * sizeof(int));
    q.wait();

    bool* flags = sycl::malloc_device<bool>(LO_LEN, q);
    int* temp_rev = sycl::malloc_device<int>(LO_LEN, q);
    uint64_t* result = sycl::malloc_device<uint64_t>(1, q);

    cout << "Query,Repetition,KernelTime_ms" << endl;
    for (int r = 0; r < REPETITIONS; r++) {
        double t = q11_modular(q, lo_date, lo_disc, lo_quant, lo_rev, d_date, d_year, flags, temp_rev, result, r + 1, is_intel);
        cout << "q11_modular," << r + 1 << "," << t << endl;
    }

    sycl::free(lo_date, q);
    sycl::free(lo_disc, q);
    sycl::free(lo_quant, q);
    sycl::free(lo_rev, q);
    sycl::free(d_date, q);
    sycl::free(d_year, q);
    sycl::free(flags, q);
    sycl::free(temp_rev, q);
    sycl::free(result, q);

    return 0;
}
