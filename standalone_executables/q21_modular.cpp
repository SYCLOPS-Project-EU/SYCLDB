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

double q21_modular(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev,
                   int* p_part, int* p_cat, int* p_brand, int* s_supp, int* s_reg,
                   int* d_date, int* d_year, bool* flags, uint64_t* result, int rep, bool is_intel) {
    cout << "\n--- Repetition " << rep << " Step-by-Step Breakdown ---" << endl;
    PerfTimer t; t.start();
    
    // Step 1: Init flags
    time_step("Init Flags (LO_LEN)", q, [&]() { return q.fill(flags, true, LO_LEN); }, is_intel);
    
    // Step 2: Build Part HT
    bool* ht_p = sycl::malloc_device<bool>(P_LEN, q);
    int* ht_p_val = sycl::malloc_device<int>(P_LEN, q);
    bool* p_flags = sycl::malloc_device<bool>(P_LEN, q);
    
    time_step("Fill P_Flags (P_LEN)", q, [&]() { return q.fill(p_flags, true, P_LEN); }, is_intel);
    
    time_step("Select Part (cat=1)", q, [&]() {
        SelectionKernelLiteral kernel_p(EQ, NONE, p_flags, p_flags, p_cat, 1, P_LEN);
        return q.parallel_for(P_LEN, kernel_p);
    }, is_intel);
    
    time_step("Fill Part HT", q, [&]() { return q.fill(ht_p, false, P_LEN); }, is_intel);
    
    time_step("Build Part HT Keys", q, [&]() {
        return q.parallel_for(P_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (p_flags[j]) {
                ht_p[HASH(p_part[j], (int)P_LEN, 0)] = true;
            }
        });
    }, is_intel);
    
    time_step("Fill Part HT Vals", q, [&]() { return q.fill(ht_p_val, 0, P_LEN); }, is_intel);
    
    time_step("Build Part HT Vals", q, [&]() {
        return q.parallel_for(P_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (p_flags[j]) {
                ht_p_val[HASH(p_part[j], (int)P_LEN, 0)] = p_brand[j];
            }
        });
    }, is_intel);
    
    // Step 3: Probe Part
    time_step("Probe Part", q, [&]() {
        return q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int val = lo_part[j];
                if (val >= 0 && val <= (int)P_LEN - 1) {
                    flags[j] = ht_p[HASH(val, (int)P_LEN, 0)];
                } else {
                    flags[j] = false;
                }
            }
        });
    }, is_intel);
    
    // Step 4: Build Supplier HT
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    
    time_step("Fill S_Flags (S_LEN)", q, [&]() { return q.fill(s_flags, true, S_LEN); }, is_intel);
    
    time_step("Select Supplier (region=AMERICA)", q, [&]() {
        SelectionKernelLiteral kernel_s(EQ, NONE, s_flags, s_flags, s_reg, 1, S_LEN);
        return q.parallel_for(S_LEN, kernel_s);
    }, is_intel);
    
    time_step("Fill Supp HT", q, [&]() { return q.fill(ht_s, false, S_LEN); }, is_intel);
    
    time_step("Build Supp HT Keys", q, [&]() {
        return q.parallel_for(S_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (s_flags[j]) {
                ht_s[HASH(s_supp[j], (int)S_LEN, 0)] = true;
            }
        });
    }, is_intel);
    
    // Step 5: Probe Supplier
    time_step("Probe Supplier", q, [&]() {
        return q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int val = lo_supp[j];
                if (val >= 0 && val <= (int)S_LEN - 1) {
                    flags[j] = ht_s[HASH(val, (int)S_LEN, 0)];
                } else {
                    flags[j] = false;
                }
            }
        });
    }, is_intel);
    
    // Step 6: Build Date HT
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    
    time_step("Fill D_Flags (D_LEN)", q, [&]() { return q.fill(d_flags, true, D_LEN); }, is_intel);
    
    time_step("Fill Date HT Vals", q, [&]() { return q.fill(ht_d_val, 0, D_LEN); }, is_intel);
    
    time_step("Build Date HT Vals", q, [&]() {
        return q.parallel_for(D_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (d_flags[j]) {
                ht_d_val[HASH(d_date[j], (int)D_LEN, 0)] = d_year[j];
            }
        });
    }, is_intel);
    
    // Step 7: Probe Date & Aggregate Group-By
    time_step("Fill Result Buffer (7 * 1000)", q, [&]() { return q.fill(result, 0ULL, 7 * 1000); }, is_intel);
    
    time_step("Probe Date & Aggregate Group-By", q, [&]() {
        return q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
                uint64_t j = idx[0];
                if (flags[j]) {
                    int brand = ht_p_val[HASH(lo_part[j], (int)P_LEN, 0)];
                    int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                    int group = (brand * 7 + (year - 1992)) % (7000);
                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                    sum_obj.fetch_add((uint64_t)lo_rev[j]);
                }
            });
        });
    }, is_intel);
    
    sycl::free(ht_p, q); sycl::free(ht_p_val, q); sycl::free(p_flags, q);
    sycl::free(ht_s, q); sycl::free(s_flags, q);
    sycl::free(ht_d_val, q); sycl::free(d_flags, q);
    
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
    int* h_lo_part = loadColumn<int>("lo_partkey", LO_LEN, q);
    int* h_lo_supp = loadColumn<int>("lo_suppkey", LO_LEN, q);
    int* h_lo_rev = loadColumn<int>("lo_revenue", LO_LEN, q);
    int* h_d_date = loadColumn<int>("d_datekey", D_LEN, q);
    int* h_d_year = loadColumn<int>("d_year", D_LEN, q);
    int* h_p_part = loadColumn<int>("p_partkey", P_LEN, q);
    int* h_p_cat = loadColumn<int>("p_category", P_LEN, q);
    int* h_p_brand1 = loadColumn<int>("p_brand1", P_LEN, q);
    int* h_s_supp = loadColumn<int>("s_suppkey", S_LEN, q);
    int* h_s_reg = loadColumn<int>("s_region", S_LEN, q);

    if (!h_lo_date || !h_lo_part || !h_lo_supp || !h_lo_rev || !h_d_date || !h_d_year || !h_p_part || !h_p_cat || !h_p_brand1 || !h_s_supp || !h_s_reg) {
        cerr << "Error loading columns!" << endl;
        return 1;
    }

    int* lo_date = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_part = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_supp = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_rev = sycl::malloc_device<int>(LO_LEN, q);
    int* d_date = sycl::malloc_device<int>(D_LEN, q);
    int* d_year = sycl::malloc_device<int>(D_LEN, q);
    int* p_part = sycl::malloc_device<int>(P_LEN, q);
    int* p_cat = sycl::malloc_device<int>(P_LEN, q);
    int* p_brand1 = sycl::malloc_device<int>(P_LEN, q);
    int* s_supp = sycl::malloc_device<int>(S_LEN, q);
    int* s_reg = sycl::malloc_device<int>(S_LEN, q);

    q.memcpy(lo_date, h_lo_date, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_part, h_lo_part, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_supp, h_lo_supp, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_rev, h_lo_rev, (size_t)LO_LEN * sizeof(int));
    q.memcpy(d_date, h_d_date, (size_t)D_LEN * sizeof(int));
    q.memcpy(d_year, h_d_year, (size_t)D_LEN * sizeof(int));
    q.memcpy(p_part, h_p_part, (size_t)P_LEN * sizeof(int));
    q.memcpy(p_cat, h_p_cat, (size_t)P_LEN * sizeof(int));
    q.memcpy(p_brand1, h_p_brand1, (size_t)P_LEN * sizeof(int));
    q.memcpy(s_supp, h_s_supp, (size_t)S_LEN * sizeof(int));
    q.memcpy(s_reg, h_s_reg, (size_t)S_LEN * sizeof(int));
    q.wait();

    bool* flags = sycl::malloc_device<bool>(LO_LEN, q);
    uint64_t* result = sycl::malloc_device<uint64_t>(7 * 1000, q);

    cout << "Query,Repetition,KernelTime_ms" << endl;
    for (int r = 0; r < REPETITIONS; r++) {
        double t = q21_modular(q, lo_date, lo_part, lo_supp, lo_rev, p_part, p_cat, p_brand1, s_supp, s_reg, d_date, d_year, flags, result, r + 1, is_intel);
        cout << "q21_modular," << r + 1 << "," << t << endl;
    }

    sycl::free(lo_date, q);
    sycl::free(lo_part, q);
    sycl::free(lo_supp, q);
    sycl::free(lo_rev, q);
    sycl::free(d_date, q);
    sycl::free(d_year, q);
    sycl::free(p_part, q);
    sycl::free(p_cat, q);
    sycl::free(p_brand1, q);
    sycl::free(s_supp, q);
    sycl::free(s_reg, q);
    sycl::free(flags, q);
    sycl::free(result, q);

    return 0;
}
