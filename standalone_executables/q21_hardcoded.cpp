#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <numeric>
#include <sycl/sycl.hpp>

#undef DATA_DIR
#define DATA_DIR "/media/ssb/s100_columnar/"
#define SF 100
#include "SYCLDB/include/kernels.hpp"
#include "SYCLDB/ssb/ssb_utils.hpp"

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

// Helper to allocate hash tables
int* alloc_ht(int len, sycl::queue& q) {
    int* ht = sycl::malloc_device<int>(2 * len, q);
    q.fill(ht, 0, 2 * len).wait();
    return ht;
}

double q21_hardcoded(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev,
                     int* d_date, int* d_year, int* p_cat, int* p_part, int* p_brand, int* s_reg, int* s_supp, unsigned long long* res) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_p = alloc_ht(P_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { build_keys_vals_element(d_date[i], d_year[i], true, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(P_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, p_cat[i], NONE, EQ, 1); if (sf) build_keys_vals_element(p_part[i], p_brand[i], sf, P_LEN, ht_p, P_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_reg[i], NONE, EQ, 1); if (sf) build_keys_element(s_supp[i], sf, S_LEN, ht_s, S_LEN, 0); });
    
    q.fill(res, 0ULL, 7 * 1000).wait();
    PerfTimer t; t.start();
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int brand, year; bool sf = true;
        probe_keys_vals_element(lo_part[idx], brand, sf, P_LEN, ht_p, P_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        probe_keys_element(lo_supp[idx], sf, S_LEN, ht_s, S_LEN, 0);
        if (sf) {
            int group = (brand * 7 + (year - 1992)) % (7000);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)lo_rev[idx]);
        }
    }).wait();
    double elapsed = t.stop();
    
    sycl::free(ht_d, q); sycl::free(ht_p, q); sycl::free(ht_s, q);
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

    unsigned long long* result = sycl::malloc_device<unsigned long long>(7 * 1000, q);

    cout << "Query,Repetition,KernelTime_ms" << endl;
    for (int r = 0; r < REPETITIONS; r++) {
        double t = q21_hardcoded(q, lo_date, lo_part, lo_supp, lo_rev, d_date, d_year, p_cat, p_part, p_brand1, s_reg, s_supp, result);
        cout << "q21_hardcoded," << r + 1 << "," << t << endl;
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
    sycl::free(result, q);

    return 0;
}
