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

double q11_hardcoded(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_rev, int* d_date, int* d_year, unsigned long long* res) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) {
        bool sf = false;
        selection_element(sf, d_year[i], NONE, EQ, 1993);
        if (sf) build_keys_vals_element(d_date[i], d_year[i], sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
    });
    
    q.fill(res, 0ULL, 1).wait();
    PerfTimer t; t.start();
    q.parallel_for(sycl::range<1>(LO_LEN), sycl::reduction(res, sycl::plus<unsigned long long>()), [=](sycl::id<1> idx, auto& sum) {
        int year; bool sf = true;
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        selection_element(sf, lo_quant[idx], AND, LT, 25);
        selection_element(sf, lo_disc[idx], AND, GE, 1);
        selection_element(sf, lo_disc[idx], AND, LE, 3);
        if (sf) {
            sum.combine((unsigned long long)(lo_rev[idx] * lo_disc[idx]));
        }
    }).wait();
    double elapsed = t.stop();
    sycl::free(ht_d, q);
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

    unsigned long long* result = sycl::malloc_device<unsigned long long>(1, q);

    cout << "Query,Repetition,KernelTime_ms" << endl;
    for (int r = 0; r < REPETITIONS; r++) {
        double t = q11_hardcoded(q, lo_date, lo_disc, lo_quant, lo_rev, d_date, d_year, result);
        cout << "q11_hardcoded," << r + 1 << "," << t << endl;
    }

    sycl::free(lo_date, q);
    sycl::free(lo_disc, q);
    sycl::free(lo_quant, q);
    sycl::free(lo_rev, q);
    sycl::free(d_date, q);
    sycl::free(d_year, q);
    sycl::free(result, q);

    return 0;
}
