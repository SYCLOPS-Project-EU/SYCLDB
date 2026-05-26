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

#define REPETITIONS 3
double g_total_internal_ms = 0;

void collect_internal_time(sycl::event e) {
    char* tgt = getenv("TARGET_DEVICE");
    if (tgt && std::string(tgt) == "intel") {
        g_total_internal_ms = 0.0;
        return;
    }
    try {
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        g_total_internal_ms += (double)(end - start) / 1000000.0;
    } catch (...) {}
}

// --- Performance Timer ---
struct PerfTimer {
    chrono::high_resolution_clock::time_point start_time;
    void start() { start_time = chrono::high_resolution_clock::now(); }
    double stop() {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

// Helper to move column to device
template<typename T>
T* to_device(T* host_ptr, uint64_t n, sycl::queue& q) {
    if (!host_ptr) return nullptr;
    T* dev_ptr = sycl::malloc_device<T>(n, q);
    q.memcpy(dev_ptr, host_ptr, n * sizeof(T)).wait();
    return dev_ptr;
}

// --- Modular Operators (Unsegmented) ---

void init_flags_modular(sycl::queue& q, bool* flags, uint64_t n) {
    auto e = q.fill(flags, true, n);
    e.wait();
    collect_internal_time(e);
}

void selection_modular(sycl::queue& q, bool* flags, int* col, comp_op op, int val, logical_op l_op, uint64_t n) {
    SelectionKernelLiteral kernel(op, l_op, flags, flags, col, val, n);
    auto e = q.parallel_for(n, kernel);
    e.wait();
    collect_internal_time(e);
}

void probe_modular(sycl::queue& q, bool* flags, int* col, bool* ht, int ht_min, int ht_max, uint64_t n) {
    int ht_len = ht_max - ht_min + 1;
    auto e = q.parallel_for(n, [=](sycl::id<1> idx) {
        uint64_t j = idx[0];
        if (flags[j]) {
            int val = col[j];
            if (val >= ht_min && val <= ht_max) {
                flags[j] = ht[HASH(val, ht_len, ht_min)];
            } else {
                flags[j] = false;
            }
        }
    });
    e.wait();
    collect_internal_time(e);
}

void project_mul_modular(sycl::queue& q, bool* flags, int* col1, int* col2, int* out_col, uint64_t n) {
    auto e = q.parallel_for(n, [=](sycl::id<1> idx) {
        uint64_t j = idx[0];
        if (flags[j]) out_col[j] = col1[j] * col2[j];
    });
    e.wait();
    collect_internal_time(e);
}

void aggregate_modular(sycl::queue& q, bool* flags, int* col, uint64_t* result, uint64_t n) {
    auto e1 = q.fill(result, 0ULL, 1);
    e1.wait();
    collect_internal_time(e1);
    auto e = q.parallel_for(sycl::range<1>(n), sycl::reduction(result, sycl::plus<uint64_t>()), [=](sycl::id<1> idx, auto& sum) {
        uint64_t j = idx[0];
        if (flags[j]) {
            sum.combine((uint64_t)col[j]);
        }
    });
    e.wait();
    collect_internal_time(e);
}

void build_keys_modular(sycl::queue& q, int* col, bool* flags, uint64_t n, bool* ht, int ht_len, int ht_min) {
    auto e1 = q.fill(ht, false, ht_len);
    e1.wait();
    collect_internal_time(e1);
    auto e = q.parallel_for(n, [=](sycl::id<1> idx) {
        uint64_t j = idx[0];
        if (flags[j]) {
            int val = col[j];
            ht[HASH(val, ht_len, ht_min)] = true;
        }
    });
    e.wait();
    collect_internal_time(e);
}

void build_keys_vals_modular(sycl::queue& q, int* col, int* vals, bool* flags, uint64_t n, int* ht, int ht_len, int ht_min) {
    auto e1 = q.fill(ht, 0, ht_len);
    e1.wait();
    collect_internal_time(e1);
    auto e = q.parallel_for(n, [=](sycl::id<1> idx) {
        uint64_t j = idx[0];
        if (flags[j]) {
            int val = col[j];
            ht[HASH(val, ht_len, ht_min)] = vals[j];
        }
    });
    e.wait();
    collect_internal_time(e);
}

// --- SSB Queries ---

double q11(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_rev, int* d_date, int* d_year, bool* flags, int* temp_rev, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, EQ, 1993, NONE, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    selection_modular(q, flags, lo_quant, LT, 25, AND, LO_LEN);
    selection_modular(q, flags, lo_disc, GE, 1, AND, LO_LEN);
    selection_modular(q, flags, lo_disc, LE, 3, AND, LO_LEN);
    project_mul_modular(q, flags, lo_rev, lo_disc, temp_rev, LO_LEN);
    aggregate_modular(q, flags, temp_rev, result, LO_LEN);
    sycl::free(ht_d, q); sycl::free(d_flags, q);
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q12(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_rev, int* d_date, int* d_year, bool* flags, int* temp_rev, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, EQ, 1994, NONE, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    selection_modular(q, flags, lo_quant, GE, 26, AND, LO_LEN);
    selection_modular(q, flags, lo_quant, LE, 35, AND, LO_LEN);
    selection_modular(q, flags, lo_disc, GE, 4, AND, LO_LEN);
    selection_modular(q, flags, lo_disc, LE, 6, AND, LO_LEN);
    project_mul_modular(q, flags, lo_rev, lo_disc, temp_rev, LO_LEN);
    aggregate_modular(q, flags, temp_rev, result, LO_LEN);
    sycl::free(ht_d, q); sycl::free(d_flags, q);
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q13(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_rev, int* d_date, int* d_year, bool* flags, int* temp_rev, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, EQ, 1994, NONE, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    selection_modular(q, flags, lo_quant, GE, 36, AND, LO_LEN);
    selection_modular(q, flags, lo_quant, LE, 40, AND, LO_LEN);
    selection_modular(q, flags, lo_disc, GE, 7, AND, LO_LEN);
    selection_modular(q, flags, lo_disc, LE, 9, AND, LO_LEN);
    project_mul_modular(q, flags, lo_rev, lo_disc, temp_rev, LO_LEN);
    aggregate_modular(q, flags, temp_rev, result, LO_LEN);
    sycl::free(ht_d, q); sycl::free(d_flags, q);
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q21(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev, int* p_part, int* p_cat, int* p_brand, int* s_supp, int* s_reg, int* d_date, int* d_year, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_p = sycl::malloc_device<bool>(P_LEN, q);
    int* ht_p_val = sycl::malloc_device<int>(P_LEN, q);
    bool* p_flags = sycl::malloc_device<bool>(P_LEN, q);
    q.fill(p_flags, true, P_LEN).wait();
    selection_modular(q, p_flags, p_cat, EQ, 1, NONE, P_LEN);
    build_keys_modular(q, p_part, p_flags, P_LEN, ht_p, P_LEN, 0);
    build_keys_vals_modular(q, p_part, p_brand, p_flags, P_LEN, ht_p_val, P_LEN, 0);
    probe_modular(q, flags, lo_part, ht_p, 0, P_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_reg, EQ, 1, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    
    auto e1 = q.fill(result, 0ULL, 7 * 1000);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int brand = ht_p_val[HASH(lo_part[j], (int)P_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (brand * 7 + (year - 1992)) % 7000;
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)lo_rev[j]);
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_p, q); sycl::free(ht_p_val, q); sycl::free(p_flags, q);
    sycl::free(ht_s, q); sycl::free(s_flags, q);
    sycl::free(ht_d_val, q); sycl::free(d_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q22(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev, int* p_part, int* p_cat, int* p_brand, int* s_supp, int* s_reg, int* d_date, int* d_year, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_p = sycl::malloc_device<bool>(P_LEN, q);
    int* ht_p_val = sycl::malloc_device<int>(P_LEN, q);
    bool* p_flags = sycl::malloc_device<bool>(P_LEN, q);
    q.fill(p_flags, true, P_LEN).wait();
    selection_modular(q, p_flags, p_cat, GE, 10, NONE, P_LEN);
    selection_modular(q, p_flags, p_cat, LE, 11, AND, P_LEN);
    build_keys_modular(q, p_part, p_flags, P_LEN, ht_p, P_LEN, 0);
    build_keys_vals_modular(q, p_part, p_brand, p_flags, P_LEN, ht_p_val, P_LEN, 0);
    probe_modular(q, flags, lo_part, ht_p, 0, P_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_reg, EQ, 1, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    
    auto e1 = q.fill(result, 0ULL, 7 * 1000);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int brand = ht_p_val[HASH(lo_part[j], (int)P_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (brand * 7 + (year - 1992)) % 7000;
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)lo_rev[j]);
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_p, q); sycl::free(ht_p_val, q); sycl::free(p_flags, q);
    sycl::free(ht_s, q); sycl::free(s_flags, q);
    sycl::free(ht_d_val, q); sycl::free(d_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q23(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev, int* p_part, int* p_brand, int* s_supp, int* s_reg, int* d_date, int* d_year, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_p = sycl::malloc_device<bool>(P_LEN, q);
    int* ht_p_val = sycl::malloc_device<int>(P_LEN, q);
    bool* p_flags = sycl::malloc_device<bool>(P_LEN, q);
    q.fill(p_flags, true, P_LEN).wait();
    selection_modular(q, p_flags, p_brand, EQ, 260, NONE, P_LEN);
    build_keys_modular(q, p_part, p_flags, P_LEN, ht_p, P_LEN, 0);
    build_keys_vals_modular(q, p_part, p_brand, p_flags, P_LEN, ht_p_val, P_LEN, 0);
    probe_modular(q, flags, lo_part, ht_p, 0, P_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_reg, EQ, 3, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    
    auto e1 = q.fill(result, 0ULL, 7 * 1000);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int brand = ht_p_val[HASH(lo_part[j], (int)P_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (brand * 7 + (year - 1992)) % 7000;
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)lo_rev[j]);
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_p, q); sycl::free(ht_p_val, q); sycl::free(p_flags, q);
    sycl::free(ht_s, q); sycl::free(s_flags, q);
    sycl::free(ht_d_val, q); sycl::free(d_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q31(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev, int* d_date, int* d_year, int* c_cust, int* c_reg, int* c_nat, int* s_supp, int* s_reg, int* s_nat, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, GE, 1992, NONE, D_LEN);
    selection_modular(q, d_flags, d_year, LE, 1997, AND, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    
    bool* ht_c = sycl::malloc_device<bool>(C_LEN, q);
    int* ht_c_val = sycl::malloc_device<int>(C_LEN, q);
    bool* c_flags = sycl::malloc_device<bool>(C_LEN, q);
    q.fill(c_flags, true, C_LEN).wait();
    selection_modular(q, c_flags, c_reg, EQ, 2, NONE, C_LEN);
    build_keys_modular(q, c_cust, c_flags, C_LEN, ht_c, C_LEN, 0);
    build_keys_vals_modular(q, c_cust, c_nat, c_flags, C_LEN, ht_c_val, C_LEN, 0);
    probe_modular(q, flags, lo_cust, ht_c, 0, C_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    int* ht_s_val = sycl::malloc_device<int>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_reg, EQ, 2, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    build_keys_vals_modular(q, s_supp, s_nat, s_flags, S_LEN, ht_s_val, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    auto e1 = q.fill(result, 0ULL, 7 * 25 * 25);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int snat = ht_s_val[HASH(lo_supp[j], (int)S_LEN, 0)];
                int cnat = ht_c_val[HASH(lo_cust[j], (int)C_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (snat * 25 * 7 + cnat * 7 + (year - 1992)) % (7 * 25 * 25);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)lo_rev[j]);
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_d, q); sycl::free(ht_d_val, q); sycl::free(d_flags, q); 
    sycl::free(ht_c, q); sycl::free(ht_c_val, q); sycl::free(c_flags, q); 
    sycl::free(ht_s, q); sycl::free(ht_s_val, q); sycl::free(s_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q32(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev, int* d_date, int* d_year, int* c_cust, int* c_nat, int* c_city, int* s_supp, int* s_nat, int* s_city, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, GE, 1992, NONE, D_LEN);
    selection_modular(q, d_flags, d_year, LE, 1997, AND, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    
    bool* ht_c = sycl::malloc_device<bool>(C_LEN, q);
    int* ht_c_val = sycl::malloc_device<int>(C_LEN, q);
    bool* c_flags = sycl::malloc_device<bool>(C_LEN, q);
    q.fill(c_flags, true, C_LEN).wait();
    selection_modular(q, c_flags, c_nat, EQ, 1, NONE, C_LEN);
    build_keys_modular(q, c_cust, c_flags, C_LEN, ht_c, C_LEN, 0);
    build_keys_vals_modular(q, c_cust, c_city, c_flags, C_LEN, ht_c_val, C_LEN, 0);
    probe_modular(q, flags, lo_cust, ht_c, 0, C_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    int* ht_s_val = sycl::malloc_device<int>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_nat, EQ, 1, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    build_keys_vals_modular(q, s_supp, s_city, s_flags, S_LEN, ht_s_val, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    auto e1 = q.fill(result, 0ULL, 7 * 250 * 250);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int scit = ht_s_val[HASH(lo_supp[j], (int)S_LEN, 0)];
                int ccit = ht_c_val[HASH(lo_cust[j], (int)C_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (scit * 250 * 7 + ccit * 7 + (year - 1992)) % (7 * 250 * 250);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)lo_rev[j]);
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_d, q); sycl::free(ht_d_val, q); sycl::free(d_flags, q); 
    sycl::free(ht_c, q); sycl::free(ht_c_val, q); sycl::free(c_flags, q); 
    sycl::free(ht_s, q); sycl::free(ht_s_val, q); sycl::free(s_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q33(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev, int* d_date, int* d_year, int* c_cust, int* c_nat, int* c_city, int* s_supp, int* s_nat, int* s_city, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, GE, 1992, NONE, D_LEN);
    selection_modular(q, d_flags, d_year, LE, 1997, AND, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    
    bool* ht_c = sycl::malloc_device<bool>(C_LEN, q);
    int* ht_c_val = sycl::malloc_device<int>(C_LEN, q);
    bool* c_flags = sycl::malloc_device<bool>(C_LEN, q);
    q.fill(c_flags, true, C_LEN).wait();
    selection_modular(q, c_flags, c_nat, EQ, 10, NONE, C_LEN);
    build_keys_modular(q, c_cust, c_flags, C_LEN, ht_c, C_LEN, 0);
    build_keys_vals_modular(q, c_cust, c_city, c_flags, C_LEN, ht_c_val, C_LEN, 0);
    probe_modular(q, flags, lo_cust, ht_c, 0, C_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    int* ht_s_val = sycl::malloc_device<int>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_nat, EQ, 10, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    build_keys_vals_modular(q, s_supp, s_city, s_flags, S_LEN, ht_s_val, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    auto e1 = q.fill(result, 0ULL, 7 * 250 * 250);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int scit = ht_s_val[HASH(lo_supp[j], (int)S_LEN, 0)];
                int ccit = ht_c_val[HASH(lo_cust[j], (int)C_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (scit * 250 * 7 + ccit * 7 + (year - 1992)) % (7 * 250 * 250);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)lo_rev[j]);
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_d, q); sycl::free(ht_d_val, q); sycl::free(d_flags, q); 
    sycl::free(ht_c, q); sycl::free(ht_c_val, q); sycl::free(c_flags, q); 
    sycl::free(ht_s, q); sycl::free(ht_s_val, q); sycl::free(s_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q34(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev, int* d_date, int* d_year, int* c_cust, int* c_nat, int* c_city, int* s_supp, int* s_nat, int* s_city, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, EQ, 1995, NONE, D_LEN); // Wait, Q3.4 uses EQ 1995? Let's check original modular_ssb.
    // In modular_ssb.cpp, q34 uses: d_year EQ 1995.
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    
    bool* ht_c = sycl::malloc_device<bool>(C_LEN, q);
    int* ht_c_val = sycl::malloc_device<int>(C_LEN, q);
    bool* c_flags = sycl::malloc_device<bool>(C_LEN, q);
    q.fill(c_flags, true, C_LEN).wait();
    selection_modular(q, c_flags, c_nat, EQ, 10, NONE, C_LEN); // Original modular_ssb: EQ 10
    build_keys_modular(q, c_cust, c_flags, C_LEN, ht_c, C_LEN, 0);
    build_keys_vals_modular(q, c_cust, c_city, c_flags, C_LEN, ht_c_val, C_LEN, 0);
    probe_modular(q, flags, lo_cust, ht_c, 0, C_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    int* ht_s_val = sycl::malloc_device<int>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_nat, EQ, 10, NONE, S_LEN); // Original modular_ssb: EQ 10
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    build_keys_vals_modular(q, s_supp, s_city, s_flags, S_LEN, ht_s_val, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    auto e1 = q.fill(result, 0ULL, 7 * 250 * 250);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int scit = ht_s_val[HASH(lo_supp[j], (int)S_LEN, 0)];
                int ccit = ht_c_val[HASH(lo_cust[j], (int)C_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (scit * 250 * 7 + ccit * 7 + (year - 1992)) % (7 * 250 * 250);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)lo_rev[j]);
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_d, q); sycl::free(ht_d_val, q); sycl::free(d_flags, q); 
    sycl::free(ht_c, q); sycl::free(ht_c_val, q); sycl::free(c_flags, q); 
    sycl::free(ht_s, q); sycl::free(ht_s_val, q); sycl::free(s_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q41(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_cust, int* lo_rev, int* lo_scost, int* d_date, int* d_year, int* c_cust, int* c_reg, int* c_nat, int* s_supp, int* s_reg, int* p_part, int* p_mfgr, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, EQ, 1997, NONE, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    
    bool* ht_c = sycl::malloc_device<bool>(C_LEN, q);
    int* ht_c_val = sycl::malloc_device<int>(C_LEN, q);
    bool* c_flags = sycl::malloc_device<bool>(C_LEN, q);
    q.fill(c_flags, true, C_LEN).wait();
    selection_modular(q, c_flags, c_reg, EQ, 1, NONE, C_LEN);
    build_keys_modular(q, c_cust, c_flags, C_LEN, ht_c, C_LEN, 0);
    build_keys_vals_modular(q, c_cust, c_nat, c_flags, C_LEN, ht_c_val, C_LEN, 0);
    probe_modular(q, flags, lo_cust, ht_c, 0, C_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_reg, EQ, 1, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    bool* ht_p = sycl::malloc_device<bool>(P_LEN, q);
    bool* p_flags = sycl::malloc_device<bool>(P_LEN, q);
    q.fill(p_flags, true, P_LEN).wait();
    selection_modular(q, p_flags, p_mfgr, GE, 1, NONE, P_LEN);
    selection_modular(q, p_flags, p_mfgr, LE, 2, AND, P_LEN);
    build_keys_modular(q, p_part, p_flags, P_LEN, ht_p, P_LEN, 0);
    probe_modular(q, flags, lo_part, ht_p, 0, P_LEN - 1, LO_LEN);
    
    auto e1 = q.fill(result, 0ULL, 7 * 25);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int cnat = ht_c_val[HASH(lo_cust[j], (int)C_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = (cnat * 7 + (year - 1992)) % (7 * 25);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)(lo_rev[j] - lo_scost[j]));
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_d, q); sycl::free(ht_d_val, q); sycl::free(d_flags, q); 
    sycl::free(ht_c, q); sycl::free(ht_c_val, q); sycl::free(c_flags, q); 
    sycl::free(ht_s, q); sycl::free(s_flags, q); 
    sycl::free(ht_p, q); sycl::free(p_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q42(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_cust, int* lo_rev, int* lo_scost, int* d_date, int* d_year, int* c_cust, int* c_reg, int* s_supp, int* s_reg, int* s_nat, int* p_part, int* p_mfgr, int* p_cat, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, GE, 1997, NONE, D_LEN);
    selection_modular(q, d_flags, d_year, LE, 1998, AND, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    
    bool* ht_c = sycl::malloc_device<bool>(C_LEN, q);
    bool* c_flags = sycl::malloc_device<bool>(C_LEN, q);
    q.fill(c_flags, true, C_LEN).wait();
    selection_modular(q, c_flags, c_reg, EQ, 1, NONE, C_LEN);
    build_keys_modular(q, c_cust, c_flags, C_LEN, ht_c, C_LEN, 0);
    probe_modular(q, flags, lo_cust, ht_c, 0, C_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    int* ht_s_val = sycl::malloc_device<int>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_reg, EQ, 1, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    build_keys_vals_modular(q, s_supp, s_nat, s_flags, S_LEN, ht_s_val, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    bool* ht_p = sycl::malloc_device<bool>(P_LEN, q);
    int* ht_p_val = sycl::malloc_device<int>(P_LEN, q);
    bool* p_flags = sycl::malloc_device<bool>(P_LEN, q);
    q.fill(p_flags, true, P_LEN).wait();
    selection_modular(q, p_flags, p_mfgr, GE, 1, NONE, P_LEN);
    selection_modular(q, p_flags, p_mfgr, LE, 2, AND, P_LEN);
    build_keys_modular(q, p_part, p_flags, P_LEN, ht_p, P_LEN, 0);
    build_keys_vals_modular(q, p_part, p_cat, p_flags, P_LEN, ht_p_val, P_LEN, 0);
    probe_modular(q, flags, lo_part, ht_p, 0, P_LEN - 1, LO_LEN);
    
    auto e1 = q.fill(result, 0ULL, 7 * 25 * 25);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int snat = ht_s_val[HASH(lo_supp[j], (int)S_LEN, 0)];
                int cat = ht_p_val[HASH(lo_part[j], (int)P_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = ((year - 1992) * 25 * 25 + snat * 25 + cat) % (7 * 25 * 25);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)(lo_rev[j] - lo_scost[j]));
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_d, q); sycl::free(ht_d_val, q); sycl::free(d_flags, q); 
    sycl::free(ht_c, q); sycl::free(c_flags, q); 
    sycl::free(ht_s, q); sycl::free(ht_s_val, q); sycl::free(s_flags, q); 
    sycl::free(ht_p, q); sycl::free(ht_p_val, q); sycl::free(p_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
    return elapsed;
}

double q43(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_cust, int* lo_rev, int* lo_scost, int* d_date, int* d_year, int* c_cust, int* c_reg, int* s_supp, int* s_nat, int* s_city, int* p_part, int* p_cat, int* p_brand, bool* flags, uint64_t* result) {
    g_total_internal_ms = 0;
    PerfTimer t; t.start();
    init_flags_modular(q, flags, LO_LEN);
    
    bool* ht_d = sycl::malloc_device<bool>(D_LEN, q);
    int* ht_d_val = sycl::malloc_device<int>(D_LEN, q);
    bool* d_flags = sycl::malloc_device<bool>(D_LEN, q);
    q.fill(d_flags, true, D_LEN).wait();
    selection_modular(q, d_flags, d_year, GE, 1997, NONE, D_LEN);
    selection_modular(q, d_flags, d_year, LE, 1998, AND, D_LEN);
    build_keys_modular(q, d_date, d_flags, D_LEN, ht_d, D_LEN, 0);
    build_keys_vals_modular(q, d_date, d_year, d_flags, D_LEN, ht_d_val, D_LEN, 0);
    probe_modular(q, flags, lo_date, ht_d, 0, 10000000, LO_LEN);
    
    bool* ht_c = sycl::malloc_device<bool>(C_LEN, q);
    bool* c_flags = sycl::malloc_device<bool>(C_LEN, q);
    q.fill(c_flags, true, C_LEN).wait();
    selection_modular(q, c_flags, c_reg, EQ, 1, NONE, C_LEN);
    build_keys_modular(q, c_cust, c_flags, C_LEN, ht_c, C_LEN, 0);
    probe_modular(q, flags, lo_cust, ht_c, 0, C_LEN - 1, LO_LEN);
    
    bool* ht_s = sycl::malloc_device<bool>(S_LEN, q);
    int* ht_s_val = sycl::malloc_device<int>(S_LEN, q);
    bool* s_flags = sycl::malloc_device<bool>(S_LEN, q);
    q.fill(s_flags, true, S_LEN).wait();
    selection_modular(q, s_flags, s_nat, EQ, 24, NONE, S_LEN);
    build_keys_modular(q, s_supp, s_flags, S_LEN, ht_s, S_LEN, 0);
    build_keys_vals_modular(q, s_supp, s_city, s_flags, S_LEN, ht_s_val, S_LEN, 0);
    probe_modular(q, flags, lo_supp, ht_s, 0, S_LEN - 1, LO_LEN);
    
    bool* ht_p = sycl::malloc_device<bool>(P_LEN, q);
    int* ht_p_val = sycl::malloc_device<int>(P_LEN, q);
    bool* p_flags = sycl::malloc_device<bool>(P_LEN, q);
    q.fill(p_flags, true, P_LEN).wait();
    selection_modular(q, p_flags, p_cat, EQ, 3, NONE, P_LEN);
    build_keys_modular(q, p_part, p_flags, P_LEN, ht_p, P_LEN, 0);
    build_keys_vals_modular(q, p_part, p_brand, p_flags, P_LEN, ht_p_val, P_LEN, 0);
    probe_modular(q, flags, lo_part, ht_p, 0, P_LEN - 1, LO_LEN);
    
    auto e1 = q.fill(result, 0ULL, 7 * 250 * 1000);
    e1.wait();
    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
            uint64_t j = idx[0];
            if (flags[j]) {
                int scit = ht_s_val[HASH(lo_supp[j], (int)S_LEN, 0)];
                int brand = ht_p_val[HASH(lo_part[j], (int)P_LEN, 0)];
                int year = ht_d_val[HASH(lo_date[j], (int)D_LEN, 0)];
                int group = ((year - 1992) * 250 * 1000 + scit * 1000 + brand) % (7 * 250 * 1000);
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(result[group]);
                sum_obj.fetch_add((uint64_t)(lo_rev[j] - lo_scost[j]));
            }
        });
    });
    e.wait();
    collect_internal_time(e);
    
    sycl::free(ht_d, q); sycl::free(ht_d_val, q); sycl::free(d_flags, q); 
    sycl::free(ht_c, q); sycl::free(c_flags, q); 
    sycl::free(ht_s, q); sycl::free(ht_s_val, q); sycl::free(s_flags, q); 
    sycl::free(ht_p, q); sycl::free(ht_p_val, q); sycl::free(p_flags, q);
    
    double elapsed = t.stop();
    cout << q.get_device().get_info<sycl::info::device::name>() << ", " << elapsed << " ms, (InternalTimeKernel: " << g_total_internal_ms << " ms), ";
    g_total_internal_ms = 0;
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

    // Load Columns
    int* h_lo_date = loadColumn<int>("lo_orderdate", LO_LEN, q);
    int* h_lo_cust = loadColumn<int>("lo_custkey", LO_LEN, q);
    int* h_lo_supp = loadColumn<int>("lo_suppkey", LO_LEN, q);
    int* h_lo_part = loadColumn<int>("lo_partkey", LO_LEN, q);
    int* h_lo_rev = loadColumn<int>("lo_revenue", LO_LEN, q);
    int* h_lo_disc = loadColumn<int>("lo_discount", LO_LEN, q);
    int* h_lo_quant = loadColumn<int>("lo_quantity", LO_LEN, q);
    int* h_lo_price = loadColumn<int>("lo_extendedprice", LO_LEN, q);
    int* h_lo_scost = loadColumn<int>("lo_supplycost", LO_LEN, q);
    int* h_d_date = loadColumn<int>("d_datekey", D_LEN, q);
    int* h_d_year = loadColumn<int>("d_year", D_LEN, q);
    int* h_p_part = loadColumn<int>("p_partkey", P_LEN, q);
    int* h_p_cat = loadColumn<int>("p_category", P_LEN, q);
    int* h_p_brand1 = loadColumn<int>("p_brand1", P_LEN, q);
    int* h_p_mfgr = loadColumn<int>("p_mfgr", P_LEN, q);
    int* h_s_supp = loadColumn<int>("s_suppkey", S_LEN, q);
    int* h_s_reg = loadColumn<int>("s_region", S_LEN, q);
    int* h_s_nat = loadColumn<int>("s_nation", S_LEN, q);
    int* h_s_city = loadColumn<int>("s_city", S_LEN, q);
    int* h_c_cust = loadColumn<int>("c_custkey", C_LEN, q);
    int* h_c_reg = loadColumn<int>("c_region", C_LEN, q);
    int* h_c_nat = loadColumn<int>("c_nation", C_LEN, q);
    int* h_c_city = loadColumn<int>("c_city", C_LEN, q);

    if (!h_lo_date || !h_lo_cust || !h_lo_supp || !h_lo_part || !h_lo_rev || !h_lo_disc || !h_lo_quant || !h_lo_price || !h_lo_scost || !h_d_date || !h_d_year || !h_p_part || !h_p_cat || !h_p_brand1 || !h_p_mfgr || !h_s_supp || !h_s_reg || !h_s_nat || !h_s_city || !h_c_cust || !h_c_reg || !h_c_nat || !h_c_city) {
        cerr << "Error loading columns!" << endl; return 1;
    }

    int* lo_date = to_device(h_lo_date, LO_LEN, q);
    int* lo_cust = to_device(h_lo_cust, LO_LEN, q);
    int* lo_supp = to_device(h_lo_supp, LO_LEN, q);
    int* lo_part = to_device(h_lo_part, LO_LEN, q);
    int* lo_rev = to_device(h_lo_rev, LO_LEN, q);
    int* lo_disc = to_device(h_lo_disc, LO_LEN, q);
    int* lo_quant = to_device(h_lo_quant, LO_LEN, q);
    int* lo_price = to_device(h_lo_price, LO_LEN, q);
    int* lo_scost = to_device(h_lo_scost, LO_LEN, q);
    int* d_date = to_device(h_d_date, D_LEN, q);
    int* d_year = to_device(h_d_year, D_LEN, q);
    int* p_part = to_device(h_p_part, P_LEN, q);
    int* p_cat = to_device(h_p_cat, P_LEN, q);
    int* p_brand1 = to_device(h_p_brand1, P_LEN, q);
    int* p_mfgr = to_device(h_p_mfgr, P_LEN, q);
    int* s_supp = to_device(h_s_supp, S_LEN, q);
    int* s_reg = to_device(h_s_reg, S_LEN, q);
    int* s_nat = to_device(h_s_nat, S_LEN, q);
    int* s_city = to_device(h_s_city, S_LEN, q);
    int* c_cust = to_device(h_c_cust, C_LEN, q);
    int* c_reg = to_device(h_c_reg, C_LEN, q);
    int* c_nat = to_device(h_c_nat, C_LEN, q);
    int* c_city = to_device(h_c_city, C_LEN, q);

    bool* flags = sycl::malloc_device<bool>(LO_LEN, q);
    int* temp_rev = sycl::malloc_device<int>(LO_LEN, q);
    // Allocate max size result table needed by Q4.3 (7 * 250 * 1000)
    uint64_t* result = sycl::malloc_device<uint64_t>(7 * 250 * 1000, q);

    cout << "Query,Repetition,Device,WallTime,InternalTime" << endl;

    for (int r = 0; r < REPETITIONS; r++) {
        cout << "q11," << r + 1 << ","; q11(q, lo_date, lo_disc, lo_quant, lo_rev, d_date, d_year, flags, temp_rev, result); cout << endl;
        cout << "q12," << r + 1 << ","; q12(q, lo_date, lo_disc, lo_quant, lo_rev, d_date, d_year, flags, temp_rev, result); cout << endl;
        cout << "q13," << r + 1 << ","; q13(q, lo_date, lo_disc, lo_quant, lo_rev, d_date, d_year, flags, temp_rev, result); cout << endl;
        cout << "q21," << r + 1 << ","; q21(q, lo_date, lo_part, lo_supp, lo_rev, p_part, p_cat, p_brand1, s_supp, s_reg, d_date, d_year, flags, result); cout << endl;
        cout << "q22," << r + 1 << ","; q22(q, lo_date, lo_part, lo_supp, lo_rev, p_part, p_cat, p_brand1, s_supp, s_reg, d_date, d_year, flags, result); cout << endl;
        cout << "q23," << r + 1 << ","; q23(q, lo_date, lo_part, lo_supp, lo_rev, p_part, p_brand1, s_supp, s_reg, d_date, d_year, flags, result); cout << endl;
        cout << "q31," << r + 1 << ","; q31(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_cust, c_reg, c_nat, s_supp, s_reg, s_nat, flags, result); cout << endl;
        cout << "q32," << r + 1 << ","; q32(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_cust, c_nat, c_city, s_supp, s_nat, s_city, flags, result); cout << endl;
        cout << "q33," << r + 1 << ","; q33(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_cust, c_nat, c_city, s_supp, s_nat, s_city, flags, result); cout << endl;
        cout << "q34," << r + 1 << ","; q34(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_cust, c_nat, c_city, s_supp, s_nat, s_city, flags, result); cout << endl;
        cout << "q41," << r + 1 << ","; q41(q, lo_date, lo_part, lo_supp, lo_cust, lo_rev, lo_scost, d_date, d_year, c_cust, c_reg, c_nat, s_supp, s_reg, p_part, p_mfgr, flags, result); cout << endl;
        cout << "q42," << r + 1 << ","; q42(q, lo_date, lo_part, lo_supp, lo_cust, lo_rev, lo_scost, d_date, d_year, c_cust, c_reg, s_supp, s_reg, s_nat, p_part, p_mfgr, p_cat, flags, result); cout << endl;
        cout << "q43," << r + 1 << ","; q43(q, lo_date, lo_part, lo_supp, lo_cust, lo_rev, lo_scost, d_date, d_year, c_cust, c_reg, s_supp, s_nat, s_city, p_part, p_cat, p_brand1, flags, result); cout << endl;
    }

    return 0;
}
