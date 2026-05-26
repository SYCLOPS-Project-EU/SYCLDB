
#include "SYCLDB/include/kernels.hpp"

// Override DATA_DIR and length constants for SF100
#undef DATA_DIR
#define DATA_DIR "/media/ssb/s100_columnar/"

#define SF 100
#include "SYCLDB/ssb/ssb_utils.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

#define REPETITIONS 10

// --- Performance Timer ---
struct PerfTimer {
    chrono::high_resolution_clock::time_point s;
    PerfTimer() { s = chrono::high_resolution_clock::now(); }
    double ms() { return chrono::duration<double, milli>(chrono::high_resolution_clock::now() - s).count(); }
};

// --- Dimension Hash Table Helpers ---
int* alloc_ht(size_t slots, sycl::queue& q) {
    int* ht = sycl::malloc_device<int>(2 * slots, q);
    q.memset(ht, 0, 2 * slots * sizeof(int)).wait();
    return ht;
}

// --- Query Implementations (Full Set) ---

double q11(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_price) {
    unsigned long long* res = sycl::malloc_device<unsigned long long>(1, q);
    q.memset(res, 0, 8).wait();
    PerfTimer t;
    q.parallel_for(sycl::range<1>(LO_LEN), sycl::reduction(res, sycl::plus<unsigned long long>()), [=](sycl::id<1> idx, auto& sum) {
        bool sf = false;
        selection_element(sf, lo_date[idx], NONE, GT, 19930000);
        selection_element(sf, lo_date[idx], AND, LT, 19940000);
        selection_element(sf, lo_quant[idx], AND, LT, 25);
        selection_element(sf, lo_disc[idx], AND, GE, 1);
        selection_element(sf, lo_disc[idx], AND, LE, 3);
        if (sf) {
            sum.combine((unsigned long long)(lo_disc[idx] * lo_price[idx]));
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(res, q);
    return elapsed;
}

double q12(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_price) {
    unsigned long long* res = sycl::malloc_device<unsigned long long>(1, q);
    q.memset(res, 0, 8).wait();
    PerfTimer t;
    q.parallel_for(sycl::range<1>(LO_LEN), sycl::reduction(res, sycl::plus<unsigned long long>()), [=](sycl::id<1> idx, auto& sum) {
        bool sf = false;
        selection_element(sf, lo_date[idx], NONE, GE, 19940101);
        selection_element(sf, lo_date[idx], AND, LE, 19940131);
        selection_element(sf, lo_quant[idx], AND, GE, 26);
        selection_element(sf, lo_quant[idx], AND, LE, 35);
        selection_element(sf, lo_disc[idx], AND, GE, 4);
        selection_element(sf, lo_disc[idx], AND, LE, 6);
        if (sf) {
            sum.combine((unsigned long long)(lo_disc[idx] * lo_price[idx]));
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(res, q);
    return elapsed;
}

double q13(sycl::queue& q, int* lo_date, int* lo_disc, int* lo_quant, int* lo_price) {
    unsigned long long* res = sycl::malloc_device<unsigned long long>(1, q);
    q.memset(res, 0, 8).wait();
    PerfTimer t;
    q.parallel_for(sycl::range<1>(LO_LEN), sycl::reduction(res, sycl::plus<unsigned long long>()), [=](sycl::id<1> idx, auto& sum) {
        bool sf = false;
        selection_element(sf, lo_date[idx], NONE, GE, 19940204);
        selection_element(sf, lo_date[idx], AND, LE, 19940210);
        selection_element(sf, lo_quant[idx], AND, GE, 26);
        selection_element(sf, lo_quant[idx], AND, LE, 35);
        selection_element(sf, lo_disc[idx], AND, GE, 5);
        selection_element(sf, lo_disc[idx], AND, LE, 7);
        if (sf) {
            sum.combine((unsigned long long)(lo_disc[idx] * lo_price[idx]));
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(res, q);
    return elapsed;
}

double q21(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev,
           int* d_date, int* d_year, int* p_cat, int* p_part, int* p_brand, int* s_reg, int* s_supp) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_p = alloc_ht(P_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { build_keys_vals_element(d_date[i], d_year[i], true, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(P_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, p_cat[i], NONE, EQ, 1); if (sf) build_keys_vals_element(p_part[i], p_brand[i], sf, P_LEN, ht_p, P_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_reg[i], NONE, EQ, 1); if (sf) build_keys_element(s_supp[i], sf, S_LEN, ht_s, S_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 1000, q);
    q.memset(res, 0, 7 * 1000 * 8).wait();
    PerfTimer t;
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
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_p, q); sycl::free(ht_s, q); sycl::free(res, q);
    return elapsed;
}

double q22(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev,
           int* d_date, int* d_year, int* p_brand, int* p_part, int* s_reg, int* s_supp) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_p = alloc_ht(P_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { build_keys_vals_element(d_date[i], d_year[i], true, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(P_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, p_brand[i], NONE, GE, 260); selection_element(sf, p_brand[i], AND, LE, 267); if (sf) build_keys_vals_element(p_part[i], p_brand[i], sf, P_LEN, ht_p, P_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_reg[i], NONE, EQ, 2); if (sf) build_keys_element(s_supp[i], sf, S_LEN, ht_s, S_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 1000, q);
    q.memset(res, 0, 7 * 1000 * 8).wait();
    PerfTimer t;
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
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_p, q); sycl::free(ht_s, q); sycl::free(res, q);
    return elapsed;
}

double q23(sycl::queue& q, int* lo_date, int* lo_part, int* lo_supp, int* lo_rev,
           int* d_date, int* d_year, int* p_brand, int* p_part, int* s_reg, int* s_supp) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_p = alloc_ht(P_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { build_keys_vals_element(d_date[i], d_year[i], true, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(P_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, p_brand[i], NONE, EQ, 260); if (sf) build_keys_vals_element(p_part[i], p_brand[i], sf, P_LEN, ht_p, P_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_reg[i], NONE, EQ, 3); if (sf) build_keys_element(s_supp[i], sf, S_LEN, ht_s, S_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 1000, q);
    q.memset(res, 0, 7 * 1000 * 8).wait();
    PerfTimer t;
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
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_p, q); sycl::free(ht_s, q); sycl::free(res, q);
    return elapsed;
}

double q31(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev,
           int* d_date, int* d_year, int* c_nat, int* c_cust, int* c_reg, int* s_nat, int* s_supp, int* s_reg) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_c = alloc_ht(C_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, d_year[i], NONE, GE, 1992); selection_element(sf, d_year[i], AND, LE, 1997); if (sf) build_keys_vals_element(d_date[i], d_year[i], sf, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(C_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, c_reg[i], NONE, EQ, 2); if (sf) build_keys_vals_element(c_cust[i], c_nat[i], sf, C_LEN, ht_c, C_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_reg[i], NONE, EQ, 2); if (sf) build_keys_vals_element(s_supp[i], s_nat[i], sf, S_LEN, ht_s, S_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 25 * 25, q);
    q.memset(res, 0, 7 * 25 * 25 * 8).wait();
    PerfTimer t;
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int snat, cnat, year; bool sf = true;
        probe_keys_vals_element(lo_supp[idx], snat, sf, S_LEN, ht_s, S_LEN, 0);
        probe_keys_vals_element(lo_cust[idx], cnat, sf, C_LEN, ht_c, C_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        if (sf) {
            int group = (snat * 25 * 7 + cnat * 7 + (year - 1992)) % (7 * 25 * 25);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)lo_rev[idx]);
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_c, q); sycl::free(ht_s, q); sycl::free(res, q);
    return elapsed;
}

double q32(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev,
           int* d_date, int* d_year, int* c_nat, int* c_cust, int* c_nation_val, int* s_nat, int* s_supp, int* s_nation_val) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_c = alloc_ht(C_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, d_year[i], NONE, GE, 1992); selection_element(sf, d_year[i], AND, LE, 1997); if (sf) build_keys_vals_element(d_date[i], d_year[i], sf, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(C_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, c_nat[i], NONE, EQ, 1); if (sf) build_keys_vals_element(c_cust[i], c_nation_val[i], sf, C_LEN, ht_c, C_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_nat[i], NONE, EQ, 1); if (sf) build_keys_vals_element(s_supp[i], s_nation_val[i], sf, S_LEN, ht_s, S_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 250 * 250, q);
    q.memset(res, 0, 7 * 250 * 250 * 8).wait();
    PerfTimer t;
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int scit, ccit, year; bool sf = true;
        probe_keys_vals_element(lo_supp[idx], scit, sf, S_LEN, ht_s, S_LEN, 0);
        probe_keys_vals_element(lo_cust[idx], ccit, sf, C_LEN, ht_c, C_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        if (sf) {
            int group = (scit * 250 * 7 + ccit * 7 + (year - 1992)) % (7 * 250 * 250);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)lo_rev[idx]);
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_c, q); sycl::free(ht_s, q); sycl::free(res, q);
    return elapsed;
}

double q33(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev,
           int* d_date, int* d_year, int* c_nat, int* c_cust, int* c_city_val, int* s_nat, int* s_supp, int* s_city_val) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_c = alloc_ht(C_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, d_year[i], NONE, GE, 1992); selection_element(sf, d_year[i], AND, LE, 1997); if (sf) build_keys_vals_element(d_date[i], d_year[i], sf, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(C_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, c_nat[i], NONE, EQ, 1); selection_element(sf, c_nat[i], OR, EQ, 2); if (sf) build_keys_vals_element(c_cust[i], c_city_val[i], sf, C_LEN, ht_c, C_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_nat[i], NONE, EQ, 1); selection_element(sf, s_nat[i], OR, EQ, 2); if (sf) build_keys_vals_element(s_supp[i], s_city_val[i], sf, S_LEN, ht_s, S_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 250 * 250, q);
    q.memset(res, 0, 7 * 250 * 250 * 8).wait();
    PerfTimer t;
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int scit, ccit, year; bool sf = true;
        probe_keys_vals_element(lo_supp[idx], scit, sf, S_LEN, ht_s, S_LEN, 0);
        probe_keys_vals_element(lo_cust[idx], ccit, sf, C_LEN, ht_c, C_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        if (sf) {
            int group = (scit * 250 * 7 + ccit * 7 + (year - 1992)) % (7 * 250 * 250);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)lo_rev[idx]);
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_c, q); sycl::free(ht_s, q); sycl::free(res, q);
    return elapsed;
}

double q34(sycl::queue& q, int* lo_date, int* lo_cust, int* lo_supp, int* lo_rev,
           int* d_date, int* d_year, int* c_city, int* c_cust, int* s_city, int* s_supp) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_c = alloc_ht(C_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, d_year[i], NONE, EQ, 1995); selection_element(sf, d_year[i], OR, EQ, 1996); if (sf) build_keys_vals_element(d_date[i], d_year[i], sf, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(C_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, c_city[i], NONE, EQ, 1); selection_element(sf, c_city[i], OR, EQ, 2); if (sf) build_keys_vals_element(c_cust[i], c_city[i], sf, C_LEN, ht_c, C_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_city[i], NONE, EQ, 1); selection_element(sf, s_city[i], OR, EQ, 2); if (sf) build_keys_vals_element(s_supp[i], s_city[i], sf, S_LEN, ht_s, S_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 250 * 250, q);
    q.memset(res, 0, 7 * 250 * 250 * 8).wait();
    PerfTimer t;
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int scit, ccit, year; bool sf = true;
        probe_keys_vals_element(lo_supp[idx], scit, sf, S_LEN, ht_s, S_LEN, 0);
        probe_keys_vals_element(lo_cust[idx], ccit, sf, C_LEN, ht_c, C_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        if (sf) {
            int group = (scit * 250 * 7 + ccit * 7 + (year - 1992)) % (7 * 250 * 250);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)lo_rev[idx]);
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_c, q); sycl::free(ht_s, q); sycl::free(res, q);
    return elapsed;
}

double q41(sycl::queue& q, int* lo_date, int* lo_part, int* lo_cust, int* lo_supp, int* lo_rev, int* lo_scost,
           int* d_date, int* d_year, int* c_reg, int* c_cust, int* c_nat, int* s_reg, int* s_supp, int* p_mfgr, int* p_part) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_c = alloc_ht(C_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    int* ht_p = alloc_ht(P_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { build_keys_vals_element(d_date[i], d_year[i], true, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(C_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, c_reg[i], NONE, EQ, 1); if (sf) build_keys_vals_element(c_cust[i], c_nat[i], sf, C_LEN, ht_c, C_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_reg[i], NONE, EQ, 1); if (sf) build_keys_element(s_supp[i], sf, S_LEN, ht_s, S_LEN, 0); });
    q.parallel_for(P_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, p_mfgr[i], NONE, EQ, 1); selection_element(sf, p_mfgr[i], OR, EQ, 2); if (sf) build_keys_element(p_part[i], sf, P_LEN, ht_p, P_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 25, q);
    q.memset(res, 0, 7 * 25 * 8).wait();
    PerfTimer t;
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int cnat, year; bool sf = true;
        probe_keys_element(lo_part[idx], sf, P_LEN, ht_p, P_LEN, 0);
        probe_keys_vals_element(lo_cust[idx], cnat, sf, C_LEN, ht_c, C_LEN, 0);
        probe_keys_element(lo_supp[idx], sf, S_LEN, ht_s, S_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        if (sf) {
            int group = (cnat * 7 + (year - 1992)) % (7 * 25);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)(lo_rev[idx] - lo_scost[idx]));
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_c, q); sycl::free(ht_s, q); sycl::free(ht_p, q); sycl::free(res, q);
    return elapsed;
}

double q42(sycl::queue& q, int* lo_date, int* lo_part, int* lo_cust, int* lo_supp, int* lo_rev, int* lo_scost,
           int* d_date, int* d_year, int* c_reg, int* c_cust, int* s_reg, int* s_supp, int* s_nat, int* p_mfgr, int* p_part, int* p_cat) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_c = alloc_ht(C_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    int* ht_p = alloc_ht(P_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, d_year[i], NONE, EQ, 1997); selection_element(sf, d_year[i], OR, EQ, 1998); if (sf) build_keys_vals_element(d_date[i], d_year[i], sf, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(C_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, c_reg[i], NONE, EQ, 1); if (sf) build_keys_element(c_cust[i], sf, C_LEN, ht_c, C_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_reg[i], NONE, EQ, 1); if (sf) build_keys_vals_element(s_supp[i], s_nat[i], sf, S_LEN, ht_s, S_LEN, 0); });
    q.parallel_for(P_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, p_mfgr[i], NONE, EQ, 1); selection_element(sf, p_mfgr[i], OR, EQ, 2); if (sf) build_keys_vals_element(p_part[i], p_cat[i], sf, P_LEN, ht_p, P_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 25 * 25, q);
    q.memset(res, 0, 7 * 25 * 25 * 8).wait();
    PerfTimer t;
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int snat, cat, year; bool sf = true;
        probe_keys_element(lo_cust[idx], sf, C_LEN, ht_c, C_LEN, 0);
        probe_keys_vals_element(lo_supp[idx], snat, sf, S_LEN, ht_s, S_LEN, 0);
        probe_keys_vals_element(lo_part[idx], cat, sf, P_LEN, ht_p, P_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        if (sf) {
            int group = ((year - 1992) * 25 * 25 + snat * 25 + cat) % (7 * 25 * 25);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)(lo_rev[idx] - lo_scost[idx]));
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_c, q); sycl::free(ht_s, q); sycl::free(ht_p, q); sycl::free(res, q);
    return elapsed;
}

double q43(sycl::queue& q, int* lo_date, int* lo_part, int* lo_cust, int* lo_supp, int* lo_rev, int* lo_scost,
           int* d_date, int* d_year, int* c_reg, int* c_cust, int* s_nat, int* s_supp, int* s_city, int* p_cat, int* p_part, int* p_brand) {
    int* ht_d = alloc_ht(19981230-19920101+1, q);
    int* ht_c = alloc_ht(C_LEN, q);
    int* ht_s = alloc_ht(S_LEN, q);
    int* ht_p = alloc_ht(P_LEN, q);
    q.parallel_for(D_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, d_year[i], NONE, EQ, 1997); selection_element(sf, d_year[i], OR, EQ, 1998); if (sf) build_keys_vals_element(d_date[i], d_year[i], sf, D_LEN, ht_d, 19981230-19920101+1, 19920101); });
    q.parallel_for(C_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, c_reg[i], NONE, EQ, 1); if (sf) build_keys_element(c_cust[i], sf, C_LEN, ht_c, C_LEN, 0); });
    q.parallel_for(S_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, s_nat[i], NONE, EQ, 24); if (sf) build_keys_vals_element(s_supp[i], s_city[i], sf, S_LEN, ht_s, S_LEN, 0); });
    q.parallel_for(P_LEN, [=](sycl::id<1> i) { bool sf = false; selection_element(sf, p_cat[i], NONE, EQ, 3); if (sf) build_keys_vals_element(p_part[i], p_brand[i], sf, P_LEN, ht_p, P_LEN, 0); });
    unsigned long long* res = sycl::malloc_device<unsigned long long>(7 * 250 * 1000, q);
    q.memset(res, 0, 7 * 250 * 1000 * 8).wait();
    PerfTimer t;
    q.parallel_for(LO_LEN, [=](sycl::id<1> idx) {
        int scit, brand, year; bool sf = true;
        probe_keys_element(lo_cust[idx], sf, C_LEN, ht_c, C_LEN, 0);
        probe_keys_vals_element(lo_supp[idx], scit, sf, S_LEN, ht_s, S_LEN, 0);
        probe_keys_vals_element(lo_part[idx], brand, sf, P_LEN, ht_p, P_LEN, 0);
        probe_keys_vals_element(lo_date[idx], year, sf, D_LEN, ht_d, 19981230-19920101+1, 19920101);
        if (sf) {
            int group = ((year - 1992) * 250 * 1000 + scit * 1000 + brand) % (7 * 250 * 1000);
            auto sum_obj = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device>(res[group]);
            sum_obj.fetch_add((unsigned long long)(lo_rev[idx] - lo_scost[idx]));
        }
    }).wait();
    double elapsed = t.ms();
    sycl::free(ht_d, q); sycl::free(ht_c, q); sycl::free(ht_s, q); sycl::free(ht_p, q); sycl::free(res, q);
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
    cout << "Query,Repetition,KernelTime_ms" << endl;

    // Load columns
    int* h_lo_date = loadColumn<int>("lo_orderdate", LO_LEN, q);
    int* h_lo_disc = loadColumn<int>("lo_discount", LO_LEN, q);
    int* h_lo_quant = loadColumn<int>("lo_quantity", LO_LEN, q);
    int* h_lo_price = loadColumn<int>("lo_extendedprice", LO_LEN, q);
    int* h_lo_rev = loadColumn<int>("lo_revenue", LO_LEN, q);
    int* h_lo_scost = loadColumn<int>("lo_supplycost", LO_LEN, q);
    int* h_lo_part = loadColumn<int>("lo_partkey", LO_LEN, q);
    int* h_lo_supp = loadColumn<int>("lo_suppkey", LO_LEN, q);
    int* h_lo_cust = loadColumn<int>("lo_custkey", LO_LEN, q);
    int* h_d_date = loadColumn<int>("d_datekey", D_LEN, q);
    int* h_d_year = loadColumn<int>("d_year", D_LEN, q);
    int* h_p_part = loadColumn<int>("p_partkey", P_LEN, q);
    int* h_p_mfgr = loadColumn<int>("p_mfgr", P_LEN, q);
    int* h_p_cat = loadColumn<int>("p_category", P_LEN, q);
    int* h_p_brand = loadColumn<int>("p_brand1", P_LEN, q);
    int* h_s_supp = loadColumn<int>("s_suppkey", S_LEN, q);
    int* h_s_reg = loadColumn<int>("s_region", S_LEN, q);
    int* h_s_nat = loadColumn<int>("s_nation", S_LEN, q);
    int* h_s_city = loadColumn<int>("s_city", S_LEN, q);
    int* h_c_cust = loadColumn<int>("c_custkey", C_LEN, q);
    int* h_c_reg = loadColumn<int>("c_region", C_LEN, q);
    int* h_c_nat = loadColumn<int>("c_nation", C_LEN, q);
    int* h_c_city = loadColumn<int>("c_city", C_LEN, q);

    if (!h_lo_date || !h_lo_disc || !h_lo_quant || !h_lo_price || !h_lo_rev || !h_lo_scost || !h_lo_part || !h_lo_supp || !h_lo_cust ||
        !h_d_date || !h_d_year || !h_p_part || !h_p_mfgr || !h_p_cat || !h_p_brand || !h_s_supp || !h_s_reg || !h_s_nat || !h_s_city ||
        !h_c_cust || !h_c_reg || !h_c_nat || !h_c_city) {
        cerr << "Error: Failed to load one or more columns from " << DATA_DIR << endl;
        return 1;
    }

    // Device memory
    int* lo_date = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_disc = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_quant = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_price = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_rev = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_scost = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_part = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_supp = sycl::malloc_device<int>(LO_LEN, q);
    int* lo_cust = sycl::malloc_device<int>(LO_LEN, q);
    int* d_date = sycl::malloc_device<int>(D_LEN, q);
    int* d_year = sycl::malloc_device<int>(D_LEN, q);
    int* p_part = sycl::malloc_device<int>(P_LEN, q);
    int* p_mfgr = sycl::malloc_device<int>(P_LEN, q);
    int* p_cat = sycl::malloc_device<int>(P_LEN, q);
    int* p_brand = sycl::malloc_device<int>(P_LEN, q);
    int* s_supp = sycl::malloc_device<int>(S_LEN, q);
    int* s_reg = sycl::malloc_device<int>(S_LEN, q);
    int* s_nat = sycl::malloc_device<int>(S_LEN, q);
    int* s_city = sycl::malloc_device<int>(S_LEN, q);
    int* c_cust = sycl::malloc_device<int>(C_LEN, q);
    int* c_reg = sycl::malloc_device<int>(C_LEN, q);
    int* c_nat = sycl::malloc_device<int>(C_LEN, q);
    int* c_city = sycl::malloc_device<int>(C_LEN, q);

    q.memcpy(lo_date, h_lo_date, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_disc, h_lo_disc, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_quant, h_lo_quant, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_price, h_lo_price, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_rev, h_lo_rev, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_scost, h_lo_scost, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_part, h_lo_part, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_supp, h_lo_supp, (size_t)LO_LEN * sizeof(int));
    q.memcpy(lo_cust, h_lo_cust, (size_t)LO_LEN * sizeof(int)).wait();
    q.memcpy(d_date, h_d_date, (size_t)D_LEN * sizeof(int));
    q.memcpy(d_year, h_d_year, (size_t)D_LEN * sizeof(int));
    q.memcpy(p_part, h_p_part, (size_t)P_LEN * sizeof(int));
    q.memcpy(p_mfgr, h_p_mfgr, (size_t)P_LEN * sizeof(int));
    q.memcpy(p_cat, h_p_cat, (size_t)P_LEN * sizeof(int));
    q.memcpy(p_brand, h_p_brand, (size_t)P_LEN * sizeof(int));
    q.memcpy(s_supp, h_s_supp, (size_t)S_LEN * sizeof(int));
    q.memcpy(s_reg, h_s_reg, (size_t)S_LEN * sizeof(int));
    q.memcpy(s_nat, h_s_nat, (size_t)S_LEN * sizeof(int));
    q.memcpy(s_city, h_s_city, (size_t)S_LEN * sizeof(int));
    q.memcpy(c_cust, h_c_cust, (size_t)C_LEN * sizeof(int));
    q.memcpy(c_reg, h_c_reg, (size_t)C_LEN * sizeof(int));
    q.memcpy(c_nat, h_c_nat, (size_t)C_LEN * sizeof(int));
    q.memcpy(c_city, h_c_city, (size_t)C_LEN * sizeof(int)).wait();

    for (int r = 0; r < REPETITIONS; r++) {
        cout << "q11," << r + 1 << "," << q11(q, lo_date, lo_disc, lo_quant, lo_price) << endl;
        cout << "q12," << r + 1 << "," << q12(q, lo_date, lo_disc, lo_quant, lo_price) << endl;
        cout << "q13," << r + 1 << "," << q13(q, lo_date, lo_disc, lo_quant, lo_price) << endl;
        cout << "q21," << r + 1 << "," << q21(q, lo_date, lo_part, lo_supp, lo_rev, d_date, d_year, p_cat, p_part, p_brand, s_reg, s_supp) << endl;
        cout << "q22," << r + 1 << "," << q22(q, lo_date, lo_part, lo_supp, lo_rev, d_date, d_year, p_brand, p_part, s_reg, s_supp) << endl;
        cout << "q23," << r + 1 << "," << q23(q, lo_date, lo_part, lo_supp, lo_rev, d_date, d_year, p_brand, p_part, s_reg, s_supp) << endl;
        cout << "q31," << r + 1 << "," << q31(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_nat, c_cust, c_reg, s_nat, s_supp, s_reg) << endl;
        cout << "q32," << r + 1 << "," << q32(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_nat, c_cust, c_city, s_nat, s_supp, c_city) << endl;
        cout << "q33," << r + 1 << "," << q33(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_nat, c_cust, c_city, s_nat, s_supp, c_city) << endl;
        cout << "q34," << r + 1 << "," << q34(q, lo_date, lo_cust, lo_supp, lo_rev, d_date, d_year, c_city, c_cust, s_city, s_supp) << endl;
        cout << "q41," << r + 1 << "," << q41(q, lo_date, lo_part, lo_cust, lo_supp, lo_rev, lo_scost, d_date, d_year, c_reg, c_cust, c_nat, s_reg, s_supp, p_mfgr, p_part) << endl;
        cout << "q42," << r + 1 << "," << q42(q, lo_date, lo_part, lo_cust, lo_supp, lo_rev, lo_scost, d_date, d_year, c_reg, c_cust, s_reg, s_supp, s_nat, p_mfgr, p_part, p_cat) << endl;
        cout << "q43," << r + 1 << "," << q43(q, lo_date, lo_part, lo_cust, lo_supp, lo_rev, lo_scost, d_date, d_year, c_reg, c_cust, s_nat, s_supp, s_city, p_cat, p_part, p_brand) << endl;
    }

    return 0;
}
