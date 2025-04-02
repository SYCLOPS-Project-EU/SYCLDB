#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <sycl/sycl.hpp>

using namespace std;

#define SF 20

#define BASE_PATH "test/ssb/data/"

#if SF == 1
#define DATA_DIR BASE_PATH "s1_columnar/"
#define LO_LEN 6001171
#define P_LEN 200000
#define S_LEN 2000
#define C_LEN 30000
#define D_LEN 2556
#elif SF == 10
#define DATA_DIR BASE_PATH "s10_columnar/"
#define LO_LEN 59986214
#define P_LEN 800000
#define S_LEN 20000
#define C_LEN 300000
#define D_LEN 2556
#elif SF == 20
#define DATA_DIR BASE_PATH "s20_columnar/"
#define LO_LEN 119994746
#define P_LEN 1000000
#define S_LEN 40000
#define C_LEN 600000
#define D_LEN 2556
#else // SF == 100
#define DATA_DIR BASE_PATH "s100_columnar/"
#define LO_LEN 600043265
#define P_LEN 1400000
#define S_LEN 200000
#define C_LEN 3000000
#define D_LEN 2556

#endif

void wait_and_add_time(sycl::event e, float &total_time) {
  e.wait();
  const auto start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
  float time = (end - start) / 1e6;
  total_time += time;
  std::cout << "so far took: " << time << std::endl;
}

int index_of(string *arr, int len, string val) {
  for (int i = 0; i < len; i++)
    if (arr[i] == val)
      return i;

  return -1;
}

string lookup(string col_name) {
  string lineorder[] = {"lo_orderkey",      "lo_linenumber",    "lo_custkey",
                        "lo_partkey",       "lo_suppkey",       "lo_orderdate",
                        "lo_orderpriority", "lo_shippriority",  "lo_quantity",
                        "lo_extendedprice", "lo_ordtotalprice", "lo_discount",
                        "lo_revenue",       "lo_supplycost",    "lo_tax",
                        "lo_commitdate",    "lo_shipmode"};
  string part[] = {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                   "p_color",   "p_type", "p_size", "p_container"};
  string supplier[] = {"s_suppkey", "s_name",   "s_address", "s_city",
                       "s_nation",  "s_region", "s_phone"};
  string customer[] = {"c_custkey", "c_name",   "c_address", "c_city",
                       "c_nation",  "c_region", "c_phone",   "c_mktsegment"};
  string date[] = {"d_datekey",
                   "d_date",
                   "d_dayofweek",
                   "d_month",
                   "d_year",
                   "d_yearmonthnum",
                   "d_yearmonth",
                   "d_daynuminweek",
                   "d_daynuminmonth",
                   "d_daynuminyear",
                   "d_sellingseason",
                   "d_lastdayinweekfl",
                   "d_lastdayinmonthfl",
                   "d_holidayfl",
                   "d_weekdayfl"};

  if (col_name[0] == 'l') {
    int index = index_of(lineorder, 17, col_name);
    return "LINEORDER" + to_string(index);
  } else if (col_name[0] == 's') {
    int index = index_of(supplier, 7, col_name);
    return "SUPPLIER" + to_string(index);
  } else if (col_name[0] == 'c') {
    int index = index_of(customer, 8, col_name);
    return "CUSTOMER" + to_string(index);
  } else if (col_name[0] == 'p') {
    int index = index_of(part, 9, col_name);
    return "PART" + to_string(index);
  } else if (col_name[0] == 'd') {
    int index = index_of(date, 15, col_name);
    return "DDATE" + to_string(index);
  }

  return "";
}

template <typename T>
T *loadColumn(string col_name, int num_entries, sycl::queue queue,
              bool shared_input = false) {
  // std::chrono::time_point<std::chrono::system_clock> start, end;
  // start = std::chrono::system_clock::now();
  T *h_col = sycl::malloc_host<T>(num_entries, queue);
  // end = std::chrono::system_clock::now();
  // if (col_name[0] == 'l')
  //   cout << "Allocated " << num_entries * sizeof(T) << " bytes in " <<
  //   std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
  //   << " ms" << endl;
  // T* h_col = new T[num_entries];
  string filename = DATA_DIR + lookup(col_name);
  ifstream colData(filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    return NULL;
  }

  // obtain the n of entries for arbitrary scale factor
  /*
  colData.seekp(0, std::ios::end);
  std::streampos fileSize = colData.tellp();
  int num_entries2 = static_cast<int>(fileSize / sizeof(T));
  colData.seekp(0, std::ios::beg);
  std::cout << num_entries << " " << num_entries2 << std::endl;
  */
  // start = std::chrono::system_clock::now();
  colData.read((char *)h_col, num_entries * sizeof(T));
  // end = std::chrono::system_clock::now();
  //  only print for probe columns
  // if (col_name[0] == 'l')
  //   cout << "Read " << num_entries * sizeof(T) << " bytes in " <<
  //   std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()
  //   << " ms" << endl;
  return h_col;
}

template <typename T>
int storeColumn(string col_name, int num_entries, int *h_col) {
  string filename = DATA_DIR + lookup(col_name);
  ofstream colData(filename.c_str(), ios::out | ios::binary);
  if (!colData) {
    return -1;
  }

  colData.write((char *)h_col, num_entries * sizeof(T));
  return 0;
}

/*int main() {*/
// int *h_col = new int[10];
// for (int i=0; i<10; i++) h_col[i] = i;
// storeColumn<int>("test", 10, h_col);
// int *l_col = loadColumn<int>("test", 10);
// for (int i=0; i<10; i++) cout << l_col[i] << " ";
// cout << endl;
// return 0;
/*}*/
