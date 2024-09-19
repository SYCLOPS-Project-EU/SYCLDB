#pragma once

#include <cmath>
#include <cstdio>  /* perror */
#include <cstdlib> /* posix_memalign */
#include <immintrin.h>
#include <iostream>
#include <random>
#include <thread>

#include <sycl/sycl.hpp>

using namespace std;

#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))
#define RANDR_RANGE(N) ((double)rand_r(&seed) / ((double)RAND_MAX + 1) * (N))
static int seeded = 0;

/** Check wheter seeded, if not seed the generator with current time */
static void check_seed() {
  if (!seeded) {
    srand(0);
    seeded = 1;
  }
}

/**
 * Shuffle tuples of the relation using Knuth shuffle.
 *
 * @param relation
 */
void knuth_shuffle(int *arr, int num_tuples) {
  int i;
  for (i = num_tuples - 1; i > 0; i--) {
    int j = RAND_RANGE(i);
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/**
 * Generate unique tuple IDs with Knuth shuffling
 * relation must have been allocated
 */
void random_unique_gen(int *&arr, int num_tuples) {
  int i;

  for (i = 0; i < num_tuples; i++) {
    arr[i] = (i + 1);
  }

  /* randomly shuffle elements */
  knuth_shuffle(arr, num_tuples);
}

void dummy_initialize(int *&arr, int num_tuples) {
  for (int i = 0; i < num_tuples; i++) {
    arr[i] = i;
  }
}

int create_relation_pk(int *&keys, int *&vals, int num_tuples,
                       sycl::queue queue) {
  check_seed();

  keys = sycl::malloc_host<int>(num_tuples, queue);
  vals = sycl::malloc_host<int>(num_tuples, queue);

  if (!keys || !vals) {
    perror("out of memory");
    return -1;
  }

  random_unique_gen(keys, num_tuples);
  dummy_initialize(vals, num_tuples);

  return 0;
}

int create_relation_fk(int *&keys, int *&vals, int num_tuples, const int maxid,
                       sycl::queue queue) {
  int i, iters, remainder;

  check_seed();

  keys = sycl::malloc_host<int>(num_tuples, queue);
  vals = sycl::malloc_host<int>(num_tuples, queue);

  if (!keys || !vals) {
    perror("out of memory");
    return -1;
  }

  // alternative generation method
  iters = num_tuples / maxid;
  for (i = 0; i < iters; i++) {
    int *tuples = keys + maxid * i;
    random_unique_gen(tuples, maxid);
  }

  // if num_tuples is not an exact multiple of maxid
  remainder = num_tuples % maxid;
  if (remainder > 0) {
    int *tuples = keys + maxid * iters;
    random_unique_gen(tuples, remainder);
  }

  dummy_initialize(vals, num_tuples);

  return 0;
}

void generateUniformCPU(float *&data1, float *&data2,
                        unsigned long long num_items, sycl::queue queue,
                        bool shared_input = false) {

  data1 = (shared_input) ? sycl::malloc_shared<float>(num_items, queue)
                         : sycl::malloc_host<float>(num_items, queue);
  data2 = (shared_input) ? sycl::malloc_shared<float>(num_items, queue)
                         : sycl::malloc_host<float>(num_items, queue);

  static std::mt19937 generator(std::random_device{}());
  static std::uniform_real_distribution<float> distribution(0.0, 1.0);
  // srand(1234ULL);
  for (unsigned long long i = 0; i < num_items; i++) {
    data1[i] = distribution(generator);
    data2[i] = distribution(generator);
    // data1[i] = (float)rand() / (float)RAND_MAX;
    // data2[i] = (float)rand() / (float)RAND_MAX;
  }
}
