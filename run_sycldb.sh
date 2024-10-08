#!/bin/bash

# benchmarks are 11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43
benchmarks=(11 12 13 21 22 23 31 32 33 34 41 42 43)

# iterate over all benchmarks
for i in "${benchmarks[@]}"
do
    rm ssb/q$i
    make ssb/q$i
    ./ssb/q$i -t 1 -g 1 >> sycldb_results2.txt
done
for i in "${benchmarks[@]}"
do
    ./ssb/q$i -t 0 -g 1 >> sycldb_results2.txt
done
for i in "${benchmarks[@]}"
do
    ./ssb/q$i -t 1 -g 0 >> sycldb_results2.txt
done
for i in "${benchmarks[@]}"
do
    ./ssb/q$i -t 0 -g 0 >> sycldb_results2.txt
done