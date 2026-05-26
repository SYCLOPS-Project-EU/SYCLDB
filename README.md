# SYCLDB: High-Performance GPU/XPU Query Engine with Calcite

SYCLDB is an experimental high-performance SQL query engine built on top of SYCL (using AdaptiveCpp/oneAPI) to execute relational queries across heterogeneous hardware (CPUs, NVIDIA GPUs, AMD GPUs). It connects to an Apache Calcite-based SQL query planner to parse, optimize, and generate plans which are executed natively on target devices.

---

## 1. Repository Layout (What is Where)

The repository is structured into two main parts: the native execution client (`client`) and the query planning server (`server`).

*   **`app/`**: Orchestrates the CLI, handles input SQL query files, connects to the Calcite server, and coordinates the execution loop.
*   **`runtime/`**: Responsible for device discovery (CPU/GPU queues), memory allocation strategies (device and host memory managers), and base-table loading.
*   **`executor/`**: The core execution driver (`ddor_executor.cpp`). It takes the execution plans from the Calcite planner, schedules physical operators, and handles final result writing.
*   **`kernels/`**: High-performance SYCL parallel algorithms implementing physical relational operators:
    *   `selection.hpp`: Filter evaluations.
    *   `projection.hpp`: Column mapping and arithmetic calculations.
    *   `join.hpp`: Hash table building and probing for parallel hash-joins.
    *   `aggregation.hpp`: Parallel reductions and group-by aggregations.
*   **`models/`**: Engine data structures including `Table` (representing base tables) and `TransientTable` (representing intermediate materialized results and managing pending kernel dependencies).
*   **`operations/`**: Data loading utilities (loading columnar binaries from disk) and pre-processing tasks.
*   **`gen-cpp/`**: Thrift C++ RPC client bindings used to communicate with the query planner.
*   **`server/`**: The Apache Calcite-based query planner (written in Java). It parses SQL, performs optimizations (e.g., project/filter pushdowns), and generates physical plans.
*   **`queries/`**: Contains the transformed SSB (Star Schema Benchmark) SQL queries.
*   **`standalone_executables/`**: Self-contained hardcoded and modular benchmark implementations for direct testing (e.g., `q11_hardcoded.cpp` and `q11_modular.cpp`).

---

## 2. Prerequisites

To build and run the engine, you need:
1.  **AdaptiveCpp / Intel oneAPI**: A compiler chain supporting C++20 and SYCL (e.g., `acpp`).
2.  **Apache Thrift**: Libraries and development headers for communication between client and planner.
3.  **SSB Columnar Data**: Star Schema Benchmark dataset pre-processed into binary columnar files.
4.  **Java & Gradle**: Required to run the Calcite planning server.

---

## 3. How to Run the Code

### Step A: Start the Calcite Planner Server
Before running the C++ client, start the query planner server in the background:
```bash
cd server
./gradlew run
```
The server will start listening for Thrift RPC requests on `localhost:5555`.

### Step B: Build the Engine Client
Compile the native C++ client using the provided `Makefile`:
```bash
make clean
make client
```
This builds the `./client` binary.

### Step C: Run a Query
You can run a single query by passing the SQL file path:
```bash
./client queries/transformed/q11.sql --data-dir /path/to/ssb_columnar/ --device-selector cuda
```

### Step D: Run the SSB Suite
To run the full Star Schema Benchmark suite and write results to a CSV file:
```bash
./client --benchmark-ssb --suite benchmark/ssb_queries.txt --results ssb_results.csv --data-dir /path/to/ssb_columnar/ --device-selector cuda
```

---

## 4. Key Runtime Configurations

*   **Data Path Selection (`--data-dir <path>`)**: Specifies the directory containing the binary columnar data files (e.g., `lo_orderdate`, `lo_discount`, etc.).
*   **Device Selection (`--device-selector <substring>`)**: Filters the execution devices. Pass `cpu`, `cuda`, or `opencl` (or any device name substring) to choose where to execute the workload.
*   **No-Fusion Mode**: The execution driver is optimized to run queries step-by-step without operator fusion to guarantee predictable, clean benchmarks.
*   **sycl::reduction**: Relational reductions (e.g., computing a single sum or count) natively leverage `sycl::reduction` to maximize memory bandwidth and performance on hardware-supported platforms.
