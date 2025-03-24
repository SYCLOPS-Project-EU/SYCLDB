# SYCLDB

SYCLDB is a templated SYCL device (and host) library implementing fundamental relational operators for executing analytical SQL SPJA queries. Currently, SYCLDB supports:

- **Projection**
- **Selection**
- **Hash Join**
- **Aggregation**

## Compilation & Running Sample Queries

Both the hash join and projection examples generate their own sample input. You can run them as follows:

```bash
make join
./join

make project
./project
```

## Running SSB Benchmarks

The Star Schema Benchmark (SSB) requires generating a test dataset before execution. To do this:

```bash
cd test/ssb/dbgen
make
cd ../loader
make
cd ../..
python3 util.py 20 # Scale factor 20 ~ 2GB
```

Then, return to the main directory and execute queries:

```bash
make ssb/qXX  # XX can be 11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43
./ssb/qXX
```

## Requirements

A DPCPP compiler (e.g., SYCL) is required. This also necessitates an OpenCL runtime, which is not included in the project.

To install OpenCL on an Intel CPU via oneAPI:

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
| tee /etc/apt/sources.list.d/oneAPI.list

apt update && apt install -y intel-oneapi-runtime-opencl intel-oneapi-compiler-dpcpp-cpp-runtime

source /opt/intel/oneapi/setvars.sh
```

For a DPCPP/SYCL nightly build:

```bash
wget https://github.com/intel/llvm/releases/download/nightly-2024-06-03/sycl_linux.tar.gz
tar -xzf sycl_linux.tar.gz

export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export PATH=$PWD/bin:$PATH
```

Verify correct installation with:

```bash
sycl-ls  # Lists available SYCL devices
```

If a GPU is present but not recognized, install [OneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/download/).

## Execution Flags

All benchmarks accept the following command-line arguments:

- `-p`  → Number of partitions to split the probe/projection table into.
- `-g`  → Number of GPUs/devices to use (`g=0`: CPU, `g=1`: single GPU, `g>1`: multiple devices compute independently).
- `-r`  → Number of repetitions (each repetition re-allocates memory, transfers, and computes the data).

Additional flags for **hash join**:
- `-t`  → Probe type (`0`: 1D non-tiled, `>=1`: tile-based probe).
- `-d`  → Size of the build table (tuples).
- `-f`  → Size of the probe table (tuples).

Additional flags for **projection**:
- `-t`  → Projection type (`0`: 1D non-tiled, `>=1`: tile-based projection).
- `-n`  → Table size (tuples).
- `-s`  → Computation type (`0`: dot-product, `>=1`: sigmoid).

The SSB dataset is expected in the directory generated in the first step. The `SF` macro in `ssb/ssb_utils.hpp` provides default scale factors (1, 10, 20, 100). We typically use `100` and adjust `LO_LEN` as needed.

## RISC-V Setup

There are two ways to run SYCLDB on RISC-V:

### 1. Running on a Remote RISC-V Accelerator

Establish a client-server connection using a [HAL server from the oneAPI Construction Kit](https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/examples/hal_cpu_remote_server/README.md#building-the-client). Modify `exchange.hpp` by replacing `is_gpu()` with `is_accelerator()`.

### 2. Running on a Local RISC-V CPU Backend

To run SYCLDB natively on a RISC-V CPU, a custom SYCL implementation is required. This involves setting up DPCPP, SPIR-V tools, LLVM, and OCK:

#### **DPCPP for RISC-V**

```bash
git clone https://github.com/PietroGhg/llvm -b pietro/cross_riscv
python llvm/buildbot/configure.py --host-target="RISCV" --native_cpu \
  --native_cpu_libclc_targets "riscv64-unknown-linux-gnu"
```

#### **SPIR-V Tools**

```bash
git clone https://github.com/KhronosGroup/SPIRV-Tools
cd SPIRV-Tools
cmake -DCMAKE_INSTALL_PREFIX=$PWD/build/install -G Ninja -S . -B build
ninja -C build install
```

#### **LLVM for RISC-V**

```bash
cd ~/DPCPP
git clone https://github.com/llvm/llvm-project -b release/18.x
cd llvm-project
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_ENABLE_ZLIB=Off -DLLVM_ENABLE_ZSTD=Off \
  -DCMAKE_INSTALL_PREFIX=./build/install
ninja -C build install
```

#### **Installing OCK**

```bash
export VULKAN_SDK=~/DPCPP/SPIRV-Tools/build
export LLVMInstall=~/DPCPP/llvm-project/build/install

cd ~/DPCPP
git clone https://github.com/codeplaysoftware/oneapi-construction-kit
cd oneapi-construction-kit
cmake -B build -S . -DCA_ENABLE_API=cl -GNinja \
  -DCA_CL_ENABLE_ICD_LOADER=ON -DOCL_EXTENSION_cl_khr_command_buffer=ON \
  -DOCL_EXTENSION_cl_khr_command_buffer_mutable_dispatch=ON \
  -DOCL_EXTENSION_cl_khr_extended_async_copies=ON \
  -DSpirvTools_spirv-as_EXECUTABLE=$VULKAN_SDK/tools/spirv-as \
  -DCMAKE_BUILD_TYPE=Release -DCA_ENABLE_DOCUMENTATION=Off \
  -DCMAKE_INSTALL_PREFIX=$PWD/build/install \
  -DCA_LLVM_INSTALL_DIR=$LLVMInstall
ninja -C build install
```

#### **Makefile Configuration for RISC-V**

Use the following flags to compile for RISC-V CPU:

```bash
-fsycl -fsycl-targets=spir64 --target=riscv64-redhat-linux \
  -fsycl-llc-options=-mattr=+m,+f,+a,+d
```

(Replace `-fsycl-targets=nvptx64-nvidia-cuda,spir64` if targeting NVIDIA/Intel.)

---

This improved README provides better structure, readability, and clear instructions for setting up SYCLDB on different platforms.

