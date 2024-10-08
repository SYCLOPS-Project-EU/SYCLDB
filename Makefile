# SYCL: https://github.com/intel/llvm/issues/6636
# https://support.codeplay.com/t/cmake-integration-for-oneapi-for-nvidia-gpus/542/7
CXXOPT := -std=c++20 -O3 -Wall
SYCLONDEVICE := -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64
# note: work group actions are not supported by acpp, so source code is modified
ACPP_FLAGS := --acpp-targets="generic" -DCOMPILER_IS_ACPP=1 -dc

# user can specify custom clang++ path
# then call for instance:
# make PATH_TO_LLVM=/home/ivan/llvm/build/install COMPILER=clang++ join
PATH_TO_LLVM := /usr

CXX := clang++ # default compiler
CXXFLAGS := $(SYCLONDEVICE)

ifeq ($(COMPILER),clang++)
	CXX := $(PATH_TO_LLVM)/bin/clang++
	LD_LIBRARY_PATH := $(PATH_TO_LLVM)/lib
	export LD_LIBRARY_PATH
endif

ifeq ($(COMPILER),acpp)
	CXX := acpp
	CXXFLAGS := $(ACPP_FLAGS)
endif

# Default target
all:

# Pattern rule to compile a file
%: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXOPT) $< -o $@

#./$@
#rm -f $@

