ICPX_BIN := /opt/intel/oneapi/compiler/2025.1/bin/icpx
ACPP_BIN := /media/ACPP/AdaptiveCpp-25.10.0/install/bin/acpp
ACPP_LIB := /media/ACPP/AdaptiveCpp-25.10.0/install/lib

CXXOPT := -std=c++20 -O3 -Wall

ICPX_FLAGS := -fsycl -fsycl-targets=spir64,nvptx64-nvidia-cuda,amdgcn-amd-amdhsa \
               -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_89 \
               -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a
ACPP_FLAGS := --acpp-targets=generic

COMPILER ?= icpx

ifeq ($(COMPILER), icpx)
  CXX := $(ICPX_BIN)
  CXXFLAGS := $(ICPX_FLAGS)
else ifeq ($(COMPILER), acpp)
  CXX := $(ACPP_BIN)
  CXXFLAGS := $(ACPP_FLAGS)
  export LD_LIBRARY_PATH := $(ACPP_LIB):$(LD_LIBRARY_PATH)
endif

TARGET ?= 1

# Default target
all:

# Pattern rule to compile a file
%: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXOPT) $< -o $@

# Target to compile BOTH versions for a file (e.g. make join_both)
%_both: %.cpp
	$(ICPX_BIN) $(ICPX_FLAGS) $(CXXOPT) $< -o $*_icpx
	$(ACPP_BIN) $(ACPP_FLAGS) $(CXXOPT) $< -o $*_acpp
