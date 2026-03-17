CXX := icpx
CXXOPT := -std=c++20 -O3 -Wall
CXXFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

# Default target
all: join join_tiling

# Pattern rule to compile a file
%: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXOPT) $< -o $@

clean:
	rm -f join join_tiling project

