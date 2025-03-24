CXXOPT := -std=c++20 -O3 -Wall
CXXFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64
CXX := clang++

# Default target
all:

# Pattern rule to compile a file
%: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXOPT) $< -o $@

#./$@
#rm -f $@

