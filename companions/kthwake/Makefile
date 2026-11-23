CXX ?= c++
CPPFLAGS += -Iinclude
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
.PHONY: all test example clean
all: build/kthwake
build:
	mkdir -p build
build/kthwake: src/main.cpp include/kthwake.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@
build/test: tests/test.cpp include/kthwake.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@
build/timers: timerbench/timers.cpp include/kthwake.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@
test: build/test build/kthwake build/timers
	./build/test
	python3 tests/cli.py ./build/kthwake
	python3 tests/timerbench.py ./build/timers
example: build/kthwake
	./build/kthwake --idle examples/guards.csv examples/ticks.csv
clean:
	$(RM) -r build
