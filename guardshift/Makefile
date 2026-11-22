CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
CPPFLAGS += -Iinclude
BUILD := build/native
TESTS := $(patsubst tests/%.cpp,$(BUILD)/%,$(wildcard tests/test*.cpp))

.PHONY: test quick bench
test: $(TESTS)
	@set -e; for test in $(TESTS); do "$$test"; done

$(TESTS): $(BUILD)/%: tests/%.cpp include/guardshift.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BUILD)/bench: bench/bench.cpp include/guardshift.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DNDEBUG $< -o $@

quick: $(BUILD)/bench
	$(BUILD)/bench --quick --trace replay/azure-code.csv

bench: $(BUILD)/bench
	$(BUILD)/bench --full --trace replay/azure-code.csv
