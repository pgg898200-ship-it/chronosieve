CC ?= clang
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

BASE_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Werror -Iinclude -Icsrc
SIMD_CFLAGS :=
OPT_CFLAGS := -O3
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    OPT_CFLAGS += -mcpu=native
  endif
endif
ifeq ($(UNAME_M),x86_64)
  ifdef AVX2
    SIMD_CFLAGS += -mavx2 -mfma
  endif
endif

CORE_SRCS := csrc/edetector.c csrc/conformal.c csrc/fuser.c csrc/digest.c
LDFLAGS := -lm

.PHONY: all cli test-c test-parity py-install test lint bench clean

all: cli

cli: $(CORE_SRCS) csrc/cli_main.c include/chronosieve.h csrc/cs_simd.h csrc/cs_internal.h
	$(CC) $(BASE_CFLAGS) $(SIMD_CFLAGS) $(OPT_CFLAGS) $(CORE_SRCS) csrc/cli_main.c $(LDFLAGS) -o chronosieve
	strip chronosieve

test-c: $(CORE_SRCS) tests/c/test_c_core.c
	$(CC) $(BASE_CFLAGS) $(SIMD_CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
		$(CORE_SRCS) tests/c/test_c_core.c $(LDFLAGS) -o build_test_c_asan
	./build_test_c_asan
	$(CC) $(BASE_CFLAGS) -O1 -g -DCS_FORCE_SCALAR=1 -fsanitize=address,undefined -fno-omit-frame-pointer \
		$(CORE_SRCS) tests/c/test_c_core.c $(LDFLAGS) -o build_test_c_scalar_asan
	./build_test_c_scalar_asan
	rm -rf build_test_c_asan build_test_c_asan.dSYM build_test_c_scalar_asan build_test_c_scalar_asan.dSYM

test-parity: $(CORE_SRCS) tests/c/test_parity.c
	$(CC) $(BASE_CFLAGS) $(SIMD_CFLAGS) $(OPT_CFLAGS) $(CORE_SRCS) tests/c/test_parity.c $(LDFLAGS) -o build_parity_simd
	$(CC) $(BASE_CFLAGS) -O2 -DCS_FORCE_SCALAR=1 $(CORE_SRCS) tests/c/test_parity.c $(LDFLAGS) -o build_parity_scalar
	./build_parity_simd > build_parity_simd.txt
	./build_parity_scalar > build_parity_scalar.txt
	python3 tests/c/check_parity.py build_parity_simd.txt build_parity_scalar.txt
	rm -rf build_parity_simd build_parity_scalar build_parity_simd.txt build_parity_scalar.txt

py-install:
	python3 -m pip install --no-build-isolation --no-deps -e .

test: cli test-c test-parity
	pytest -v tests/

lint:
	$(CC) $(BASE_CFLAGS) -fsyntax-only $(CORE_SRCS) csrc/cli_main.c
	ruff check python/ tests/ benchmarks/ setup.py
	mypy python/chronosieve

bench: cli
	./chronosieve bench --n 2000000
	python3 benchmarks/run_all.py

clean:
	rm -rf chronosieve build_test_c_asan build_test_c_scalar_asan build_test_*.dSYM *.dSYM \
		build_parity_simd build_parity_scalar build_parity_simd.txt build_parity_scalar.txt \
		build/ *.egg-info python/*.egg-info python/chronosieve/_core*.so
