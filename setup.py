from __future__ import annotations

import os
import platform
import sys

from setuptools import Extension, setup

arch = platform.machine().lower()
is_msvc = sys.platform == "win32" and "mingw" not in platform.python_compiler().lower()

compile_args: list[str] = []
define_macros: list[tuple[str, str | None]] = []
libraries: list[str] = []

if os.environ.get("CHRONOSIEVE_FORCE_SCALAR") == "1":
    define_macros.append(("CS_FORCE_SCALAR", "1"))

if is_msvc:
    compile_args = ["/std:c11", "/O2", "/W4"]
    if os.environ.get("CHRONOSIEVE_AVX2") == "1" and arch in ("amd64", "x86_64"):
        compile_args.append("/arch:AVX2")
else:
    compile_args = ["-std=c11", "-O3", "-Wall", "-Wextra"]
    libraries = ["m"]
    if arch in ("arm64", "aarch64"):
        if sys.platform == "darwin":
            compile_args.append("-mcpu=native")
    elif arch in ("x86_64", "amd64"):
        if os.environ.get("CHRONOSIEVE_AVX2") == "1":
            compile_args.extend(["-mavx2", "-mfma"])
        elif os.environ.get("CHRONOSIEVE_NATIVE") == "1":
            compile_args.append("-march=native")

ext = Extension(
    name="chronosieve._core",
    sources=[
        "csrc/edetector.c",
        "csrc/conformal.c",
        "csrc/fuser.c",
        "csrc/digest.c",
        "csrc/py_chronosieve.c",
    ],
    include_dirs=["include", "csrc"],
    extra_compile_args=compile_args,
    define_macros=define_macros,
    libraries=libraries,
)

setup(ext_modules=[ext])
