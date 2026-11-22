import sys
from setuptools import Extension, setup

setup(
    packages=["guardshift"],
    package_dir={"": "python"},
    ext_modules=[Extension(
        "guardshift._native", ["python/guardshift/_native.cpp"],
        include_dirs=["include"], depends=["include/guardshift.hpp"], language="c++",
        extra_compile_args=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17", "-O3"],
    )],
)
