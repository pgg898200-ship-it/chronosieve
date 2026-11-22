import sys

from setuptools import Extension, setup

compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17", "-O3", "-g0"]

setup(
    packages=["guardshift"],
    package_dir={"": "python"},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    ext_modules=[
        Extension(
            "guardshift._native",
            ["python/guardshift/_native.cpp"],
            include_dirs=["include"],
            depends=["include/guardshift.hpp"],
            define_macros=[("Py_LIMITED_API", "0x03090000")],
            py_limited_api=True,
            language="c++",
            extra_compile_args=compile_args,
        )
    ],
)
