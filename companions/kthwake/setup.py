import sys
from setuptools import Extension, setup
flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
setup(name="kthwake", version="0.1.0", description="Exact event-time threshold transitions",
      license="Apache-2.0", url="https://pgg-kthwake.tancurl3.chatgpt.site",
      project_urls={"Source": "https://huggingface.co/spaces/pgg-engineering/kthwake/tree/main"},
      python_requires=">=3.8", packages=["kthwake"], package_dir={"": "python"},
      ext_modules=[Extension("kthwake._native", ["python/native.cpp"],
                             include_dirs=["include"], depends=["include/kthwake.hpp"],
                             language="c++", extra_compile_args=flags)])
