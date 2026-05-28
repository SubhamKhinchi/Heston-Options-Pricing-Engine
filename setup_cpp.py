"""
Build the Heston MCS C++ extension.

    python setup_cpp.py build_ext --inplace

The compiled module lands at pricing/heston_mcs*.so and is imported
automatically by pricing/heston_pde_american.py when available.
"""

from setuptools import Extension, setup

import pybind11

ext = Extension(
    "pricing.heston_mcs",
    sources=["pricing/cpp/heston_mcs.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-ffast-math", "-std=c++17"],
    language="c++",
)

setup(
    name="heston_mcs_cpp",
    version="1.0.0",
    ext_modules=[ext],
)
