import os

os.environ["CC"] = "clang"
os.environ["LDSHARED"] = "clang -shared"

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from Cython.Distutils import Extension
from setuptools import setup

Options.annotate = True
# Options.docstrings = False

# cython --annotate -c=-Ofast -c=-ffast-math -c=-mfpmath=sse -c=-funroll-loops -c=-march=native
# cython: boundscheck=False, wraparound=False, cdivision=True

setup(
    name="Weights",
    ext_modules=cythonize(
        [
            Extension(
                "weights",
                sources=["weights.pyx"],
                cython_directives={
                    # "language_level": "3",
                    # "boundscheck": False,
                    # "wraparound": False,
                    # "cdivision": True,
                },
                extra_compile_args=[
                    # "-Ofast",
                    # "-ffast-math",
                    # "-funroll-loops",
                    # "-march=native",
                    #
                    # "-mfpmath=sse",
                    # "-mllvm",
                    # "-force-vector-width=8",
                    # "-S",
                    # "-O3"
                ],
            )
            #   language='c++')
        ]
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)


# python setup.py build_ext --inplace
# (export CC=clang;) python setup.py build_ext --inplace

# LLVM
# clang -S -emit-llvm foo.c
# clang -cc1 foo.c -emit-llvm
# llc foo.ll

# clang -fPIC -O2 -isystem /home/michal/anaconda3/envs/ml310/include -fPIC -O2 -isystem /home/michal/anaconda3/envs/ml310/include -fPIC -I/home/michal/anaconda3/envs/ml310/lib/python3.10/site-packages/numpy/core/include -I/home/michal/anaconda3/envs/ml310/include/python3.10 -S -emit-llvm weights.c
# -mllvm -force-vector-width=8 -march=native

# ASM
# clang -fPIC -O2 -isystem /home/michal/anaconda3/envs/ml310/include -fPIC -O2 -isystem /home/michal/anaconda3/envs/ml310/include -fPIC -I/home/michal/anaconda3/envs/ml310/lib/python3.10/site-packages/numpy/core/include -I/home/michal/anaconda3/envs/ml310/include/python3.10 -S weights.c
# clang -fPIC -O2 -isystem /home/michal/anaconda3/envs/ml310/include -fPIC -O2 -isystem /home/michal/anaconda3/envs/ml310/include -fPIC -I/home/michal/anaconda3/envs/ml310/lib/python3.10/site-packages/numpy/core/include -I/home/michal/anaconda3/envs/ml310/include/python3.10 -S -Ofast -ffast-math -funroll-loops -march=native weights.c

# https://tbetcke.github.io/hpc_lecture_notes/simd.html
# https://stackoverflow.com/questions/49058949/generating-simd-instructions-from-cython-code
# https://stackoverflow.com/questions/14492436/g-optimization-beyond-o3-ofast

# SSE: xmmX
# AVX/AVX2: ymmX
# AVX-512: zmmX
