from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
# import sys
import os
from glob import glob

if os.name == 'posix' and os.uname().machine =='arm64':
    print("Building on osx-arm64")
    if os.environ.get("CC") in [None, 'gcc','clang']:
        gcc_fname = glob("/opt/homebrew/opt/gcc/bin/gcc-[0-9]*")
        assert len(gcc_fname) > 0, "but no gcc is found in the default homebrew path. Please assign environment variable CC"
        os.environ["CC"] = gcc_fname[0]
    print(f"CC = {os.environ.get('CC')}")

cargs = ["-fopenmp"]
# cargs[0] = "-openmp" # on Windows with Visual C/C++ compiler

extensions = [Extension("ccorr",["src/ccorr.pyx"], 
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                        extra_compile_args=cargs,
                        extra_link_args=cargs)]

setup(
    name="ccorr",
    ext_modules=cythonize(extensions), 
    include_dirs=[np.get_include()]
)