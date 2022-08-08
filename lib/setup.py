from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["viewport2.pyx", "util.pyx"]),
)