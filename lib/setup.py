from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["lib/util.pyx", "lib/transform.pyx"]),
)