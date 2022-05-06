from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["lib/util.pyx", "lib/transform.pyx", "lib/viewport.pyx"],
                          language_level=3),
)