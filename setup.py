from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension("intfft", ["src/intfft.cpp"], cxx_std=17)
]

setup(
    name="intfft",
    version="0.1.0",
    author="fukuroda",
    author_email="fukuroder@live.jp",
    url="https://github.com/fukuroder",
    description="Integer FFT(Fast Fourier Transform) in Python",
    long_description="Integer FFT(Fast Fourier Transform) in Python",
    ext_modules=ext_modules,
)
