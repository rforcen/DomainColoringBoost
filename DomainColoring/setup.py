from distutils.core import setup
from distutils.extension import Extension

DomainColoring = Extension(
    'DomainColoring',
    sources=['domainColoring.cpp'],
    libraries=['boost_python37-mt', 'boost_numpy37-mt'],
    extra_compile_args=['-std=c++11'] # lambda support required
)

setup(
    name='DomainColoring',
    version='0.1',
    ext_modules=[DomainColoring])

#call with: python3.7 setup.py build_ext --inplace