# https://docs.python.org/3/distutils/apiref.html
from distutils.core import setup
from distutils.extension import Extension

# import os
# os.environ['LDFLAGS'] = '-framework Accelerate'

DomainColoring = Extension(
    'DomainColoring',
    sources=['domainColoring.cpp'],
    include_dirs=['/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Headers/'],
    libraries=['boost_python37-mt', 'boost_numpy37-mt'],
    extra_compile_args=['-std=c++11'] # lambda support required
)

setup(
    name='DomainColoring',
    version='0.1',
    ext_modules=[DomainColoring])

#call with: python3.7 setup.py build_ext --inplace