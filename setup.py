#!/usr/bin/env python

# The MIT License (MIT)
# 
# Copyright (c) 2015 Peter Iannucci
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

_coreaudio = Extension('audio._coreaudio',
                       ['src/_coreaudio.pyx'],
                       extra_link_args=["-framework", "CoreAudio", "-framework", "CoreFoundation"],
                       )

setup(name='Audio',
      version='1.0.0',
      description='High performance audio I/O',
      author='Peter Iannucci',
      author_email='iannucci@mit.edu',
      url='',
      packages=['audio'],
      ext_modules=[_coreaudio],
      requires=['numpy', 'scipy', 'Cython'],
      include_dirs = [numpy.get_include()],
      cmdclass = {'build_ext': build_ext},
      )

