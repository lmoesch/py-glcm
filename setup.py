import numpy as np
from distutils.core import setup, Extension

glcm = Extension('glcm',
                    sources=['py-glcm/core/src/glcm.cpp'],
                    include_dirs=[np.get_include()])

setup(name='py-glcm',
      version='1.1a',
      description='py-glcm provides native implementations of GLCM related functions.',
      include_dirs=[np.get_include()],
      ext_modules=[glcm])


