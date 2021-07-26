# Importing packages
import numpy
from setuptools import setup
#from setuptools.extension import Extension
#from Cython.Build import cythonize
from pychange.costs import cc

if __name__ == '__main__':

    # Compiling numba extensions first
    cc.compile()
    
    # Running setup
    setup(
        name='pychange',
        version='0.0.1',
        packages=['pychange'],
        ext_modules= [cc.distutils_extension()],# + cythonize(Extension('cython_costs', ['pychange/cython_costs.pyx'])),
        install_requires=['numpy', 'numba', 'ruptures' ,'cupy'],
        #include_dirs=[numpy.get_include()]
    )