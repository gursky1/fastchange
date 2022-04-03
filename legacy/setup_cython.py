# Importing packages
import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
#from pychange.costs import cc

if __name__ == '__main__':

    # Compiling numba extensions first
    #cc.compile()
    
    # Running setup
    setup(
        name='pychange_cython',
        version='0.0.1',
        packages=['pychange_cython'],
        ext_modules= cythonize('pychange_cython/cython_costs.pyx'),#cythonize(Extension('cython_costs', ['pychange_cython/cython_costs.pyx'])),
        #install_requires=['Cython', 'numpy', 'numba'],
        include_dirs=[numpy.get_include()]
    )