# Importing packages
import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize
from pychange.costs import cc

if __name__ == '__main__':

    # Creating extensions
    # Numba
    cc.compile()
    #numba_exts = [cc.distutils_extension()]

    # Cython
    #cython_exts = cythonize(Extension('cython_costs', ['pychange/cython_costs.pyx']))
    
    # Running setup
    setup(
        name='pychange',
        version='0.0.1',
        packages=['pychange'],
        ext_modules= [cc.distutils_extension()],
        include_dirs=[numpy.get_include()]
                     #cythonize(Extension('cython_costs',
                     #                    ["pychange/cython_costs.pyx"],
                     #                    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
                     #                    include_path = [numpy.get_include()])],
    )