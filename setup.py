# Importing packages
from setuptools import setup
from setuptools import find_packages

if __name__ == '__main__':
    
    # Running setup
    setup(
        name='pychange',
        version='0.0.1',
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        install_requires=['numpy', 'numba', 'icc_rt', 'ruptures', 'rpy2']
    )