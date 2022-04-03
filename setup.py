# Importing packages
from setuptools import setup

if __name__ == '__main__':
    
    # Running setup
    setup(
        name='pychange',
        version='0.0.1',
        packages=['pychange'],
        install_requires=['numpy', 'numba', 'icc_rt', 'ruptures', 'rpy2']
    )