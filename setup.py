# Importing packages
from setuptools import setup
from setuptools import find_packages

if __name__ == '__main__':
    
    # Running setup
    setup(
        name='fastchange',
        version='0.0.1a',
        package_dir={'': 'fastchange'},
        packages=find_packages(where='fastchange'),
        install_requires=['numpy', 'numba'],
        extras_require={
            'r': ['rpy2'],
            'svml': ['icc_rt']
        }
    )
