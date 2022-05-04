# Importing packages
from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    
    # Running setup
    setup(
        name='fastchange',
        version='1.0.2',
        author="Jacob Gursky",
        author_email='gurskyjacob@gmail.com',
        description="Fast change point detection in Python",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        install_requires=['numpy', 'numba'],
        extras_require={
            'r': ['rpy2'],
            'svml': ['icc_rt']
        }
    )
