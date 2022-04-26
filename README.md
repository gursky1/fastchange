# fastchange: Fast change point detection in Python

# Quickstart

## Install fastchange

Fastchange can be installed in three different ways: conda, pip, or directly through Git. Conda is the preferred method of installation as it provides faster scipy/numpy distributions that pip, and offers the icc_rt library that can be used for an additional performance increase (see **Installing with extras**).

**1. Via conda (preferred)**
`conda install fastchange`

**2. Via pip**
`pip install fastchange`

**3. Via git**
`pip install git+https://github.com/gursky1/fastchange`

## Example notebook

See example.ipynb under examples/ for some sample code to get started with fastchange.

# Installing with extras

**icc_rt**

As per the numba documentation, you can get a performance increase using Intel's SVML library `icc_rt`.  Note that this is only available via numba's conda channel:

`conda install -c numba icc_rt`

To install fastchange with icc_rt out of the box, install with the "svml" extra:

`pip/conda install fastchange[svml]`

**rpy2**

Fastchange also offers an interface to several R changepoint libraries using the rpy2 package. Note that this requires a local installation of R to function. Fastchange provides integration with the changepoint, changepoint.np, and bocp R libraries. Note these also need to be manually installed to function. The R interface can be installed using the "r" extra keyword:

`pip/conda install fastchange[r]`

