# fastchange: Fast change point detection in Python

Fastchange is a change point detection library that is intended to be extendable, interpretable, and above all else, fast! We acheive this by adhering to the Scikit-learn API and using Numba, a just-in-time compiler that allows developers to write pure Python with NumPy that is then compiled to fast C code. 

# Quickstart

## Install fastchange

Fastchange can be installed in three different ways: conda, pip, or directly through Git. Conda is the preferred method of installation as it provides faster scipy/numpy distributions that pip, and offers the icc_rt library that can be used for an additional performance increase (see **Installing with extras**).

**1. Via conda (preferred)**
`conda install fastchange`

**2. Via pip**
`pip install fastchange`

**3. Via git**
`pip install git+https://github.com/gursky1/fastchange`

## Example notebooks

See the examples directory for some sample code to get started with fastchange.

# What's in Fastchange?

Fastchange currently implements some the most popular offline change point detection methods, including:

**Segmentation methods**

- Single, exact change point: `fastchange.seg.AmocSeg`

- Multiple, approximate change points: `fastchange.seg.BinSeg`

- Multple, exact change points: `fastchange.seg.PeltSeg`

**Cost functions**

- Normal mean and/or variance change: `fastchange.costs.normal`

- Poisson mean+variance change: `fastchange.costs.poisson.PoissonMeanVarCost`

- Poisson mean+variance change: `fastchange.costs.poisson.PoissonMeanVarCost`

- Poisson mean+variance change: `fastchange.costs.poisson.PoissonMeanVarCost`

- Nonparametric cost: `fastchange.costs.emp.EmpiricalCost`

**Penalty functions**

- MBIC: `fastchange.penalties.mbic_penalty`

- BIC: `fastchange.penalties.bic_penalty`

- AIC: `fastchange.penalties.aic_penalty`

- Hannan-Quinn: `fastchange.penalties.hq_penalty`

# Installing with extras

**svml**

As per the numba documentation, you can get a performance increase using Intel's SVML library `icc_rt`.  Note that this is only available via numba's conda channel:

`conda install -c numba icc_rt`

To install fastchange with icc_rt out of the box, install with the "svml" extra:

`pip/conda install fastchange[svml]`

**r**

Fastchange also offers an interface to several R changepoint libraries using the rpy2 package. Note that this requires a local installation of R to function. Fastchange provides integration with the changepoint, changepoint.np, and bocp R libraries. Note these also need to be manually installed to function. The R interface can be installed using the "r" extra keyword:

`pip/conda install fastchange[r]`

# References