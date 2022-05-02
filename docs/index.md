# Welcome to the Fastchange documentation!

Fastchange is a change point detection library that is intended to be extendable, interpretable, and above all else, fast! We acheive this by adhering to the Scikit-learn API and using Numba, a just-in-time compiler that allows developers to write pure Python with NumPy that is then compiled to fast C code. 

## What's in Fastchange?

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