# pychange

Outstanding items:

1. Make sure penalties work

1. Documentation

- Costs
    - L1cost
    - _l1_cost
    - L2cost
    - _l2_cost
    - NormalMeanCost
    - _normal_mean_cost
    - NormalVarCost
    - _normal_var_cost
    - NormalMeanVarCost
    - _normal_mean_var_cost
    - PoissonMeanVarCost
    - _poisson_mean_var_cost
    - ExponentialMeanVarCost
    - exponential_mean_var_cost
    - GammaMeanVarCost
    - _gamma_mean_var_cost
    - EmpiricalCost
- Online
    - ConstantHazard
    - StudentTProb
    - _student_t_pdf
    - OnlineCP
- Penalties
    - bic0_penalty
    - bic_penalty
    - mbic_penalty
    - aic0_penalty
    - aic_penalty
    - hq0_penalty
    - hq_penalty
- R
    - ROfflineChangepoint
    - ROCP
- Segment 
    - amoc_segment
    - AmocSeg
    - binary_segment
    - BinSeg
    - pelt_segment
    - PeltSeg

2. Unit testing (maybe)

3. References to works
- Costs
    - L2cost
    - NormalMeanCost
        Change in Normal mean: Hinkley, D. V. (1970) Inference About the Change-Point in a Sequence of Random Variables, Biometrika 57, 1–17
    - NormalVarCost
        - Normal: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser
    - NormalMeanVarCost
        - Change in Normal mean and variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser
    - PoissonMeanVarCost
        - Change in Poisson model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser
    - ExponentialMeanVarCost
        - Change in Exponential model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser
    - GammaMeanVarCost
        - Change in Gamma shape parameter: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser
    - EmpiricalCost
        - Haynes K, Fearnhead P, Eckley IA (2017). “A computationally efficient nonparametric approach for changepoint detection.” Statistics and Computing, 27(5), 1293–1305. ISSN 1573-1375, doi: 10.1007/s1122201696875, https://doi.org/10.1007/s11222-016-9687-5.
- Online
    - OnlineCP
        - Adams, Ryan Prescott, and David JC MacKay. "Bayesian online changepoint detection." arXiv preprint arXiv:0710.3742 (2007).
- Penalties
    - bic0_penalty
    - bic_penalty
        - Schwarz, Gideon E. (1978), "Estimating the dimension of a model", Annals of Statistics, 6 (2): 461–464, doi:10.1214/aos/1176344136, MR 0468014.
    - mbic_penalty
        - MBIC: Zhang, N. R. and Siegmund, D. O. (2007) A Modified Bayes Information Criterion with Applications to the Analysis of Comparative Genomic Hybridization Data. Biometrics 63, 22-32
    - aic0_penalty
    - aic_penalty
        - Akaike, H. (1973), "Information theory and an extension of the maximum likelihood principle", in Petrov, B. N.; Csáki, F. (eds.), 2nd International Symposium on Information Theory, Tsahkadsor, Armenia, USSR, September 2-8, 1971, Budapest: Akadémiai Kiadó, pp. 267–281. Republished in Kotz, S.; Johnson, N. L., eds. (1992), Breakthroughs in Statistics, vol. I, Springer-Verlag, pp. 610–624.
    - hq0_penalty
    - hq_penalty
        - Hannan, E. J., and B. G. Quinn (1979), "The Determination of the order of an autoregression", Journal of the Royal Statistical Society, Series B, 41: 190–195.
- R
    - ROfflineChangepoint
    - ROCP
- Segment 
    - AmocSeg
    - BinSeg
        - Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507–512
    - PeltSeg
        - PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590–1598

4. Pypi package build + submission

6. Benchmarking script

7. Paper

Paper outline
- Abstract
- Introduction
- Literature Review
- Implementation Review
- Pychange
- Future work
- References

8. Presentation

5. Potential additions:
    - FoCusum
        - https://github.com/gtromano/FOCuS
        - https://arxiv.org/abs/2110.08205
