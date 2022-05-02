#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
import math


def bic_penalty(n: int, n_params: int) -> float:
    """Segment cost penalty based on Bayesian/Schwarz information criterion (BIC/SIC).
    
    Gideon Schwarz. “Estimating the Dimension of a Model”. In: The Annals of Statistics 6.2 (1978), pp. 461-464. doi: 10.1214/aos/ 1176344136. url: https://doi.org/10.1214/aos/1176344136

    Args:
        n (int): Number of change points in model
        n_params (int): Number of parameters estimated in cost function.

    Returns:
        float: Complexity penalty for segment
    """
    return (n_params + 1) * math.log(n)


def mbic_penalty(n: int, n_params: int) -> float:
    """Segment cost penalty based on the modified Bayesian information criterion (MBIC).
    
    Nancy R. Zhang and David O. Siegmund. “A modified Bayes information criterion with applications to the analysis of comparative genomic hybridization data”. In: Biometrics 63 (2007), pp. 22-32.

    Args:
        n (int): Number of change points in model
        n_params (int): Number of parameters estimated in cost function.

    Returns:
        float: Complexity penalty for segment
    """
    return (n_params + 2) * math.log(n)


def aic_penalty(n: int, n_params: int) -> float:
    """Segment cost penalty based on Akaike information criterion (AIC).
    
    H. Akaike. “A new look at the statistical model identification”. In: IEEE Transactions on Automatic Control 19.6 (1974), pp. 716-723.doi: 10.1109/TAC.1974.1100705

    Args:
        n (int): Number of change points in model (unused).
        n_params (int): Number of parameters estimated in cost function.

    Returns:
        float: Complexity penalty for segment
    """
    return 2 * (n_params + 1)


def hq_penalty(n: int, n_params: int) -> float:
    """Segment cost penalty based on Hannan-Quinn information criterion (HQIC).
    
    Antonio Aznar Grasa. “Introduction”. In: Econometric Model Selection: A New Approach. Dordrecht: Springer Netherlands, 1989,pp. 1-4. isbn: 978-94-017-1358-0. doi: 10.1007/978-94-017-1358-0_1. url: https://doi.org/10.1007/978-94-017-1358-0_1

    Args:
        n (int): Number of change points in model
        n_params (int): Number of parameters estimated in cost function.

    Returns:
        float: Complexity penalty for segment
    """
    return 2 * (n_params + 1) * math.log(math.log(n))
