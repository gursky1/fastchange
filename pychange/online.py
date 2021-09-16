# Importing packages
import math
import numpy as np
import numba as nb

# Hazard function
@nb.experimental.jitclass([
    ('l', nb.float64),
    ('lt', nb.float64)
])
class ConstantHazard:
    
    def __init__(self, l=100):
        self.l = l
        self.lt = 1.0 / l
    
    def h(self, x):
        return np.repeat(self.lt, x)

# Gaussian probability
@nb.njit(fastmath=True)
def student_t_pdf(x, df, loc, scale):
    y = (x - loc) / scale
    c0 = (df + 1) / 2
    c1 = math.lgamma(c0)
    c2 = math.lgamma(df / 2)
    c3 = math.exp(c1 - c2)
    d1 = math.sqrt(math.pi * df)
    d2 = 1 + y ** 2 / df
    d3 = d1 * d2 ** c0
    r = c3 / d3
    return r / scale


@nb.experimental.jitclass([
    ('mu0', nb.float64),
    ('kappa0', nb.float64),
    ('alpha0', nb.float64),
    ('beta0', nb.float64),
    ('muT', nb.float64[:]),
    ('kappaT', nb.float64[:]),
    ('alphaT', nb.float64[:]),
    ('betaT', nb.float64[:]),
    ('n', nb.int64)
])
class StudentTProb:
    
    def __init__(self, m=0., k=0.01, a=0.01, b=0.0001):

        self.mu0 = m
        self.kappa0 = k
        self.alpha0 = a
        self.beta0 = b
        self.reset()
        
    def reset(self):
        
        self.muT = np.array([self.mu0])
        self.kappaT = np.array([self.kappa0])
        self.alphaT = np.array([self.alpha0])
        self.betaT = np.array([self.beta0])
        self.n = 0
        
        
    def pdf(self, x):
        n = self.n + 1
        results = np.empty((n,), dtype=np.float64)
        for i in range(n):
            loc = self.muT[i]
            var = self.betaT[i]* (self.kappaT[i] + 1) / (self.alphaT[i] * self.kappaT[i])
            scale = math.sqrt(var)
            df = 2.0 * self.alphaT[i]
            results[i] = student_t_pdf(x, df, loc, scale)
        return results
    
    def update(self, x):
        
        # Generating arrays for new values
        bT2 = np.empty(self.betaT.shape[0] + 1, dtype=np.float64)
        muT2 = np.empty(self.muT.shape[0] + 1, dtype=np.float64)
        kappaT2 = np.empty(self.kappaT.shape[0] + 1, dtype=np.float64)
        alphaT2 = np.empty(self.alphaT.shape[0] + 1, dtype=np.float64)
        
        # Populating initial values
        bT2[0] = self.beta0
        muT2[0] = self.mu0
        kappaT2[0] = self.kappa0
        alphaT2[0] = self.alpha0
        
        # Beta update
        bT2[1:] = (self.betaT + (self.kappaT * (x - self.muT) ** 2) / (2.0 * (self.kappaT + 1.)))
        
        # Mu update
        muT2[1:] = (self.kappaT * self.muT + x) / (self.kappaT + 1)
        
        # Kappa update
        kappaT2[1:] = self.kappaT + 1.0
        
        # Alpha update
        alphaT2[1:] = self.alphaT + 0.5
        
        # Updating internal values
        self.betaT = bT2
        self.muT = muT2
        self.kappaT = kappaT2
        self.alphaT = alphaT2
        
        # Updating size counter
        self.n += 1
        
        return self


@nb.experimental.jitclass([
    ('mu0', nb.float64),
    ('kappa0', nb.float64),
    ('alpha0', nb.float64),
    ('beta0', nb.float64),
    ('muT', nb.float64[:]),
    ('kappaT', nb.float64[:]),
    ('alphaT', nb.float64[:]),
    ('betaT', nb.float64[:]),
    ('n', nb.int64)
])
class StudentTProbParallel:
    
    def __init__(self, m=0., k=0.01, a=0.01, b=0.0001):

        self.mu0 = m
        self.kappa0 = k
        self.alpha0 = a
        self.beta0 = b
        self.reset()
        
    def reset(self):
        
        self.muT = np.array([self.mu0])
        self.kappaT = np.array([self.kappa0])
        self.alphaT = np.array([self.alpha0])
        self.betaT = np.array([self.beta0])
        self.n = 0
        
    def pdf(self, x):
        return _student_t_pdf(x, self.n, self.muT, self.kappaT, self.alphaT, self.betaT)
    
    def update(self, x):
        
        # Generating arrays for new values
        bT2 = np.empty(self.betaT.shape[0] + 1, dtype=np.float64)
        muT2 = np.empty(self.muT.shape[0] + 1, dtype=np.float64)
        kappaT2 = np.empty(self.kappaT.shape[0] + 1, dtype=np.float64)
        alphaT2 = np.empty(self.alphaT.shape[0] + 1, dtype=np.float64)
        
        # Populating initial values
        bT2[0] = self.beta0
        muT2[0] = self.mu0
        kappaT2[0] = self.kappa0
        alphaT2[0] = self.alpha0
        
        # Beta update
        bT2[1:] = (self.betaT + (self.kappaT * (x - self.muT) ** 2) / (2.0 * (self.kappaT + 1.)))
        
        # Mu update
        muT2[1:] = (self.kappaT * self.muT + x) / (self.kappaT + 1)
        
        # Kappa update
        kappaT2[1:] = self.kappaT + 1.0
        
        # Alpha update
        alphaT2[1:] = self.alphaT + 0.5
        
        # Updating internal values
        self.betaT = bT2
        self.muT = muT2
        self.kappaT = kappaT2
        self.alphaT = alphaT2
        
        # Updating size counter
        self.n += 1
        
        return self


@nb.njit(fastmath=True, nogil=True)
def _student_t_pdf(x, n, m, k, a, b):
    n = n + 1
    results = np.empty((n,), dtype=np.float64)
    for i in nb.prange(n):
        loc = m[i]
        var = b[i]* (k[i] + 1) / (a[i] * k[i])
        scale = math.sqrt(var)
        df = 2.0 * a[i]  
        y = (x - loc) / scale
        c0 = (df + 1) / 2
        c1 = math.lgamma(c0)
        c2 = math.lgamma(df / 2)
        c3 = math.exp(c1 - c2)
        d1 = math.sqrt(math.pi * df)
        d2 = 1 + y ** 2 / df
        d3 = d1 * d2 ** c0
        r = c3 / d3
        results[i] = r / scale
    return results


# OnlineCP class
@nb.experimental.jitclass([
    ('hazard', nb.typeof(ConstantHazard())),
    ('prob_model', nb.typeof(StudentTProb())),
    ('wait_iters', nb.int64),
    ('cp_threshold', nb.float64),
    ('cp_probs', nb.typeof([np.float64([1.0])])),
    ('n_steps', nb.int64)
])
class OnlineCP:
    """https://arxiv.org/abs/0710.3742"""
    
    def __init__(self, hazard, prob_model, wait_iters, cp_threshold):
        self.hazard = hazard
        self.prob_model = prob_model
        self.wait_iters = wait_iters
        self.cp_threshold = cp_threshold
        
        self.cp_probs = [np.float64([1.0])]
        self.n_steps = 0
    
    def update(self, x):
        
        # Iterating over data
        for i in x:
        
            # Getting last probability array
            last_cp_probs = self.cp_probs[self.n_steps]

            # Getting likelihoods for each observation
            probs = self.prob_model.pdf(i)

            # Iterating number of steps seen
            self.n_steps += 1

            # Calculating hazard function
            h = self.hazard.h(self.n_steps)
            
            # Updating growth probabilities
            _cp_m = last_cp_probs * probs
            cp_probs = np.empty(self.n_steps + 1, dtype=np.float64)
            cp_probs[1:] = _cp_m * (1 - h)
            cp_probs[0] = np.sum(_cp_m * h)
            cp_probs /= np.sum(cp_probs)

            # Upating likelihood function
            self.prob_model.update(i)
            
            # Adding probabilities to list
            self.cp_probs.append(cp_probs)
        
        return self
    
    def get_probs(self):
        # Getting cp probability values
        p_size = len(self.cp_probs) - self.wait_iters - 1
        prob_vals = np.empty(p_size, dtype=np.float64)
        for i, j in enumerate(self.cp_probs[self.wait_iters: -1]):
            if i == 0:
                prob_vals[0] = 0
            else:
                prob_vals[i] = j[self.wait_iters]
        return prob_vals
    
    def get_cps(self):
        
        prob_vals = self.get_probs()
        p_size = prob_vals.shape[0]
        # Determining if changepoints meet threshold
        cps = []
        for i in range(1, p_size):

            if prob_vals[i] < prob_vals[i - 1]:
                continue

            if i < p_size - 1:
                if prob_vals[i] < prob_vals[i + 1]:
                    continue

            if prob_vals[i] >= self.cp_threshold:
                cps.append(i)

        return cps

@nb.experimental.jitclass([
    ('hazard', nb.typeof(ConstantHazard())),
    ('prob_model', nb.typeof(StudentTProbParallel())),
    ('wait_iters', nb.int64),
    ('cp_threshold', nb.float64),
    ('cp_probs', nb.typeof([np.float64([1.0])])),
    ('n_steps', nb.int64)
])
class OnlineCPParallel:
    """https://arxiv.org/abs/0710.3742"""
    
    def __init__(self, hazard, prob_model, wait_iters, cp_threshold):
        self.hazard = hazard
        self.prob_model = prob_model
        self.wait_iters = wait_iters
        self.cp_threshold = cp_threshold
        
        self.cp_probs = [np.float64([1.0])]
        self.n_steps = 0
    
    def update(self, x):
        
        # Iterating over data
        for i in x:
        
            # Getting last probability array
            last_cp_probs = self.cp_probs[self.n_steps]

            # Getting likelihoods for each observation
            probs = self.prob_model.pdf(i)

            # Iterating number of steps seen
            self.n_steps += 1

            # Calculating hazard function
            h = self.hazard.h(self.n_steps)
            
            # Updating growth probabilities
            _cp_m = last_cp_probs * probs
            cp_probs = np.empty(self.n_steps + 1, dtype=np.float64)
            cp_probs[1:] = _cp_m * (1 - h)
            cp_probs[0] = np.sum(_cp_m * h)
            cp_probs /= np.sum(cp_probs)

            # Upating likelihood function
            self.prob_model.update(i)
            
            # Adding probabilities to list
            self.cp_probs.append(cp_probs)
        
        return self
    
    def get_probs(self):
        # Getting cp probability values
        p_size = len(self.cp_probs) - self.wait_iters - 1
        prob_vals = np.empty(p_size, dtype=np.float64)
        for i, j in enumerate(self.cp_probs[self.wait_iters: -1]):
            if i == 0:
                prob_vals[0] = 0
            else:
                prob_vals[i] = j[self.wait_iters]
        return prob_vals
    
    def get_cps(self):
        
        prob_vals = self.get_probs()
        p_size = prob_vals.shape[0]
        # Determining if changepoints meet threshold
        cps = []
        for i in range(1, p_size):

            if prob_vals[i] < prob_vals[i - 1]:
                continue

            if i < p_size - 1:
                if prob_vals[i] < prob_vals[i + 1]:
                    continue

            if prob_vals[i] >= self.cp_threshold:
                cps.append(i)

        return cps