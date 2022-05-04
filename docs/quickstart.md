# Quickstart


## Installation

First install fastchange:

`conda install -c gursky1 fastchange`

Or if you prefer pip:

`pip install fastchange`

## Usage

Example usage on an artificial signal with a change point at index 100:

```python
# Importing packages
import numpy as np
from fastchange.seg.amoc import AmocSeg
from fastchange.costs.normal import NormalMeanVarCost
from fastchange.penalties import mbic_penalty

# Creating synthetic data
data = np.hstack([
    np.random.normal(0, 1, (100,)),
    np.random.normal(10, 2, (100,))
])

# Running AMOC changepoint
model = AmocSeg(NormalMeanVarCost(), mbic_penalty).fit(data)
cpts = model.predict()
```