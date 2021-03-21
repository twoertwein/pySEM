# pySEM
This is a python implementation to fit Structural Equation Models (SEM). It follows the LISREL notation. Similar to [tensorsem](https://github.com/vankesteren/tensorsem), the SEM parameters are estimated with gradient descend (pytorch is used in pySEM).

## SEM features
The following SEM features are supported:

 * Only continuous variables are allowed (no ordinal and categorical variables);
 * Missing data (using Full Information Maximum Likelihood);
 * Support for mean structure;
 * Two-level SEM (implemented as described [here](https://doi.org/10.1007/978-0-387-73186-5_12)).

## Installation
```sh
poetry add git+https://github.com/twoertwein/pySEM
```

## Usage
To define an SEM, pySEM needs to know which parameters are free and which are fixed to a constant value. A SEM has the following parameters:

Parameters | What it represents | Default
--- | --- | ---
psi | Covariance between latent variables | Variance is fixed to 1.0 (and psi_ij is 0.0 if beta_ij is free/non-zero)
lambda_y | Factor loadings | Fixed to 0.0
beta | Latent regression | Fixed to 0.0
theta | Covariance of residuals from observed variables | Uncorrelated residuals (diagonal is free)
alpha | Mean of latent variables | Fixed to 0.0*
nu | Mean of observed variables | Free*

*Means are only estimated if the user request them, data is missing, or a two-level SEM is used.

A dictionary for each parameter is used to specify which entries are fixed/free (the dictionary key is a tuple of two strings (or only one string for means) and a float is used as value (NaN indicates that the entry is free)).


### Example CFA
Fit the [Holzinger Swineford CFA](http://lavaan.ugent.be/tutorial/cfa.html) and constrain the variance of the factors to be 1 (instead of having one factor loading fixed to 1).
```py
import pandas as pd
from pysem.sem import SEM

# get data
observed = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
latent = ["visual", "textual", "speed"]
data = pd.read_csv("tests/HolzingerSwineford1939.csv").loc[:, observed]

# define SEM
sem = SEM(
    data=data,
    observed=observed,
    latent=latent,
    # fixed standard deviation of latent variables
    psi={(i, j): float("NaN") for i in latent for j in latent if i != j},
    lambda_y={
        # visual
        ("x1", "visual"): float("NaN"),
        ("x2", "visual"): float("NaN"),
        ("x3", "visual"): float("NaN"),
        # textual
        ("x4", "textual"): float("NaN"),
        ("x5", "textual"): float("NaN"),
        ("x6", "textual"): float("NaN"),
        # speed
        ("x7", "speed"): float("NaN"),
        ("x8", "speed"): float("NaN"),
        ("x9", "speed"): float("NaN"),
    },
)

# fit SEM
sem.fit()
parameters = sem.summary()  # prints a summary and returns a dictionary of DataFrames containing the parameter estimations
```

### Example SEM
The [PoliticalDemocracy SEM from Bollen](http://lavaan.ugent.be/tutorial/sem.html) adds regressions between latent variables.
```py
import pandas as pd
from pysem.sem import SEM

# get data
observed = ["x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"]
latent = ["ind60", "dem60", "dem65"]
data = pd.read_csv("tests/PoliticalDemocracy.csv").loc[:, observed]

# define SEM
sem = SEM(
    data=data,
    observed=observed,
    latent=latent,
    # correlated residuals
    theta={
        ("y1", "y5"): float("NaN"),
        ("y2", "y4"): float("NaN"),
        ("y2", "y6"): float("NaN"),
        ("y3", "y7"): float("NaN"),
        ("y4", "y8"): float("NaN"),
        ("y6", "y8"): float("NaN"),
    },
    # latent regression
    beta={
        ("dem60", "ind60"): float("NaN"),
        ("dem65", "dem60"): float("NaN"),
        ("dem65", "ind60"): float("NaN"),
    },
    lambda_y={
        ("x1", "ind60"): float("NaN"),
        ("x2", "ind60"): float("NaN"),
        ("x3", "ind60"): float("NaN"),
        ("y1", "dem60"): float("NaN"),
        ("y2", "dem60"): float("NaN"),
        ("y3", "dem60"): float("NaN"),
        ("y4", "dem60"): float("NaN"),
        ("y5", "dem65"): float("NaN"),
        ("y6", "dem65"): float("NaN"),
        ("y7", "dem65"): float("NaN"),
        ("y8", "dem65"): float("NaN"),
    },
)
```

### Example Two-level SEM
This [example two-level SEM from lavaan](http://lavaan.ugent.be/tutorial/multilevel.html) accounts for differences between clusters (level two). SEM solvers following the LISREL notation cannot have explicit regressions from an observed variable to a latent variable. Definition variables need to be introduced (latent variables which mirror the observed variable). lavaan does that internally. pySEM requires them to be defined by the user. In the following example, definition variables are prefixed with "_". Since lavaan uses the biased (co-)variance by default, the (co-)variance is scaled to reflect that.

```py
import numpy as np
import pandas as pd
from pysem.mlsem import MLSEM


# get data
observed = ["x1", "x2", "x3", "y1", "y2", "y3"]
observed_l2 = ["y1", "y2", "y3", "w1", "w2"]
cluster = "cluster"
latent = ["fw", "_x1", "_x2", "_x3"]
latent_l2 = ["fb", "_w1", "_w2"]
data = pd.read_csv("tests/Demo_twolevel.csv")
data = data.loc[:, np.unique(observed + observed_l2 + [cluster])]
cdata = data.groupby(cluster).mean()

# scale the variance to be biased
var = (data.shape[0] - 1) / data.shape[0]
cvar = (cdata.shape[0] - 1) / cdata.shape[0]

# define SEM
sem = MLSEM(
    data=data,
    observed=observed,
    latent=latent,
    observed_l2=observed_l2,
    latent_l2=latent_l2,
    cluster=cluster,
    lambda_y={
        ("y1", "fw"): float("NaN"),
        ("y2", "fw"): float("NaN"),
        ("y3", "fw"): float("NaN"),
        ("x1", "_x1"): 1.0,
        ("x2", "_x2"): 1.0,
        ("x3", "_x3"): 1.0,
    },
    theta={("x1", "x1"): 0.0, ("x2", "x2"): 0.0, ("x3", "x3"): 0.0},
    beta={
        ("fw", "_x1"): float("NaN"),
        ("fw", "_x2"): float("NaN"),
        ("fw", "_x3"): float("NaN"),
    },
    psi={
        ("_x1", "_x1"): data["x1"].var() * var,
        ("_x2", "_x2"): data["x2"].var() * var,
        ("_x3", "_x3"): data["x3"].var() * var,
        ("_x1", "_x2"): data.loc[:, ["x1", "x2"]].cov().loc["x1", "x2"] * var,
        ("_x1", "_x3"): data.loc[:, ["x1", "x3"]].cov().loc["x1", "x3"] * var,
        ("_x3", "_x2"): data.loc[:, ["x3", "x2"]].cov().loc["x3", "x2"] * var,
   },
   alpha={
       "_x1": data["x1"].mean(),
       "_x2": data["x2"].mean(),
       "_x3": data["x3"].mean(),
    },
    lambda_y_l2={
       ("y1", "fb"): float("NaN"),
       ("y2", "fb"): float("NaN"),
       ("y3", "fb"): float("NaN"),
       ("w1", "_w1"): 1.0,
       ("w2", "_w2"): 1.0,
   },
   beta_l2={("fb", "_w1"): float("NaN"), ("fb", "_w2"): float("NaN")},
   theta_l2={("w1", "w1"): 0.0, ("w2", "w2"): 0.0},
   psi_l2={
        ("_w1", "_w1"): cdata["w1"].var() * cvar,
        ("_w2", "_w2"): cdata["w2"].var() * cvar,
        ("_w1", "_w2"): cdata.loc[:, ["w1", "w2"]].cov().loc["w1", "w2"] * cvar,
    },
    alpha_l2={"_w1": cdata["w1"].mean(), "_w2": cdata["w2"].mean()},
    nu_l2={"w1": 0.0, "w2": 0.0},
)
```
