"""
Tests whether the output is the same as lavaan's output.
"""
import numpy as np
import pandas as pd
from pytest import approx

from mlsem import MLSEM
from sem import SEM


def test_cfa():
    # CFA test: http://lavaan.ugent.be/tutorial/cfa.html
    # cfa_lavaan/openmx.r

    # get data
    observed = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
    latent = ["visual", "textual", "speed"]
    data = pd.read_csv("tests/HolzingerSwineford1939.csv")
    data = data.loc[:, ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]

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
    sem.fit()
    observed = sem.summary()

    expected = {
        "lambda_y": {
            ("x1", "visual"): 0.900,
            ("x2", "visual"): 0.498,
            ("x3", "visual"): 0.656,
            ("x4", "textual"): 0.990,
            ("x5", "textual"): 1.102,
            ("x6", "textual"): 0.917,
            ("x7", "speed"): 0.619,
            ("x8", "speed"): 0.731,
            ("x9", "speed"): 0.670,
        },
        "psi": {
            ("visual", "textual"): 0.459,
            ("visual", "speed"): 0.471,
            ("textual", "speed"): 0.283,
        },
        "theta": {
            ("x1", "x1"): 0.549,
            ("x2", "x2"): 1.134,
            ("x3", "x3"): 0.844,
            ("x4", "x4"): 0.371,
            ("x5", "x5"): 0.446,
            ("x6", "x6"): 0.356,
            ("x7", "x7"): 0.799,
            ("x8", "x8"): 0.488,
            ("x9", "x9"): 0.566,
        },
    }
    compare(expected, observed, prefix="CFA")


def test_sem_fiml():
    # sem_lavaan/openmx.r
    sem, data = setup_sem("tests/PoliticalDemocracy_missing.csv")
    sem.fit()
    observed = sem.summary()
    expected = {
        "lambda_y": {
            ("x1", "ind60"): 0.661,
            ("x2", "ind60"): 1.461,
            ("x3", "ind60"): 1.209,
            ("y1", "dem60"): 2.085,
            ("y2", "dem60"): 2.573,
            ("y3", "dem60"): 2.304,
            ("y4", "dem60"): 2.532,
            ("y5", "dem65"): 0.513,
            ("y6", "dem65"): 0.627,
            ("y7", "dem65"): 0.694,
            ("y8", "dem65"): 0.683,
        },
        "beta": {
            ("dem60", "ind60"): 0.416,
            ("dem65", "ind60"): 0.884,
            ("dem65", "dem60"): 3.256,
        },
        "theta": {
            ("x1", "x1"): 0.087,
            ("x2", "x2"): 0.059,
            ("x3", "x3"): 0.492,
            ("y1", "y1"): 1.658,
            ("y2", "y2"): 7.805,
            ("y3", "y3"): 4.269,
            ("y4", "y4"): 3.174,
            ("y5", "y5"): 2.609,
            ("y6", "y6"): 4.362,
            ("y7", "y7"): 3.720,
            ("y8", "y8"): 2.277,
            ("y1", "y5"): 0.552,
            ("y2", "y4"): 1.396,
            ("y2", "y6"): 1.155,
            ("y3", "y7"): 1.030,
            ("y4", "y8"): 0.674,
            ("y6", "y8"): 0.808,
        },
        "nu": {
            ("x1", 0): 5.020,
            ("x2", 0): 4.821,
            ("x3", 0): 3.563,
            ("y1", 0): 5.534,
            ("y2", 0): 4.270,
            ("y3", 0): 6.741,
            ("y4", 0): 4.428,
            ("y5", 0): 5.093,
            ("y6", 0): 2.946,
            ("y7", 0): 6.221,
            ("y8", 0): 4.085,
        },
    }
    compare(expected, observed, prefix="SEM_fiml")


def test_mlsem():
    # mlsem_lavaan/openmx.r
    sem, data = setup_mlsem("tests/Demo_twolevel.csv")
    sem.fit()
    observed = sem.summary()
    expected = {
        "lambda_y": {("y1", "fw"): 0.739, ("y2", "fw"): 0.572, ("y3", "fw"): 0.542},
        "beta": {("fw", "_x1"): 0.690, ("fw", "_x2"): 0.551, ("fw", "_x3"): 0.277},
        "theta": {("y1", "y1"): 0.986, ("y2", "y2"): 1.066, ("y3", "y3"): 1.011},
        "lambda_y_l2": {("y1", "fb"): 0.948, ("y2", "fb"): 0.680, ("y3", "fb"): 0.556},
        "beta_l2": {("fb", "_w1"): 0.174, ("fb", "_w2"): 0.138},
        "theta_l2": {("y1", "y1"): 0.058, ("y2", "y2"): 0.120, ("y3", "y3"): 0.149},
        "nu_l2": {("y1", 0): 0.024, ("y2", 0): -0.016, ("y3", 0): -0.042},
    }
    compare(expected, observed, prefix="MLSEM")


def test_sem():
    sem, data = setup_sem("tests/PoliticalDemocracy.csv")
    sem.fit()
    observed = sem.summary(verbose=False)
    expected = {
        "lambda_y": {
            ("x1", "ind60"): 0.670,
            ("x2", "ind60"): 1.460,
            ("x3", "ind60"): 1.218,
            ("y1", "dem60"): 1.989,
            ("y2", "dem60"): 2.500,
            ("y3", "dem60"): 2.104,
            ("y4", "dem60"): 2.516,
            ("y5", "dem65"): 0.415,
            ("y6", "dem65"): 0.492,
            ("y7", "dem65"): 0.531,
            ("y8", "dem65"): 0.526,
        },
        "beta": {
            ("dem60", "ind60"): 0.499,
            ("dem65", "ind60"): 0.923,
            ("dem65", "dem60"): 4.010,
        },
        "theta": {
            ("x1", "x1"): 0.082,
            ("x2", "x2"): 0.120,
            ("x3", "x3"): 0.467,
            ("y1", "y1"): 1.891,
            ("y2", "y2"): 7.373,
            ("y3", "y3"): 5.067,
            ("y4", "y4"): 3.148,
            ("y5", "y5"): 2.351,
            ("y6", "y6"): 4.954,
            ("y7", "y7"): 3.431,
            ("y8", "y8"): 3.254,
            ("y1", "y5"): 0.624,
            ("y2", "y4"): 1.313,
            ("y2", "y6"): 2.153,
            ("y3", "y7"): 0.795,
            ("y4", "y8"): 0.348,
            ("y6", "y8"): 1.356,
        },
    }
    compare(expected, observed, prefix="SEM")


def setup_mlsem(csv):
    # ML-SEM http://lavaan.ugent.be/tutorial/multilevel.html

    # get data
    observed = ["x1", "x2", "x3", "y1", "y2", "y3"]
    observed_l2 = ["y1", "y2", "y3", "w1", "w2"]
    cluster = "cluster"
    latent = ["fw", "_x1", "_x2", "_x3"]
    latent_l2 = ["fb", "_w1", "_w2"]
    data = pd.read_csv(csv)
    data = data.loc[:, np.unique(observed + observed_l2 + [cluster])]
    cdata = data.groupby(cluster).mean()
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
    return sem, data


def setup_sem(csv):
    # SEM test: http://lavaan.ugent.be/tutorial/sem.html

    # get data
    observed = ["x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"]
    latent = ["ind60", "dem60", "dem65"]
    data = pd.read_csv(csv)
    data = data.loc[:, observed]

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
    return sem, data


def compare(expected, observed, prefix="", tol=0.0006):
    for key, matrix in expected.items():
        for index, value in matrix.items():
            assert observed[key].loc[index] == approx(
                value, abs=tol
            ), f"{prefix} {key} ({index}) is {observed[key].loc[index]} but expected {value}"


if __name__ == "__main__":
    test_mlsem()
