from itertools import product

import numpy as np
import pandas as pd

from supu import dataframes


def test_multilevel():
    params = list(product(list(10 ** np.arange(9)), [0, 2, 9, 20], [0, 1, 2]))
    rng = np.random.default_rng(0)

    basis = [np.log(x + 1) * (z + y / 5) for x, y, z in params]
    base_df = pd.DataFrame({"result": basis * 100}, index=params * 100)
    perturbed = base_df + rng.normal(size=base_df.shape)

    desc = dataframes.describe_with_ci(perturbed)

    assert list(desc.index) == params
    assert list(desc.columns) == [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "lower_ci",
        "upper_ci",
    ]
    # We should have some values outside the confidence interval, but not too many.
    assert ((desc["min"] < desc["lower_ci"]) & (desc["lower_ci"] < desc["mean"])).all()
    assert ((desc["mean"] < desc["upper_ci"]) & (desc["upper_ci"] < desc["max"])).all()
