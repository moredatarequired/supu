from itertools import product

import numpy as np
import pandas as pd
import pytest

from supu import dataframes


def output_function(row):
    x, y, z = row
    tot, logx = x + y + z, np.log(x + 1)
    return logx * (z + y / 5), y * (np.sqrt(x) + z), tot / logx / (y + 1)


@pytest.fixture()
def multiindex_results():
    # Three independent variables, three dependent variables.
    inputs = pd.DataFrame(product(list(10 ** np.arange(9)), [0, 2, 15, 80], [0, 1, 2]))
    outputs = inputs.apply(output_function, axis=1, result_type="expand")
    joined = pd.concat([inputs, outputs], axis=1)
    joined.columns = ["x", "y", "z", "a", "b", "c"]
    base_df = joined.loc[joined.index.repeat(100)].set_index(["x", "y", "z"])
    return base_df + np.random.default_rng(0).normal(size=base_df.shape)


@pytest.mark.parametrize("column", ["a", "b", "c"])
@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_describe_series_with_ci_multiindex(column, confidence, multiindex_results):
    series = multiindex_results[column]
    desc = dataframes.describe_series_with_ci(
        series, confidence_level=confidence, n_resamples=100
    )
    assert set(desc.index) == set(series.index)
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
    vanilla_descrbe = series.groupby(level=(0, 1, 2)).describe()
    assert (vanilla_descrbe == desc[list(vanilla_descrbe.columns)]).all().all()
    # We should have some values outside the confidence interval, but not too many.
    assert ((desc["min"] < desc["lower_ci"]) & (desc["lower_ci"] < desc["mean"])).all()
    assert ((desc["mean"] < desc["upper_ci"]) & (desc["upper_ci"] < desc["max"])).all()


def test_describe_dataframe_with_ci_multiindex(multiindex_results):
    index, cols = multiindex_results.index, multiindex_results.columns
    chunks = []
    for col in cols:
        df = pd.DataFrame(sorted(set(index)), columns=index.names)
        df["variable"] = col
        df["dummy"] = True
        chunks.append(df)
    new_index = pd.concat(chunks).set_index([*list(index.names), "variable"]).index
    desc = dataframes.describe_dataframe_with_ci(multiindex_results, n_resamples=100)
    assert (desc.index == new_index.sort_values()).all()
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


def test_describe_dataframe_with_ci_no_names(multiindex_results):
    noindex = [None] * len(multiindex_results.index.names)
    multiindex_results.index.names = noindex
    desc = dataframes.describe_dataframe_with_ci(multiindex_results, n_resamples=100)
    assert desc.index.names == [*noindex, None]
