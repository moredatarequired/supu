import pandas as pd

from supu.statistics import confidence_interval


def describe_series_with_ci(data: pd.Series, **kwargs) -> pd.DataFrame:
    """Add confidence intervals to a Series's describe() output.

    The output will have the same index, and the same columns as the output of
    data.describe(), with the addition of two columns for the lower and upper bounds of
    the 95% confidence interval.
    """
    levels = tuple(range(data.index.nlevels))
    groupby = data.groupby(level=levels)
    groups = dict(list(groupby))
    cis = pd.DataFrame(
        [confidence_interval(g, **kwargs) for g in groups.values()],
        index=list(groups.keys()),
        columns=["lower_ci", "upper_ci"],
    )
    desc = groupby.describe()
    return pd.concat([desc, cis], axis=1).set_index(desc.index)


def describe_dataframe_with_ci(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add confidence intervals to a DataFrame's describe() output.

    The output will have the same columns and index as the output of data.describe(),
    with the addition of two rows for the lower and upper bounds of the 95% confidence
    interval.
    """
    stacked = data.stack()
    stacked = stacked if isinstance(stacked, pd.Series) else stacked[0]
    return describe_series_with_ci(stacked, **kwargs)
