from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def confidence_interval(data: Union[Sequence, Any], **kwargs) -> pd.Series:
    """Return the confidence interval for the mean of the data."""
    if len(data) != 1:
        # The data should be a sequence of samples, not a single sample.
        data = [data]
    statistic = kwargs.pop("statistic", np.mean)
    return pd.Series(
        bootstrap(data, statistic, axis=-1, **kwargs).confidence_interval,
        index=["lower_ci", "upper_ci"],
        name=statistic.__name__,
    )
