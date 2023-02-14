from typing import Any, NamedTuple, Sequence, Union

import numpy as np
from scipy.stats import bootstrap


def confidence_interval(data: Union[Sequence, Any], **kwargs) -> NamedTuple:
    """Return the confidence interval for the mean of the data."""
    if len(data) != 1:
        # The data should be a sequence of samples, not a single sample.
        data = [data]
    return bootstrap(
        data, np.mean, vectorized=True, axis=-1, **kwargs
    ).confidence_interval
