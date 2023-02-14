import numpy as np

from supu import statistics


def test_confidence_interval():
    data = list(range(11))
    low, high = statistics.confidence_interval(data)
    assert 3 < low < 4
    assert 6 < high < 7

    # With more samples, the confidence interval should be narrower.
    data = list(range(11)) * 10
    low, high = statistics.confidence_interval(data)
    assert 4 < low < 5
    assert 5 < high < 6


def test_confidence_interval_random():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2000))
    low, high = statistics.confidence_interval(data, n_resamples=100)
    correct = ((low < 0) & (high > 0)).sum()
    assert 90 < correct < 100
