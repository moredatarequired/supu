import numpy as np

from supu import statistics


def test_confidence_interval():
    data = list(range(11))
    ci = statistics.confidence_interval(data)
    assert 3 < ci[0] < 4
    assert 6 < ci[1] < 7

    # With more samples, the confidence interval should be narrower.
    data = list(range(11)) * 10
    ci = statistics.confidence_interval(data)
    assert 4 < ci.low < 5
    assert 5 < ci.high < 6


def test_confidence_interval_random():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 1000))
    ci = statistics.confidence_interval(data, n_resamples=100)
    correct = ((ci.low < 0) & (ci.high > 0)).sum()
    assert 90 < correct < 100
