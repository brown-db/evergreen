from evergreen.common.types import Interval
from evergreen.estimator import ConfSeqEstimator


def test_confseq_estimator():
    observations = [False, True, False, True, True, False, False, True, False, True]

    interval = ConfSeqEstimator.estimate_mean(
        observations,
        confidence_level=0.95,
        relative_error=0.05,
        population_size=None,
    )

    assert interval in Interval(0, 1)


def test_confseq_estimator_with_population_size():
    observations = [False, True, False, True, True, False, False, True, False, True]

    interval = ConfSeqEstimator.estimate_mean(
        observations,
        confidence_level=0.95,
        relative_error=0.05,
        population_size=len(observations),
    )

    assert interval.lower == interval.upper
    assert interval.lower == 5 / len(observations)
