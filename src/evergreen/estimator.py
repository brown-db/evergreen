import logging

from confseq.betting import betting_cs  # type: ignore
from evergreen.common.types import Interval

logger = logging.getLogger(__name__)


class ConfSeqEstimator:
    _BREAKS = 100

    @classmethod
    def estimate_mean(
        cls,
        observations: list[bool],
        confidence_level: float,
        relative_error: float,
        population_size: int | None,
    ) -> Interval:
        significance_level = 1 - confidence_level
        lower_cs, upper_cs = betting_cs(  # type: ignore
            observations,
            alpha=significance_level,
            N=population_size,
            breaks=cls._BREAKS,
        )
        confidence_interval = Interval(
            float(lower_cs[-1]), float(upper_cs[-1]), confidence_level, relative_error
        )

        logger.debug(
            "Mean estimate: %s, observation_count=%d, population_size=%s",
            confidence_interval,
            len(observations),
            population_size,
        )

        return confidence_interval
