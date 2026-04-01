import logging
from abc import ABC, abstractmethod

from evergreen.common.types import Interval
from evergreen.estimator import ConfSeqEstimator
from evergreen.planner.logical.types import EarlyStopComparison, Operator
from evergreen.provenance import Prov
from evergreen.storage.row import AnnotatedValue, ProvCollection

logger = logging.getLogger(__name__)


class Accumulator(ABC):
    @abstractmethod
    def update(self, annotated_values: tuple[AnnotatedValue, ...]) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> AnnotatedValue:
        pass

    @abstractmethod
    def can_stop(self, comparison: EarlyStopComparison | None) -> bool:
        pass

    @staticmethod
    def _can_deterministically_stop(
        op: Operator, value: float, max_value: float, threshold: float
    ) -> bool:
        # If the disjunct involving `max_value` and `threshold` is true, this
        # means that the original predicate is false; thus, the resulting
        # provenance polynomial will be a truncated product containing only the
        # observed positive and negative polynomials rather than the full
        # product of all positive and negative polynomials.
        match op:
            case Operator.GT:
                return value > threshold or max_value <= threshold
            case Operator.GE:
                return value >= threshold or max_value < threshold
            case Operator.LT:
                return max_value < threshold or value >= threshold
            case Operator.LE:
                return max_value <= threshold or value > threshold
            case Operator.EQ | Operator.NE:
                return value > threshold or max_value < threshold
            case _:
                raise ValueError(f"Unexpected operator: {op}")

    @staticmethod
    def _can_probabilistically_stop(
        op: Operator,
        confidence_interval: Interval,
        threshold: float,
    ) -> bool:
        match op:
            case Operator.GT:
                return (
                    confidence_interval > threshold or confidence_interval <= threshold
                )
            case Operator.GE:
                return (
                    confidence_interval >= threshold or confidence_interval < threshold
                )
            case Operator.LT:
                return (
                    confidence_interval < threshold or confidence_interval >= threshold
                )
            case Operator.LE:
                return (
                    confidence_interval <= threshold or confidence_interval > threshold
                )
            case Operator.EQ | Operator.NE:
                return (
                    confidence_interval == threshold or confidence_interval != threshold
                )
            case _:
                raise ValueError(f"Unexpected operator: {op}")


class CountIfAccumulator(Accumulator):
    def __init__(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
    ) -> None:
        self._precomputed_total = precomputed_total
        self._confidence_level = confidence_level
        self._relative_error = relative_error

        logger.debug(
            "%s init: precomputed_total=%s, confidence_level=%s, relative_error=%s",
            self.__class__.__name__,
            precomputed_total,
            confidence_level,
            relative_error,
        )

        self._count = 0
        self._observations: list[bool] = []
        self._confidence_interval: Interval | None = None
        self._prov_collection = ProvCollection()

    def update(self, annotated_values: tuple[AnnotatedValue, ...]) -> None:
        (annotated_value,) = annotated_values
        value = annotated_value.value
        prov = annotated_value.prov

        if not isinstance(value, bool):
            raise ValueError(f"Unexpected type: {type(value)}")

        assert isinstance(prov, Prov)

        if value:
            self._count += 1
            self._prov_collection.pos_polynomials.append(prov)
        else:
            self._prov_collection.neg_polynomials.append(prov)

        self._observations.append(value)

        # Invalidate cached confidence interval when new observations are added
        self._confidence_interval = None

        logger.debug(
            "%s update: count=%d, observation_count=%d",
            self.__class__.__name__,
            self._count,
            len(self._observations),
        )

    def evaluate(self) -> AnnotatedValue:
        if self._confidence_interval is not None:
            annotated_value = AnnotatedValue(
                self._confidence_interval.as_count_interval(), self._prov_collection
            )
        else:
            annotated_value = AnnotatedValue(self._count, self._prov_collection)

        logger.debug(
            "%s evaluate: value=%s", self.__class__.__name__, annotated_value.value
        )

        return annotated_value

    def can_stop(self, comparison: EarlyStopComparison | None) -> bool:
        if comparison is None:
            return False

        if not isinstance(comparison.threshold, int):
            raise ValueError(f"Unexpected type: {type(comparison.threshold)}")

        if self._precomputed_total is not None:
            # All observations processed; this avoids pulling the row for the
            # next group unnecessarily
            if self._precomputed_total == len(self._observations):
                logger.debug(
                    "%s deterministic stop: "
                    "count=%d, observation_count=%d, precomputed_total=%d",
                    self.__class__.__name__,
                    self._count,
                    len(self._observations),
                    self._precomputed_total,
                )
                return True

            remaining = self._precomputed_total - len(self._observations)
            max_count = self._count + remaining
            if self._can_deterministically_stop(
                comparison.op, self._count, max_count, comparison.threshold
            ):
                logger.debug(
                    "%s deterministic early stop: "
                    "count=%d, observation_count=%d, precomputed_total=%d",
                    self.__class__.__name__,
                    self._count,
                    len(self._observations),
                    self._precomputed_total,
                )
                return True

            if self._confidence_level is not None:
                confidence_interval = (
                    ConfSeqEstimator.estimate_mean(
                        self._observations,
                        self._confidence_level,
                        self._relative_error,
                        self._precomputed_total,
                    )
                    * self._precomputed_total
                )
                if self._can_probabilistically_stop(
                    comparison.op,
                    confidence_interval,
                    comparison.threshold,
                ):
                    self._confidence_interval = confidence_interval
                    logger.debug(
                        "%s probabilistic early stop: "
                        "count=%d, observation_count=%d, precomputed_total=%d, "
                        "confidence_interval=%s",
                        self.__class__.__name__,
                        self._count,
                        len(self._observations),
                        self._precomputed_total,
                        confidence_interval,
                    )
                    return True

        match comparison.op:
            case Operator.GT | Operator.LE | Operator.EQ | Operator.NE:
                can_stop = self._count > comparison.threshold
            case Operator.GE | Operator.LT:
                can_stop = self._count >= comparison.threshold
            case _:
                raise ValueError(f"Unexpected operator: {comparison.op}")

        if can_stop:
            logger.debug(
                "%s deterministic early stop: count=%d, observation_count=%d",
                self.__class__.__name__,
                self._count,
                len(self._observations),
            )

        return can_stop


class ProportionAccumulator(Accumulator):
    def __init__(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
    ) -> None:
        self._precomputed_total: int | None = precomputed_total
        self._confidence_level = confidence_level
        self._relative_error = relative_error

        logger.debug(
            "%s init: precomputed_total=%s, confidence_level=%s, relative_error=%s",
            self.__class__.__name__,
            precomputed_total,
            confidence_level,
            relative_error,
        )

        self._count = 0
        self._observations: list[bool] = []
        self._confidence_interval: Interval | None = None
        self._prov_collection = ProvCollection()

    def update(self, annotated_values: tuple[AnnotatedValue, ...]) -> None:
        (annotated_value,) = annotated_values
        value = annotated_value.value
        prov = annotated_value.prov

        if not isinstance(value, bool):
            raise ValueError(f"Unexpected type: {type(value)}")

        assert isinstance(prov, Prov)

        if value:
            self._count += 1
            self._prov_collection.pos_polynomials.append(prov)
        else:
            self._prov_collection.neg_polynomials.append(prov)

        self._observations.append(value)

        # Invalidate cached confidence interval when new observations are added
        self._confidence_interval = None

        logger.debug(
            "%s update: count=%d, observation_count=%d",
            self.__class__.__name__,
            self._count,
            len(self._observations),
        )

    def evaluate(self) -> AnnotatedValue:
        if self._precomputed_total is not None:
            self._prov_collection.precomputed_total = self._precomputed_total

        if self._confidence_interval is not None:
            annotated_value = AnnotatedValue(
                self._confidence_interval.as_proportion_interval(),
                self._prov_collection,
            )
        else:
            total = self._precomputed_total or len(self._observations)
            proportion = self._count / total if total else 0.0
            annotated_value = AnnotatedValue(proportion, self._prov_collection)

        logger.debug(
            "%s evaluate: value=%s", self.__class__.__name__, annotated_value.value
        )

        return annotated_value

    def can_stop(self, comparison: EarlyStopComparison | None) -> bool:
        if comparison is None:
            return False

        if not isinstance(comparison.threshold, float):
            raise ValueError(f"Unexpected type: {type(comparison.threshold)}")

        if self._precomputed_total is not None:
            # All observations processed; this avoids pulling the row for the
            # next group unnecessarily
            if self._precomputed_total == len(self._observations):
                logger.debug(
                    "%s deterministic stop: "
                    "count=%d, observation_count=%d, precomputed_total=%d",
                    self.__class__.__name__,
                    self._count,
                    len(self._observations),
                    self._precomputed_total,
                )
                return True

            proportion = self._count / self._precomputed_total
            remaining = self._precomputed_total - len(self._observations)
            max_proportion = (self._count + remaining) / self._precomputed_total
            if self._can_deterministically_stop(
                comparison.op,
                proportion,
                max_proportion,
                comparison.threshold,
            ):
                logger.debug(
                    "%s deterministic early stop: "
                    "count=%d, observation_count=%d, precomputed_total=%d",
                    self.__class__.__name__,
                    self._count,
                    len(self._observations),
                    self._precomputed_total,
                )
                return True

        if self._confidence_level is not None:
            confidence_interval = ConfSeqEstimator.estimate_mean(
                self._observations,
                self._confidence_level,
                self._relative_error,
                self._precomputed_total,
            )
            if self._can_probabilistically_stop(
                comparison.op,
                confidence_interval,
                comparison.threshold,
            ):
                self._confidence_interval = confidence_interval
                logger.debug(
                    "%s probabilistic early stop: "
                    "count=%d, observation_count=%d, precomputed_total=%s, "
                    "confidence_interval=%s",
                    self.__class__.__name__,
                    self._count,
                    len(self._observations),
                    self._precomputed_total,
                    confidence_interval,
                )
                return True

        return False


class BoolAndAccumulator(Accumulator):
    def __init__(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
        minimal_provenance: bool,
    ) -> None:
        self._precomputed_total = precomputed_total
        self._confidence_level = confidence_level
        self._relative_error = relative_error
        self._minimal_provenance = minimal_provenance

        logger.debug(
            "%s init: precomputed_total=%s, confidence_level=%s, relative_error=%s, "
            "minimal_provenance=%s",
            self.__class__.__name__,
            precomputed_total,
            confidence_level,
            relative_error,
            minimal_provenance,
        )

        self._value = True
        self._observations: list[bool] = []
        self._confidence_interval: Interval | None = None
        self._prov_collection = ProvCollection()

    def update(self, annotated_values: tuple[AnnotatedValue, ...]) -> None:
        (annotated_value,) = annotated_values
        value = annotated_value.value
        prov = annotated_value.prov

        if not isinstance(value, bool):
            raise ValueError(f"Unexpected type: {type(value)}")

        assert isinstance(prov, Prov)

        self._value &= value
        if value:
            self._prov_collection.pos_polynomials.append(prov)
        else:
            self._prov_collection.neg_polynomials.append(prov)

        self._observations.append(value)

        # Invalidate cached confidence interval when new observations are added
        self._confidence_interval = None

        logger.debug(
            "%s update: observation_count=%d",
            self.__class__.__name__,
            len(self._observations),
        )

    def evaluate(self) -> AnnotatedValue:
        polynomial: Prov

        if self._value:
            polynomial = Prov.one()
            for prov in self._prov_collection.pos_polynomials:
                polynomial *= prov
        else:
            if self._minimal_provenance:
                polynomial = self._prov_collection.neg_polynomials[0]
            else:
                polynomial = Prov.zero()
                for prov in self._prov_collection.neg_polynomials:
                    polynomial += prov

        if self._confidence_interval is not None:
            annotated_value = AnnotatedValue(
                self._confidence_interval == 1.0, polynomial
            )
        else:
            annotated_value = AnnotatedValue(self._value, polynomial)

        logger.debug(
            "%s evaluate: value=%s", self.__class__.__name__, annotated_value.value
        )

        return annotated_value

    def can_stop(self, comparison: EarlyStopComparison | None) -> bool:
        if not self._value:
            logger.debug(
                "%s deterministic early stop: value=False", self.__class__.__name__
            )
            return True

        # All observations processed; this avoids pulling the row for the
        # next group unnecessarily
        if self._precomputed_total is not None and self._precomputed_total == len(
            self._observations
        ):
            logger.debug(
                "%s deterministic stop: observation_count=%d, precomputed_total=%d",
                self.__class__.__name__,
                len(self._observations),
                self._precomputed_total,
            )
            return True

        if self._confidence_level is not None:
            confidence_interval = ConfSeqEstimator.estimate_mean(
                self._observations,
                self._confidence_level,
                self._relative_error,
                self._precomputed_total,
            )
            if self._can_probabilistically_stop(
                Operator.EQ,
                confidence_interval,
                threshold=1.0,
            ):
                self._confidence_interval = confidence_interval
                logger.debug(
                    "%s probabilistic early stop: "
                    "observation_count=%d, precomputed_total=%s, "
                    "confidence_interval=%s",
                    self.__class__.__name__,
                    len(self._observations),
                    self._precomputed_total,
                    confidence_interval,
                )
                return True

        return False


class BoolOrAccumulator(Accumulator):
    def __init__(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
        minimal_provenance: bool,
    ) -> None:
        self._precomputed_total = precomputed_total
        self._confidence_level = confidence_level
        self._relative_error = relative_error
        self._minimal_provenance = minimal_provenance

        logger.debug(
            "%s init: precomputed_total=%s, confidence_level=%s, relative_error=%s, "
            "minimal_provenance=%s",
            self.__class__.__name__,
            precomputed_total,
            confidence_level,
            relative_error,
            minimal_provenance,
        )

        self._value = False
        self._prov_collection = ProvCollection()
        self._observations: list[bool] = []
        self._confidence_interval: Interval | None = None

    def update(self, annotated_values: tuple[AnnotatedValue, ...]) -> None:
        (annotated_value,) = annotated_values
        value = annotated_value.value
        prov = annotated_value.prov

        if not isinstance(value, bool):
            raise ValueError(f"Unexpected type: {type(value)}")

        assert isinstance(prov, Prov)

        self._value |= value
        if value:
            self._prov_collection.pos_polynomials.append(prov)
        else:
            self._prov_collection.neg_polynomials.append(prov)

        self._observations.append(value)

        # Invalidate cached confidence interval when new observations are added
        self._confidence_interval = None

        logger.debug(
            "%s update: observation_count=%d",
            self.__class__.__name__,
            len(self._observations),
        )

    def evaluate(self) -> AnnotatedValue:
        polynomial: Prov

        if self._value:
            if self._minimal_provenance:
                polynomial = self._prov_collection.pos_polynomials[0]
            else:
                polynomial = Prov.zero()
                for prov in self._prov_collection.pos_polynomials:
                    polynomial += prov
        else:
            polynomial = Prov.one()
            for prov in self._prov_collection.neg_polynomials:
                polynomial *= prov

        if self._confidence_interval is not None:
            annotated_value = AnnotatedValue(self._confidence_interval >= 1, polynomial)
        else:
            annotated_value = AnnotatedValue(self._value, polynomial)

        logger.debug(
            "%s evaluate: value=%s", self.__class__.__name__, annotated_value.value
        )

        return annotated_value

    def can_stop(self, comparison: EarlyStopComparison | None) -> bool:
        if self._value:
            logger.debug(
                "%s deterministic early stop: value=True", self.__class__.__name__
            )
            return True

        # All observations processed; this avoids pulling the row for the
        # next group unnecessarily
        if self._precomputed_total is not None and self._precomputed_total == len(
            self._observations
        ):
            logger.debug(
                "%s deterministic stop: observation_count=%d, precomputed_total=%d",
                self.__class__.__name__,
                len(self._observations),
                self._precomputed_total,
            )
            return True

        if self._confidence_level is not None and self._precomputed_total is not None:
            confidence_interval = (
                ConfSeqEstimator.estimate_mean(
                    self._observations,
                    self._confidence_level,
                    self._relative_error,
                    self._precomputed_total,
                )
                * self._precomputed_total
            )
            if self._can_probabilistically_stop(
                Operator.GE, confidence_interval, threshold=1
            ):
                self._confidence_interval = confidence_interval
                logger.debug(
                    "%s probabilistic early stop: "
                    "observation_count=%d, precomputed_total=%d, "
                    "confidence_interval=%s",
                    self.__class__.__name__,
                    len(self._observations),
                    self._precomputed_total,
                    confidence_interval,
                )
                return True

        return False
