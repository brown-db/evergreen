from abc import ABC, abstractmethod

from evergreen.planner.physical.accumulator import (
    Accumulator,
    BoolAndAccumulator,
    BoolOrAccumulator,
    CountIfAccumulator,
    ProportionAccumulator,
)


class AggregateUDF(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def return_type(self, arg_types: tuple[type, ...]) -> type:
        pass

    @abstractmethod
    def supports_estimation(self, has_shuffle: bool) -> bool:
        pass

    @abstractmethod
    def create_accumulator(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
        minimal_provenance: bool,
    ) -> Accumulator:
        pass


class CountIfUDF(AggregateUDF):
    def name(self) -> str:
        return "count_if"

    def return_type(self, arg_types: tuple[type, ...]) -> type:
        return int

    def supports_estimation(self, has_shuffle: bool) -> bool:
        return has_shuffle

    def create_accumulator(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
        minimal_provenance: bool,
    ) -> Accumulator:
        return CountIfAccumulator(precomputed_total, confidence_level, relative_error)


class ProportionUDF(AggregateUDF):
    def name(self) -> str:
        return "proportion"

    def return_type(self, arg_types: tuple[type, ...]) -> type:
        return float

    def supports_estimation(self, has_shuffle: bool) -> bool:
        return has_shuffle

    def create_accumulator(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
        minimal_provenance: bool,
    ) -> Accumulator:
        return ProportionAccumulator(
            precomputed_total, confidence_level, relative_error
        )


class BoolAndUDF(AggregateUDF):
    def name(self) -> str:
        return "bool_and"

    def return_type(self, arg_types: tuple[type, ...]) -> type:
        return bool

    def supports_estimation(self, has_shuffle: bool) -> bool:
        return has_shuffle

    def create_accumulator(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
        minimal_provenance: bool,
    ) -> Accumulator:
        return BoolAndAccumulator(
            precomputed_total, confidence_level, relative_error, minimal_provenance
        )


class BoolOrUDF(AggregateUDF):
    def name(self) -> str:
        return "bool_or"

    def return_type(self, arg_types: tuple[type, ...]) -> type:
        return bool

    def supports_estimation(self, has_shuffle: bool) -> bool:
        return has_shuffle

    def create_accumulator(
        self,
        precomputed_total: int | None,
        confidence_level: float | None,
        relative_error: float,
        minimal_provenance: bool,
    ) -> Accumulator:
        return BoolOrAccumulator(
            precomputed_total, confidence_level, relative_error, minimal_provenance
        )
