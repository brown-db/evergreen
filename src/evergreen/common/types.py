from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import ceil, floor
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsComparison(Protocol):
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...


@dataclass(frozen=True, eq=False)
class Interval:
    # Lower bound of the interval
    lower: float
    # Upper bound of the interval
    upper: float
    # [0, 1]-valued real representing (1 - significance level)
    confidence_level: float = 1.0
    # [0, 1]-valued real representing the relative error
    relative_error: float = 0.0

    def __post_init__(self) -> None:
        assert self.lower <= self.upper

    def __lt__(self, other: Interval | float) -> bool:
        if isinstance(other, Interval):
            return self.upper < other.lower
        return self.upper < other

    def __le__(self, other: Interval | float) -> bool:
        if isinstance(other, Interval):
            return self.upper <= other.lower
        return self.upper <= other

    def __gt__(self, other: Interval | float) -> bool:
        if isinstance(other, Interval):
            return self.lower > other.upper
        return self.lower > other

    def __ge__(self, other: Interval | float) -> bool:
        if isinstance(other, Interval):
            return self.lower >= other.upper
        return self.lower >= other

    def __eq__(self, other: Interval | float) -> bool:  # type: ignore
        if isinstance(other, Interval):
            relative_error = max(self.relative_error, other.relative_error)
            tolerance = max(abs(other.lower), abs(other.upper)) * relative_error
            return (
                abs(self.lower - other.lower) <= tolerance
                and abs(self.upper - other.upper) <= tolerance
            )
        tolerance = abs(other) * self.relative_error
        return self in Interval(other - tolerance, other + tolerance)

    def __ne__(self, other: Interval | float) -> bool:  # type: ignore
        if isinstance(other, Interval):
            return not self.intersects(other)
        tolerance = abs(other) * self.relative_error
        return not self.intersects(Interval(other - tolerance, other + tolerance))

    def __contains__(self, other: Interval) -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def intersects(self, other: Interval) -> bool:
        return self.lower <= other.upper and other.lower <= self.upper

    def __add__(self, scalar: float) -> Interval:
        return Interval(
            self.lower + scalar,
            self.upper + scalar,
            self.confidence_level,
            self.relative_error,
        )

    def __sub__(self, scalar: float) -> Interval:
        return Interval(
            self.lower - scalar,
            self.upper - scalar,
            self.confidence_level,
            self.relative_error,
        )

    def __mul__(self, scalar: float) -> Interval:
        assert scalar >= 0.0
        return Interval(
            self.lower * scalar,
            self.upper * scalar,
            self.confidence_level,
            self.relative_error,
        )

    def __round__(self, ndigits: int | None = None) -> Interval:
        return Interval(
            round(self.lower, ndigits),
            round(self.upper, ndigits),
            self.confidence_level,
            self.relative_error,
        )

    def __floor__(self) -> Interval:
        return Interval(
            floor(self.lower),
            floor(self.upper),
            self.confidence_level,
            self.relative_error,
        )

    def __ceil__(self) -> Interval:
        return Interval(
            ceil(self.lower),
            ceil(self.upper),
            self.confidence_level,
            self.relative_error,
        )

    def as_count_interval(self) -> CountInterval:
        return CountInterval(
            self.lower,
            self.upper,
            self.confidence_level,
            self.relative_error,
        )

    def as_proportion_interval(self) -> ProportionInterval:
        return ProportionInterval(
            self.lower, self.upper, self.confidence_level, self.relative_error
        )


class CountInterval(Interval):
    def __add__(self, scalar: float) -> CountInterval:
        interval = super().__add__(scalar)
        return interval.as_count_interval()

    def __sub__(self, scalar: float) -> CountInterval:
        interval = super().__sub__(scalar)
        return interval.as_count_interval()

    def __mul__(self, scalar: float) -> CountInterval:
        interval = super().__mul__(scalar)
        return interval.as_count_interval()

    def __round__(self, ndigits: int | None = None) -> CountInterval:
        interval = super().__round__(ndigits)
        return interval.as_count_interval()

    def __floor__(self) -> CountInterval:
        interval = super().__floor__()
        return interval.as_count_interval()

    def __ceil__(self) -> CountInterval:
        interval = super().__ceil__()
        return interval.as_count_interval()


class ProportionInterval(Interval):
    def __add__(self, scalar: float) -> ProportionInterval:
        interval = super().__add__(scalar)
        return interval.as_proportion_interval()

    def __sub__(self, scalar: float) -> ProportionInterval:
        interval = super().__sub__(scalar)
        return interval.as_proportion_interval()

    def __mul__(self, scalar: float) -> ProportionInterval:
        interval = super().__mul__(scalar)
        return interval.as_proportion_interval()

    def __round__(self, ndigits: int | None = None) -> ProportionInterval:
        interval = super().__round__(ndigits)
        return interval.as_proportion_interval()

    def __floor__(self) -> ProportionInterval:
        interval = super().__floor__()
        return interval.as_proportion_interval()

    def __ceil__(self) -> ProportionInterval:
        interval = super().__ceil__()
        return interval.as_proportion_interval()


class FusedPlanType(Enum):
    FILTER = auto()
    PROJECTION = auto()
