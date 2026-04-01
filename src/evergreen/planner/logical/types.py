from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from evergreen.common.types import SupportsComparison


class Operator(Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "&"
    OR = "|"
    MAX_DOT = "max_dot"

    def return_type(self) -> type:
        match self:
            case (
                Operator.EQ
                | Operator.NE
                | Operator.LT
                | Operator.LE
                | Operator.GT
                | Operator.GE
                | Operator.AND
                | Operator.OR
            ):
                return bool
            case Operator.MAX_DOT:
                return float

    def flip(self) -> Operator:
        match self:
            case Operator.EQ | Operator.NE | Operator.AND | Operator.OR:
                return self
            case Operator.LT:
                return Operator.GT
            case Operator.LE:
                return Operator.GE
            case Operator.GT:
                return Operator.LT
            case Operator.GE:
                return Operator.LE
            case Operator.MAX_DOT:
                return self

    def negate(self) -> Operator:
        match self:
            case Operator.GE:
                return Operator.LT
            case Operator.GT:
                return Operator.LE
            case Operator.LE:
                return Operator.GT
            case Operator.LT:
                return Operator.GE
            case Operator.EQ:
                return Operator.NE
            case Operator.NE:
                return Operator.EQ
            case _:
                raise ValueError(f"Unexpected operator: {self}")

    def evaluate(self, left: object, right: object) -> object:
        # Equality works on any type
        if self == Operator.EQ:
            return left == right
        if self == Operator.NE:
            return left != right

        # Boolean operators
        if self in (Operator.AND, Operator.OR):
            if not isinstance(left, bool) or not isinstance(right, bool):
                raise ValueError(f"Unexpected types: {type(left)}, {type(right)}")
            return (left and right) if self == Operator.AND else (left or right)

        # Max dot product
        if self == Operator.MAX_DOT:
            sentence_embeddings = np.array(left)
            if sentence_embeddings.size == 0:
                return -1.0
            query_embedding = np.array(right)
            return float(np.max(sentence_embeddings @ query_embedding))

        # Comparison operators (LT, LE, GT, GE)
        if not isinstance(left, SupportsComparison) or not isinstance(
            right, SupportsComparison
        ):
            raise ValueError(f"Unexpected types: {type(left)}, {type(right)}")

        match self:
            case Operator.LT:
                return left < right
            case Operator.LE:
                return left <= right
            case Operator.GT:
                return left > right
            case Operator.GE:
                return left >= right


@dataclass(frozen=True)
class EarlyStopComparison:
    column_name: str
    op: Operator
    threshold: object
