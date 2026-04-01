from __future__ import annotations

import reprlib
from dataclasses import dataclass, field
from typing import cast

from evergreen.catalog.schema import Schema
from evergreen.provenance import Monomial, Prov
from evergreen.storage.row_id import RowId


class Row:
    INTERNAL_ID_FIELD_NAME = "_id"
    AGG_ROW_ID = "_agg"

    def __init__(self, values: tuple[object, ...]) -> None:
        if all(isinstance(value, AnnotatedValue) for value in values):
            annotated_values = cast(tuple[AnnotatedValue, ...], values)
            self._values = tuple(av.value for av in annotated_values)
            self._prov = {
                i: av.prov
                for i, av in enumerate(annotated_values)
                if av.prov is not None
            }
        else:
            self._values = values
            self._prov: dict[int, Prov | ProvCollection] = {}

    def __getitem__(self, index: int) -> object:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Row):
            return False
        return self._values == other._values

    def __str__(self) -> str:
        return f"({', '.join(reprlib.repr(value) for value in self._values)})"

    def __repr__(self) -> str:
        return f"Row(values={self._values}, prov={self._prov})"

    def id(self, schema: Schema) -> RowId:
        return tuple(self._values[i] for i in schema.key_indices())

    def get_prov(self, index: int) -> Prov | ProvCollection | None:
        return self._prov.get(index)

    def get_prov_monomials(self, index: int) -> tuple[Monomial, ...]:
        prov = self.get_prov(index)
        if not isinstance(prov, Prov):
            raise ValueError(f"Provenance at index {index} is not a Prov")
        return prov.monomials()

    def with_values(self, annotated_values: tuple[AnnotatedValue, ...]) -> Row:
        existing = tuple(
            AnnotatedValue(self._values[i], self._prov.get(i))
            for i in range(len(self._values))
        )
        return Row(existing + annotated_values)


@dataclass(frozen=True)
class AnnotatedValue:
    value: object
    prov: Prov | ProvCollection | None


@dataclass
class ProvCollection:
    pos_polynomials: list[Prov] = field(default_factory=list[Prov])
    neg_polynomials: list[Prov] = field(default_factory=list[Prov])
    precomputed_total: int | None = None

    def total(self) -> int:
        if self.precomputed_total is not None:
            return self.precomputed_total
        return len(self.pos_polynomials) + len(self.neg_polynomials)
