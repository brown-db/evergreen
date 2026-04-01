from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evergreen.storage.row_id import RowId

if TYPE_CHECKING:
    from evergreen.planner.logical.expr import Expr


@dataclass(frozen=True)
class Zero:
    def __str__(self) -> str:
        return "0"


@dataclass(frozen=True)
class One:
    def __str__(self) -> str:
        return "1"


@dataclass(frozen=True)
class Token:
    row_id: RowId
    predicate: Expr
    sign: bool

    def __str__(self) -> str:
        return f"({self.row_id}, {self.predicate}, {self.sign})"


@dataclass(frozen=True)
class Sum:
    children: tuple[ProvExpr, ...]

    def __str__(self) -> str:
        return f"({' + '.join(str(child) for child in self.children)})"


@dataclass(frozen=True)
class Product:
    children: tuple[ProvExpr, ...]

    def __str__(self) -> str:
        return f"({' * '.join(str(child) for child in self.children)})"


type ProvExpr = Zero | One | Token | Sum | Product

type Monomial = frozenset[Token]
type NormalForm = frozenset[Monomial]


class Prov:
    def __init__(self, expr: ProvExpr):
        self._expr = expr
        self._nf_cache: NormalForm | None = None

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"Prov({self._expr})"

    @classmethod
    def zero(cls) -> Prov:
        return cls(Zero())

    @classmethod
    def one(cls) -> Prov:
        return cls(One())

    @classmethod
    def pos_token(cls, row_id: RowId, predicate: Expr) -> Prov:
        return cls(Token(row_id, predicate, sign=True))

    @classmethod
    def neg_token(cls, row_id: RowId, predicate: Expr) -> Prov:
        return cls(Token(row_id, predicate, sign=False))

    def __add__(self, other: Prov) -> Prov:
        # Flatten nested sums
        left = self._expr.children if isinstance(self._expr, Sum) else (self._expr,)
        right = other._expr.children if isinstance(other._expr, Sum) else (other._expr,)
        return Prov(Sum(left + right))

    def __mul__(self, other: Prov) -> Prov:
        # Flatten nested products
        left = self._expr.children if isinstance(self._expr, Product) else (self._expr,)
        right = (
            other._expr.children if isinstance(other._expr, Product) else (other._expr,)
        )
        return Prov(Product(left + right))

    def dnf(self) -> NormalForm:
        """Converts a `ProvExpr` to irredundant (i.e., absorbed) disjunctive normal form
        (DNF). This is the canonical form in the PosBool semiring.
        """
        if self._nf_cache is None:
            self._nf_cache = self._to_dnf(self._expr)
        return self._nf_cache

    def monomials(self) -> tuple[Monomial, ...]:
        return tuple(self.dnf())

    @classmethod
    def _to_dnf(cls, expr: ProvExpr) -> NormalForm:
        match expr:
            case Zero():
                return frozenset()
            case One():
                return frozenset([frozenset()])
            case Token():
                return frozenset([frozenset([expr])])
            case Sum(children):
                # Union of all children's DNFs
                sum_result: NormalForm = frozenset()
                for child in children:
                    sum_result |= cls._to_dnf(child)
                return cls._absorb(sum_result)
            case Product(children):
                # Cartesian product of all children's DNFs
                product_result: NormalForm = frozenset([frozenset()])
                for child in children:
                    child_dnf = cls._to_dnf(child)
                    product_result = frozenset(
                        monomial_result | monomial_child
                        for monomial_result in product_result
                        for monomial_child in child_dnf
                    )
                return cls._absorb(product_result)

    @staticmethod
    def _absorb(nf: NormalForm) -> NormalForm:
        """Applies absorption to a polynomial in normal form. A semiring is absorptive
        if p + p*q = p, or equivalently, if multiplication is decreasing, i.e., p*q
        <= p. When interpreting each monomial as a set, this means that no monomial
        is a subset of another. Thus, we remove all the (absorbed) monomials that
        are supersets of another.

        Reference: Grädel, E. and Tannen, V. (2024). "Provenance Analysis and
        Semiring Semantics for First-Order Logic." Section 8.3, p. 29.
        https://arxiv.org/abs/2412.07986
        """
        result = set(nf)
        for monomial_a in nf:
            for monomial_b in nf:
                if monomial_a != monomial_b and monomial_a <= monomial_b:
                    result.discard(monomial_b)
        return frozenset(result)

    def __eq__(self, other: Prov) -> bool:  # type: ignore
        return self.dnf() == other.dnf()
