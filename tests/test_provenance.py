from evergreen.planner.logical.expr import col
from evergreen.provenance import Prov


def test_equivalent():
    w = Prov.pos_token((), col("w"))
    x = Prov.pos_token((), col("x"))
    x_bar = Prov.neg_token((), col("x"))
    y = Prov.pos_token((), col("y"))
    z = Prov.pos_token((), col("z"))

    # Reflexive
    assert x == x
    assert x + y == x + y
    assert x * y == x * y

    # Additive identity
    assert x + Prov.zero() == x
    assert Prov.zero() + x == x

    # Multiplicative identity
    assert x * Prov.one() == x
    assert Prov.one() * x == x

    # Multiplicative annihilation
    assert x * Prov.zero() == Prov.zero()
    assert Prov.zero() * x == Prov.zero()

    # Associativity
    assert (x + y) + z == x + (y + z)
    assert (x * y) * z == x * (y * z)

    # Commutativity
    assert x + y == y + x
    assert x * y == y * x

    # Idempotency of addition
    assert x + x == x
    assert x + x + x == x

    # Idempotency of multiplication
    assert x * x == x
    assert x * x * x == x

    # Absorption
    assert x + (x * y) == x
    assert (x * y) + x == x

    # Distributivity
    assert x * (y + z) == (x * y) + (x * z)
    assert (x + y) * z == (x * z) + (y * z)
    assert (x + y) * (z + w) == (x * z) + (x * w) + (y * z) + (y * w)

    # NOT equivalent
    assert x != y
    assert x != x_bar
    assert x + y != x + z
    assert x * y != x * z


def test_large_product():
    tokens = [Prov.pos_token((i,), col(f"t{i}")) for i in range(10_000)]

    product = Prov.one()
    for token in tokens:
        product *= token

    monomials = product.monomials()

    assert len(monomials) == 1
    assert len(monomials[0]) == 10_000


def test_large_sum():
    tokens = [Prov.pos_token((i,), col(f"t{i}")) for i in range(10_000)]

    total = Prov.zero()
    for token in tokens:
        total += token

    monomials = total.monomials()

    assert len(monomials) == 10_000
    assert all(len(m) == 1 for m in monomials)


def test_large_sum_of_products():
    total = Prov.zero()
    for i in range(0, 20_000, 2):
        t1 = Prov.pos_token((i,), col(f"t{i}"))
        t2 = Prov.pos_token((i + 1,), col(f"t{i + 1}"))
        total += t1 * t2

    monomials = total.monomials()

    assert len(monomials) == 10_000
    assert all(len(m) == 2 for m in monomials)
