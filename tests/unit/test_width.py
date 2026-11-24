"""FE-01 tests of the NumPy effective-width profiles b(X)."""

import numpy as np
import pytest

from running_contact_validation.fe.width import (
    ConstantWidth,
    LinearWidth,
    check_width_values,
)


def test_constant_width_values():
    b = ConstantWidth(0.09)
    X = np.linspace(-1.0, 1.0, 7)
    np.testing.assert_array_equal(b(X), np.full_like(X, 0.09))
    assert b(np.array(0.3)).shape == ()


def test_constant_width_range_enforced():
    b = ConstantWidth(0.09, X_range=(0.0, 0.3))
    with pytest.raises(ValueError, match="outside"):
        b(np.array([0.31]))


def test_linear_width_reproduces_stations_and_midpoint():
    b = LinearWidth(X0=0.0, b0=0.08, X1=0.28, b1=0.11)
    np.testing.assert_allclose(
        b(np.array([0.0, 0.14, 0.28])), [0.08, 0.095, 0.11], rtol=1e-15
    )
    assert b.slope == pytest.approx(0.03 / 0.28)


def test_linear_width_rejects_extrapolation():
    b = LinearWidth(0.0, 0.08, 0.28, 0.11)
    with pytest.raises(ValueError, match="extrapolation"):
        b(np.array([-0.01, 0.1]))


@pytest.mark.parametrize("bad", [0.0, -0.01, np.nan, np.inf])
def test_nonpositive_or_nonfinite_width_rejected(bad):
    with pytest.raises(ValueError):
        ConstantWidth(bad)
    with pytest.raises(ValueError):
        LinearWidth(0.0, 0.08, 0.28, bad)


def test_invalid_linear_stations_rejected():
    with pytest.raises(ValueError):
        LinearWidth(0.28, 0.08, 0.0, 0.11)


def test_check_width_values():
    check_width_values([0.1, 0.2])
    with pytest.raises(ValueError):
        check_width_values([0.1, -0.2])
