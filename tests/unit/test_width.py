"""FE-01 tests of the NumPy effective-width profiles b(X)."""

import numpy as np
import pytest

from running_contact_validation.fe.width import (
    ConstantWidth,
    LinearWidth,
    PiecewiseLinearWidth,
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


# --- PiecewiseLinearWidth (measured-station interpolation) -------------------

STATIONS_X = [0.0, 0.05, 0.14, 0.22, 0.28]
STATIONS_B = [0.075, 0.085, 0.070, 0.105, 0.095]


def write_csv(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_piecewise_linear_reproduces_stations_and_interpolates():
    b = PiecewiseLinearWidth(STATIONS_X, STATIONS_B)
    np.testing.assert_allclose(b(np.array(STATIONS_X)), STATIONS_B, rtol=1e-15)
    mid = 0.5 * (np.array(STATIONS_X[:-1]) + np.array(STATIONS_X[1:]))
    expected = 0.5 * (np.array(STATIONS_B[:-1]) + np.array(STATIONS_B[1:]))
    np.testing.assert_allclose(b(mid), expected, rtol=1e-14)
    assert b.X_range == (0.0, 0.28)


def test_piecewise_linear_two_stations_matches_linear_width():
    X = np.linspace(0.0, 0.28, 11)
    pw = PiecewiseLinearWidth([0.0, 0.28], [0.08, 0.11])
    np.testing.assert_allclose(pw(X), LinearWidth(0.0, 0.08, 0.28, 0.11)(X), rtol=1e-14)


def test_piecewise_linear_rejects_extrapolation():
    b = PiecewiseLinearWidth(STATIONS_X, STATIONS_B)
    with pytest.raises(ValueError, match="extrapolation"):
        b(np.array([0.281]))


@pytest.mark.parametrize(
    "X, b",
    [
        ([0.0], [0.1]),  # too few stations
        ([0.0, 0.1], [0.1]),  # length mismatch
        ([0.0, 0.1, 0.1], [0.1, 0.1, 0.1]),  # repeated station
        ([0.1, 0.0], [0.1, 0.1]),  # decreasing
        ([0.0, 0.1], [0.1, 0.0]),  # non-positive width
        ([0.0, np.nan], [0.1, 0.1]),  # non-finite position
    ],
)
def test_piecewise_linear_invalid_stations(X, b):
    with pytest.raises(ValueError):
        PiecewiseLinearWidth(X, b)


def test_from_csv_round_trip(tmp_path):
    lines = ["# midsole width, metres", "X_m, b_m, note_id"]
    lines += [f"{x}, {w}, {i}" for i, (x, w) in enumerate(zip(STATIONS_X, STATIONS_B))]
    path = write_csv(tmp_path / "width.csv", "\n".join(lines) + "\n")
    b = PiecewiseLinearWidth.from_csv(path)
    np.testing.assert_array_equal(b.X_stations, STATIONS_X)
    np.testing.assert_array_equal(b.b_stations, STATIONS_B)


def test_from_csv_custom_columns(tmp_path):
    path = write_csv(tmp_path / "w.csv", "pos,width\n0.0,0.08\n0.28,0.11\n")
    b = PiecewiseLinearWidth.from_csv(path, X_column="pos", b_column="width")
    assert b(np.array(0.14)) == pytest.approx(0.095)


@pytest.mark.parametrize(
    "text, match",
    [
        ("X,b\n0.0,0.08\n0.28,0.11\n", "lacks column"),
        ("X_m,b_m\n0.0,0.08\n0.28,\n", "missing or non-numeric"),
        ("X_m,b_m\n0.0,0.08\n0.28,abc\n", "missing or non-numeric"),
        ("X_m,b_m\n0.28,0.08\n0.0,0.11\n", "strictly increasing"),
        ("X_m,b_m\n0.0,0.08\n", "at least two"),
        ("X_m,b_m\n0.0,0.08\n0.28,-0.01\n", "> 0"),
    ],
)
def test_from_csv_rejects_bad_files(tmp_path, text, match):
    path = write_csv(tmp_path / "bad.csv", text)
    with pytest.raises(ValueError, match=match):
        PiecewiseLinearWidth.from_csv(path)
