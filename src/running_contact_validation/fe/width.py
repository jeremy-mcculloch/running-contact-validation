"""Effective out-of-plane width profiles ``b(X)`` for the 2D sagittal model.

``b(X) > 0`` [m] weights reference-volume integrals of the 2D continuum,
e.g. ``M = integral_Omega0 rho_0 b(X) dA`` (``docs/scientific_specification.md``
Section 16.2). It is a function of the **reference** heel-to-toe coordinate
``X`` only and must be evaluated at element quadrature points in reference
coordinates; see :mod:`running_contact_validation.fe.variable_width`.

This module is NumPy-only. Each profile's formula is written with plain
arithmetic so the same :meth:`WidthProfile.__call__` also accepts a UFL
expression for ``X`` (e.g. ``ufl.SpatialCoordinate(mesh)[0]`` on an
undeformed reference mesh).

Profiles: :class:`ConstantWidth` and :class:`LinearWidth` are analytic
verification benchmarks. :class:`PiecewiseLinearWidth` linearly interpolates
measured midsole-width stations (D21e), optionally read from CSV; linear
interpolation was requested by the researcher in PR #9 review. No
extrapolation beyond the measured stations is performed (Section 17.2).
"""

from __future__ import annotations

import csv
import math
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "ConstantWidth",
    "LinearWidth",
    "PiecewiseLinearWidth",
    "WidthProfile",
    "check_width_values",
]


def check_width_values(values: npt.ArrayLike) -> None:
    """Raise ``ValueError`` unless all widths are finite and strictly positive."""
    b = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(b)):
        raise ValueError("effective width b(X) must be finite")
    if b.size and not np.all(b > 0.0):
        raise ValueError(f"effective width b(X) must be > 0 m; min = {b.min():.6g} m")


class WidthProfile(ABC):
    """Positive effective-width function ``b(X)`` [m] of the reference coordinate ``X`` [m]."""

    @property
    @abstractmethod
    def X_range(self) -> tuple[float, float]:
        """Closed interval of reference ``X`` on which the profile is defined [m]."""

    @abstractmethod
    def _formula(self, X: Any) -> Any:
        """Width formula; must use only arithmetic valid for NumPy arrays and UFL."""

    def __call__(self, X: Any) -> Any:
        """Evaluate ``b`` at ``X``.

        For array-like ``X`` returns a float64 array and checks that every
        point lies in :attr:`X_range` (within a relative tolerance) and that
        the result is positive. For a UFL expression returns the
        corresponding UFL expression (no checks are possible symbolically).
        """
        if _is_ufl(X):
            return self._formula(X)
        Xa = np.asarray(X, dtype=np.float64)
        lo, hi = self.X_range
        tol = 1e-9 * max(1.0, abs(lo), abs(hi))
        if Xa.size and (Xa.min() < lo - tol or Xa.max() > hi + tol):
            raise ValueError(
                f"X outside width-profile range [{lo}, {hi}] m "
                f"(got [{Xa.min():.6g}, {Xa.max():.6g}]); no extrapolation rule is defined"
            )
        b = np.asarray(self._formula(Xa), dtype=np.float64) * np.ones_like(Xa)
        check_width_values(b)
        return b


class ConstantWidth(WidthProfile):
    """Uniform width ``b(X) = b0`` (benchmark only, Section 15.2/16.1)."""

    def __init__(
        self, b0: float, X_range: tuple[float, float] = (-math.inf, math.inf)
    ) -> None:
        check_width_values(b0)
        if not X_range[0] < X_range[1]:
            raise ValueError("X_range must satisfy X_min < X_max")
        self.b0 = float(b0)
        self._X_range = (float(X_range[0]), float(X_range[1]))

    @property
    def X_range(self) -> tuple[float, float]:
        return self._X_range

    def _formula(self, X: Any) -> Any:
        return self.b0 + 0.0 * X


class LinearWidth(WidthProfile):
    """Width varying linearly between two stations ``(X0, b0)`` and ``(X1, b1)``.

    ``b(X) = b0 + (b1 - b0) (X - X0) / (X1 - X0)`` for ``X0 <= X <= X1``.
    Both endpoint widths must be positive, which guarantees positivity on the
    whole interval. Evaluation outside ``[X0, X1]`` is rejected.
    """

    def __init__(self, X0: float, b0: float, X1: float, b1: float) -> None:
        check_width_values([b0, b1])
        if not X0 < X1:
            raise ValueError("LinearWidth requires X0 < X1")
        self.X0, self.b0, self.X1, self.b1 = float(X0), float(b0), float(X1), float(b1)

    @property
    def X_range(self) -> tuple[float, float]:
        return (self.X0, self.X1)

    @property
    def slope(self) -> float:
        """``db/dX`` [m/m]."""
        return (self.b1 - self.b0) / (self.X1 - self.X0)

    def _formula(self, X: Any) -> Any:
        return self.b0 + self.slope * (X - self.X0)


class PiecewiseLinearWidth(WidthProfile):
    """Width linearly interpolated between stations ``(X_i, b_i)``.

    ``b(X)`` is the continuous piecewise-linear interpolant of the stations,
    so it reproduces every station value exactly. Stations must be strictly
    increasing in ``X`` and every ``b_i`` positive, which guarantees ``b > 0``
    on ``[X_0, X_n]``. Evaluation outside ``[X_0, X_n]`` is rejected: no
    extrapolation rule is defined (Section 17.2).

    The interpolant has slope discontinuities at interior stations. A
    width-weighted integral is integrated exactly by polynomial quadrature
    only when element edges coincide with the stations; otherwise the
    quadrature error vanishes with mesh refinement.

    For UFL ``X`` the interpolant is written in ramp form,
    ``b(X) = b_0 + s_0 (X - X_0) + sum_{i=1}^{n-1} (s_i - s_{i-1}) max(X - X_i, 0)``,
    with segment slopes ``s_i``; for arrays ``numpy.interp`` is used.
    """

    def __init__(self, X: npt.ArrayLike, b: npt.ArrayLike) -> None:
        Xs = np.asarray(X, dtype=np.float64).reshape(-1)
        bs = np.asarray(b, dtype=np.float64).reshape(-1)
        if Xs.shape != bs.shape:
            raise ValueError("station positions X and widths b must have equal length")
        if Xs.size < 2:
            raise ValueError("PiecewiseLinearWidth requires at least two stations")
        if not np.all(np.isfinite(Xs)):
            raise ValueError("station positions X must be finite")
        if not np.all(np.diff(Xs) > 0.0):
            raise ValueError("station positions X must be strictly increasing")
        check_width_values(bs)
        self.X_stations = Xs
        self.b_stations = bs
        self.slopes = np.diff(bs) / np.diff(Xs)

    @classmethod
    def from_csv(
        cls,
        path: str | os.PathLike[str],
        X_column: str = "X_m",
        b_column: str = "b_m",
    ) -> PiecewiseLinearWidth:
        """Read stations from a comma-separated file with a header row.

        The file must contain columns ``X_column`` (reference heel-to-toe
        position, metres) and ``b_column`` (effective width, metres); other
        columns are ignored. Lines starting with ``#`` are comments. Rows
        must already be sorted by strictly increasing ``X``; the file is not
        reordered, so a mis-sorted file is reported rather than hidden.
        """
        name = os.fspath(path)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [
                [cell.strip() for cell in row]
                for row in csv.reader(fh)
                if row
                and any(c.strip() for c in row)
                and not row[0].lstrip().startswith("#")
            ]
        if not rows:
            raise ValueError(f"width CSV {name!r} has no header row")
        header, body = rows[0], rows[1:]
        missing = [c for c in (X_column, b_column) if c not in header]
        if missing:
            raise ValueError(
                f"width CSV {name!r} lacks column(s) {missing}; found {header}"
            )
        iX, ib = header.index(X_column), header.index(b_column)
        try:
            X = [float(r[iX]) for r in body]
            b = [float(r[ib]) for r in body]
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"width CSV {name!r} has missing or non-numeric values"
            ) from exc
        return cls(X, b)

    @property
    def X_range(self) -> tuple[float, float]:
        return (float(self.X_stations[0]), float(self.X_stations[-1]))

    def _formula(self, X: Any) -> Any:
        if not _is_ufl(X):
            return np.interp(X, self.X_stations, self.b_stations)
        import ufl

        Xs, s = self.X_stations, self.slopes
        expr = float(self.b_stations[0]) + float(s[0]) * (X - float(Xs[0]))
        for i in range(1, s.size):
            ds = float(s[i] - s[i - 1])
            if ds != 0.0:
                expr = expr + ds * ufl.max_value(X - float(Xs[i]), 0.0)
        return expr


def _is_ufl(X: Any) -> bool:
    try:
        import ufl
    except ImportError:  # pragma: no cover - pure-Python environment
        return False
    return isinstance(X, ufl.core.expr.Expr)
