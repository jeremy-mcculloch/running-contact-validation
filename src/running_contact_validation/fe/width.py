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

Scope note: the production ``b(X)`` comes from measured midsole width
(approved, D21e), but the numerical interpolation rule for measured stations
is not yet selected (Section 17.2, 24.6/R8). Only analytic constant and linear
profiles are provided here; they are verification benchmarks, not a selected
measured-width interpolant.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "ConstantWidth",
    "LinearWidth",
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


def _is_ufl(X: Any) -> bool:
    try:
        import ufl
    except ImportError:  # pragma: no cover - pure-Python environment
        return False
    return isinstance(X, ufl.core.expr.Expr)
