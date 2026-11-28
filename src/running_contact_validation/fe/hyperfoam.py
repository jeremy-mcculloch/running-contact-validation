"""One-term compressible Ogden/Storakers hyperfoam (pure NumPy reference).

Implements the approved published energy of ``docs/scientific_specification.md``
Section 24.2 (decisions D25-D27, Abaqus ``HYPERFOAM`` convention, ``N = 1``)::

    W(F) = (2 mu / alpha^2) [ lambda1^alpha + lambda2^alpha + lambda3^alpha - 3
                              + (J^(-alpha beta) - 1) / beta ],
    J = det F = lambda1 lambda2 lambda3 > 0,

where ``lambda_i`` are the **full** principal stretches (square roots of the
eigenvalues of ``C = F^T F``; not isochoric stretches). ``W`` is an energy per
unit *reference* volume [J/m^3 = Pa].

Parameters and units
--------------------
* ``mu`` [Pa]: initial (small-strain) shear modulus in this normalization.
* ``alpha`` [-]: nonlinear stretch exponent (nonzero).
* ``beta`` [-]: compressibility parameter. The ``beta -> 0`` limit is
  continuous, ``(J^(-alpha beta) - 1)/beta -> -alpha ln J``.

Small-strain moduli implied by this normalization (derived by expanding ``W``
to second order about ``F = I``; see the tests)::

    shear modulus   mu_0 = mu
    Lame lambda_0        = 2 mu beta
    bulk modulus    K_0  = 2 mu (beta + 1/3)
    Poisson ratio   nu_0 = beta / (1 + 2 beta)      (beta = nu_0 / (1 - 2 nu_0))

Admissibility enforced here: ``mu > 0``, ``alpha != 0`` and ``beta > -1/3``
(positive small-strain shear and bulk moduli), all finite. These are
*necessary* small-strain conditions only; stability over the strain range
actually reached must be checked separately for each identified parameter set
(spec Section 24.2). The energy is finite for every ``F`` with ``J > 0``;
``J <= 0`` raises :class:`InadmissibleDeformationError`.

Stress
------
Because ``W`` is an isotropic function of ``C``, the second Piola-Kirchhoff
stress is the matrix function (well defined at repeated stretches)::

    S = 2 dW/dC = (2 mu / alpha) [ C^((alpha - 2)/2) - J^(-alpha beta) C^(-1) ],
    P = F S,        sigma = P F^T / J.

so ``P(I) = 0``. Under plane strain (``F33 = 1``) the out-of-plane stress
``P33 = (2 mu / alpha)(1 - J^(-alpha beta))`` is generally nonzero.

This module evaluates the full three-dimensional energy for any ``3x3`` ``F``
(it does not assume plane strain) and is an independent reference for the UFL
implementation in :mod:`running_contact_validation.fe.hyperfoam_ufl`. It
contains no viscous stress (FE-03).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "FoamMaterialMap",
    "HyperfoamParameters",
    "InadmissibleDeformationError",
    "cauchy_stress",
    "check_admissible_jacobian",
    "first_piola_kirchhoff_stress",
    "inplane_stretch_sum_coefficients",
    "principal_stretches",
    "second_piola_kirchhoff_stress",
    "strain_energy_density",
    "volumetric_term",
]

FloatArray = npt.NDArray[np.float64]


class InadmissibleDeformationError(ValueError):
    """Raised for deformation gradients with ``J = det F <= 0`` (or non-finite)."""


@dataclass(frozen=True)
class HyperfoamParameters:
    """One-term hyperfoam coefficients in the Section 24.2 (Abaqus) convention.

    Attributes
    ----------
    mu:
        Initial shear modulus [Pa]; must be > 0.
    alpha:
        Stretch exponent [-]; must be nonzero.
    beta:
        Compressibility parameter [-]; must be > -1/3. ``beta = 0`` is allowed
        and evaluated through the continuous limit.
    """

    mu: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for name in ("mu", "alpha", "beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.floating, np.integer)
            ):
                raise TypeError(
                    f"hyperfoam {name} must be a real number; got {value!r}"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"hyperfoam {name} must be finite; got {value}")
            object.__setattr__(self, name, value)
        if not self.mu > 0.0:
            raise ValueError(f"hyperfoam mu [Pa] must be > 0; got {self.mu}")
        if self.alpha == 0.0:
            raise ValueError("hyperfoam alpha must be nonzero")
        if not self.beta > -1.0 / 3.0:
            raise ValueError(
                f"hyperfoam beta must be > -1/3 (positive small-strain bulk modulus); "
                f"got {self.beta}"
            )

    @property
    def shear_modulus(self) -> float:
        """Small-strain shear modulus ``mu_0 = mu`` [Pa]."""
        return self.mu

    @property
    def lame_lambda(self) -> float:
        """Small-strain Lame parameter ``lambda_0 = 2 mu beta`` [Pa]."""
        return 2.0 * self.mu * self.beta

    @property
    def bulk_modulus(self) -> float:
        """Small-strain bulk modulus ``K_0 = 2 mu (beta + 1/3)`` [Pa]."""
        return 2.0 * self.mu * (self.beta + 1.0 / 3.0)

    @property
    def poisson_ratio(self) -> float:
        """Small-strain Poisson ratio ``nu_0 = beta / (1 + 2 beta)`` [-]."""
        return self.beta / (1.0 + 2.0 * self.beta)


class FoamMaterialMap(Mapping[int, HyperfoamParameters]):
    """Hyperfoam parameters keyed by foam-region identifier (mesh cell tag).

    Region identifiers are non-negative integers matching the values of a
    DOLFINx cell ``MeshTags``. Each region has an independent parameter set
    (spec D22). The map is immutable after construction.
    """

    def __init__(self, materials: Mapping[int, HyperfoamParameters]) -> None:
        if len(materials) == 0:
            raise ValueError("FoamMaterialMap requires at least one foam region")
        checked: dict[int, HyperfoamParameters] = {}
        for key, params in materials.items():
            if (
                isinstance(key, bool)
                or not isinstance(key, (int, np.integer))
                or key < 0
            ):
                raise ValueError(
                    f"foam region id must be a non-negative integer; got {key!r}"
                )
            if not isinstance(params, HyperfoamParameters):
                raise TypeError(
                    f"region {key}: expected HyperfoamParameters; got {type(params).__name__}"
                )
            checked[int(key)] = params
        self._materials = checked

    def __getitem__(self, key: int) -> HyperfoamParameters:
        try:
            return self._materials[key]
        except KeyError:
            raise KeyError(f"no hyperfoam parameters for foam region {key}") from None

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._materials))

    def __len__(self) -> int:
        return len(self._materials)

    def __repr__(self) -> str:
        return f"FoamMaterialMap({dict(sorted(self._materials.items()))!r})"


def check_admissible_jacobian(F: npt.ArrayLike) -> FloatArray:
    """Return ``J = det F``, raising :class:`InadmissibleDeformationError` if any ``J <= 0``."""
    F = np.asarray(F, dtype=np.float64)
    if F.shape[-2:] != (3, 3):
        raise ValueError(f"F must have trailing shape (3, 3); got {F.shape}")
    with np.errstate(invalid="ignore"):
        J = np.linalg.det(F)
    bad = ~(np.isfinite(J) & (J > 0.0))
    if np.any(bad):
        raise InadmissibleDeformationError(
            f"inadmissible deformation: J = det F must be > 0 (got min J = {np.min(J):.6g}, "
            f"{int(np.count_nonzero(bad))} invalid state(s)); inverted or degenerate "
            "states are outside the hyperfoam domain"
        )
    return J


def volumetric_term(J: npt.ArrayLike, alpha: float, beta: float) -> FloatArray:
    """``(J^(-alpha beta) - 1)/beta``, continuous in ``beta`` (``-alpha ln J`` at 0).

    Uses ``expm1`` so that small ``|beta|`` loses no precision.
    """
    lnJ = np.log(np.asarray(J, dtype=np.float64))
    if beta == 0.0:
        return -alpha * lnJ
    return np.expm1(-alpha * beta * lnJ) / beta


def _sym_eig(C: FloatArray) -> tuple[FloatArray, FloatArray]:
    return np.linalg.eigh(0.5 * (C + np.swapaxes(C, -1, -2)))


def _matrix_power(c: FloatArray, Q: FloatArray, p: float) -> FloatArray:
    return np.einsum("...ik,...k,...jk->...ij", Q, c**p, Q)


def principal_stretches(F: npt.ArrayLike) -> FloatArray:
    """Full principal stretches ``lambda_i`` (ascending), shape ``(..., 3)``."""
    F = np.asarray(F, dtype=np.float64)
    check_admissible_jacobian(F)
    c, _ = _sym_eig(np.einsum("...kI,...kJ->...IJ", F, F))
    return np.sqrt(c)


def strain_energy_density(F: npt.ArrayLike, params: HyperfoamParameters) -> FloatArray:
    """Stored energy ``W(F)`` per unit reference volume [J/m^3]."""
    F = np.asarray(F, dtype=np.float64)
    J = check_admissible_jacobian(F)
    lam = principal_stretches(F)
    a = params.alpha
    dev = np.sum(lam**a, axis=-1) - 3.0
    return 2.0 * params.mu / a**2 * (dev + volumetric_term(J, a, params.beta))


def second_piola_kirchhoff_stress(
    F: npt.ArrayLike, params: HyperfoamParameters
) -> FloatArray:
    """``S = (2 mu/alpha)[C^((alpha-2)/2) - J^(-alpha beta) C^(-1)]`` [Pa]."""
    F = np.asarray(F, dtype=np.float64)
    J = check_admissible_jacobian(F)
    C = np.einsum("...kI,...kJ->...IJ", F, F)
    c, Q = _sym_eig(C)
    a, b = params.alpha, params.beta
    vol = J ** (-a * b)
    S = _matrix_power(c, Q, 0.5 * a - 1.0) - vol[..., None, None] * _matrix_power(
        c, Q, -1.0
    )
    return 2.0 * params.mu / a * S


def first_piola_kirchhoff_stress(
    F: npt.ArrayLike, params: HyperfoamParameters
) -> FloatArray:
    """``P = dW/dF = F S`` [Pa] (nominal stress, reference-area based)."""
    F = np.asarray(F, dtype=np.float64)
    return F @ second_piola_kirchhoff_stress(F, params)


def cauchy_stress(F: npt.ArrayLike, params: HyperfoamParameters) -> FloatArray:
    """``sigma = P F^T / J`` [Pa]."""
    F = np.asarray(F, dtype=np.float64)
    J = check_admissible_jacobian(F)
    P = first_piola_kirchhoff_stress(F, params)
    return (P @ np.swapaxes(F, -1, -2)) / J[..., None, None]


def inplane_stretch_sum_coefficients(
    s: npt.ArrayLike, D: npt.ArrayLike, p: float
) -> dict[str, FloatArray]:
    """Divided-difference coefficients of ``g = q+^p + q-^p``, ``q+- = s +- sqrt(D)``.

    NumPy reference for the ``divided_difference`` method of
    :mod:`running_contact_validation.fe.hyperfoam_ufl` (which uses Kahan's
    ``ln(u)/(u-1)`` and ``(w-1)/ln(w)`` instead of ``log1p``/``expm1``).
    Returns ``g, g_s, g_D, g_ss, g_sD`` and ``K = g_DD * D = (g_ss - 2 g_D)/4``;
    the divided differences ``(q+^m - q-^m)/(q+ - q-)`` are evaluated as
    ``m q+^(m-1) * log1p(z)/z * expm1(m L)/(m L)`` with ``z = -2 sqrt(D)/q+``,
    ``L = log1p(z)``. ``s`` and ``D`` are in units of ``C`` (dimensionless).
    """
    s = np.asarray(s, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    delta = np.sqrt(D)
    qp, qm = s + delta, s - delta
    z = -2.0 * delta / qp
    L = np.log1p(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.where(z == 0.0, 1.0, L / np.where(z == 0.0, 1.0, z))

    def divdiff(m: float) -> FloatArray:
        if m == 0.0:
            return np.zeros_like(s)
        y = m * L
        with np.errstate(invalid="ignore", divide="ignore"):
            er = np.where(y == 0.0, 1.0, np.expm1(y) / np.where(y == 0.0, 1.0, y))
        return m * qp ** (m - 1.0) * lr * er

    g_ss = p * (p - 1.0) * (qp ** (p - 2.0) + qm ** (p - 2.0))
    g_D = p * divdiff(p - 1.0)
    return {
        "g": qp**p + qm**p,
        "g_s": p * (qp ** (p - 1.0) + qm ** (p - 1.0)),
        "g_D": g_D,
        "g_ss": g_ss,
        "g_sD": p * (p - 1.0) * divdiff(p - 2.0),
        "K": 0.25 * g_ss - 0.5 * g_D,
    }
