"""One-term compressible Ogden/Storakers hyperfoam in UFL (requires FEniCSx).

UFL implementation of the approved Section 24.2 energy (see
:mod:`running_contact_validation.fe.hyperfoam` for the formula, parameter
convention, units and admissibility)::

    W = (2 mu/alpha^2) [ sum_i lambda_i^alpha - 3 + (J^(-alpha beta) - 1)/beta ].

The first Piola-Kirchhoff stress ``P = dW/dF`` and the material tangent
``A = d^2W/dF dF`` are obtained by UFL automatic differentiation of this same
energy; the FE residual and Jacobian are Gateaux derivatives of the
width-weighted stored energy ``Pi = sum_r int b(X) W_r(F) dA``.

Plane-strain evaluation of the 3D energy
----------------------------------------
For the embedded plane-strain deformation gradient (``F33 = 1``, zero
out-of-plane shear; :func:`running_contact_validation.fe.plane_strain.deformation_gradient`)
``C`` is block diagonal, so the three principal stretches are those of the
in-plane block ``C2`` plus ``lambda3^2 = C33`` (= 1). The energy is evaluated
exactly as the full 3D potential on that ``C``, keeping the explicit
dependence on ``C33`` and ``J = det F``, so ``P33`` (the transverse nominal
stress) is retained and generally nonzero. The functions here **require**
``F`` of this block form: ``C13`` and ``C23`` are not read. On the
plane-strain manifold the resulting ``P`` equals the 3D stress (the
out-of-plane shear stresses vanish by reflection symmetry), and the tangent
is exact for all in-plane variations used by the plane-strain residual.

Repeated in-plane stretches
---------------------------
With ``s = tr(C2)/2`` and ``D = ((C11 - C22)/2)^2 + C12^2`` the in-plane
eigenvalues are ``s +- sqrt(D)``. Direct differentiation of ``sqrt(D)`` is
singular at ``D = 0`` (including ``F = I``), although the symmetric sum
``g = (s + sqrt D)^p + (s - sqrt D)^p`` (``p = alpha/2``) is smooth. For
``r = D/s^2 < r_switch`` the exact even series
``g = 2 s^p sum_k binom(p, 2k) r^k`` is used, truncated where the next term
is below ``1e-16``; otherwise the closed form is used with ``D`` bounded away
from zero inside the square root so that neither branch can produce NaN.

Alternative: ``method="divided_difference"`` (no series, no threshold)
---------------------------------------------------------------------
The stress and tangent are built explicitly from the derivatives of
``g(s, D)`` (see :class:`_DividedDifference`)::

    dg/dC2       = (g_s/2) I + g_D Dev
    d(dg/dC2)[dC] = (g_ss/4) tr(dC) I + (g_sD/2)[(Dev:dC) I + tr(dC) Dev]
                    + K (N:dC) N + g_D dev(dC),     N = Dev/sqrt(D),

with ``S_in-plane = (4 mu/alpha^2) dg/dC2``, ``P = F S``, and
``dP[dF] = dF S + F dS[dC]``, ``dC = dF^T F + F^T dF``. The divided
differences ``g_D = f'[q+, q-]`` and ``g_sD = f''[q+, q-]`` are evaluated
without cancellation from Kahan's ``ln(u)/(u-1)`` and ``(w-1)/ln(w)``;
``K = g_DD D = (g_ss - 2 g_D)/4``. Only exact-equality guards are used. The
smooth out-of-plane/volumetric part keeps automatic differentiation. Use
:func:`internal_virtual_work` and :func:`tangent_form` (not
``ufl.derivative`` of the energy) with this method. ``series`` remains the
default.

``beta -> 0``
-------------
For ``|beta| < BETA_SERIES_THRESHOLD`` the volumetric term is evaluated as
``-alpha ln J * expm1(x)/x`` with ``x = -alpha beta ln J`` and the Taylor
series of ``expm1(x)/x`` (exact ``-alpha ln J`` at ``beta = 0``), otherwise
as ``(J^(-alpha beta) - 1)/beta``. Both are smooth in ``F`` and agree to
round-off at the switch.

Inverted states
---------------
UFL forms cannot raise. Call :func:`check_admissible_deformation` on the
displacement before evaluating/assembling (and on each Newton iterate);
``J <= 0`` at any quadrature point raises
:class:`~running_contact_validation.fe.hyperfoam.InadmissibleDeformationError`.
The check samples quadrature points of the same rule used for integration.

Parameters are Python floats per foam region, so they are compiled into the
form; changing a coefficient requires rebuilding the form.
"""

from __future__ import annotations

import math

import basix
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from running_contact_validation.fe import plane_strain as ps
from running_contact_validation.fe.hyperfoam import (
    FoamMaterialMap,
    HyperfoamParameters,
    InadmissibleDeformationError,
)
from running_contact_validation.fe.variable_width import WidthField

__all__ = [
    "BETA_SERIES_THRESHOLD",
    "METHODS",
    "STRETCH_SERIES_THRESHOLD",
    "check_admissible_deformation",
    "first_piola_kirchhoff_stress",
    "internal_virtual_work",
    "material_tangent",
    "minimum_jacobian",
    "stored_energy",
    "strain_energy_density",
    "stress_increment",
    "tangent_form",
]

#: Switch ``r = D/s^2`` between the even series and the closed form.
STRETCH_SERIES_THRESHOLD = 1e-3
#: Switch ``|beta|`` between the series and closed-form volumetric term.
BETA_SERIES_THRESHOLD = 1e-4
_SERIES_TERMS_BETA = 9  # x^0..x^8; |x| <= 1e-4*|alpha ln J|, remainder ~ x^9/10!


def _binom(p: float, n: int) -> float:
    out = 1.0
    for j in range(n):
        out *= (p - j) / (j + 1)
    return out


def _stretch_series_coefficients(
    p: float, r_max: float, tol: float = 1e-16
) -> list[float]:
    """``binom(p, 2k)`` for k = 0..K, with remainder ``|binom(p,2K+2)| r_max^(K+1) < tol``."""
    coeffs: list[float] = []
    for k in range(64):
        coeffs.append(_binom(p, 2 * k))
        nxt = abs(_binom(p, 2 * k + 2)) * r_max ** (k + 1)
        if nxt < tol:
            return coeffs
    raise ValueError(
        f"stretch series does not converge for alpha = {2 * p}; |alpha| too large"
    )


def _inplane_power_sum(C: ufl.core.expr.Expr, p: float) -> ufl.core.expr.Expr:
    """``lambda1^(2p) + lambda2^(2p)`` for the in-plane block of ``C``, smooth at D = 0."""
    a, b, c = C[0, 0], C[1, 1], C[0, 1]
    s = 0.5 * (a + b)
    D = (0.5 * (a - b)) ** 2 + c**2
    r = D / s**2
    rs = STRETCH_SERIES_THRESHOLD
    coeffs = _stretch_series_coefficients(p, rs)
    series = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        series = coeff + r * series
    series = 2.0 * s**p * series
    sqrtD = ufl.sqrt(ufl.max_value(D, rs * s**2))
    closed = (s + sqrtD) ** p + (s - sqrtD) ** p
    return ufl.conditional(ufl.lt(r, rs), series, closed)


def _volumetric_term(
    J: ufl.core.expr.Expr, alpha: float, beta: float
) -> ufl.core.expr.Expr:
    if abs(beta) >= BETA_SERIES_THRESHOLD:
        return (J ** (-alpha * beta) - 1.0) / beta
    lnJ = ufl.ln(J)
    if beta == 0.0:
        return -alpha * lnJ
    x = -alpha * beta * lnJ
    phi = 1.0 / math.factorial(_SERIES_TERMS_BETA)
    for k in range(_SERIES_TERMS_BETA - 2, -1, -1):  # expm1(x)/x = sum x^k/(k+1)!
        phi = 1.0 / math.factorial(k + 1) + x * phi
    return -alpha * lnJ * phi


def _check_shape(F: ufl.core.expr.Expr) -> None:
    if ufl.shape(F) != (3, 3):
        raise ValueError(
            f"hyperfoam requires the embedded 3x3 plane-strain F; got shape {ufl.shape(F)}"
        )


METHODS = ("series", "divided_difference")


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(
            f"unknown hyperfoam method {method!r}; expected one of {METHODS}"
        )


def _rest_energy(
    F: ufl.core.expr.Expr, params: HyperfoamParameters
) -> ufl.core.expr.Expr:
    """Out-of-plane and volumetric part ``(2mu/a^2)[C33^p - 3 + vol(J)]`` (smooth)."""
    mu, alpha, beta = params.mu, params.alpha, params.beta
    C33 = (F.T * F)[2, 2]
    vol = _volumetric_term(ufl.det(F), alpha, beta)
    return 2.0 * mu / alpha**2 * (C33 ** (0.5 * alpha) - 3.0 + vol)


def strain_energy_density(
    F: ufl.core.expr.Expr, params: HyperfoamParameters, method: str = "series"
) -> ufl.core.expr.Expr:
    """``W(F)`` [J/m^3] for an embedded plane-strain ``F`` (``C13 = C23 = 0``).

    ``method="series"`` uses the series/closed-form switch (differentiable by
    UFL); ``method="divided_difference"`` evaluates the closed form
    ``q+^p + q-^p`` directly and must **not** be differentiated by UFL (use
    :func:`first_piola_kirchhoff_stress` / :func:`material_tangent` instead).
    """
    _check_shape(F)
    _check_method(method)
    alpha = params.alpha
    C = F.T * F
    if method == "series":
        g = _inplane_power_sum(C, 0.5 * alpha)
    else:
        g = _DividedDifference(C, 0.5 * alpha).g
    return 2.0 * params.mu / alpha**2 * g + _rest_energy(F, params)


def first_piola_kirchhoff_stress(
    F: ufl.core.expr.Expr, params: HyperfoamParameters, method: str = "series"
) -> ufl.core.expr.Expr:
    """``P = dW/dF`` [Pa] (3x3; ``P33`` is the plane-strain transverse stress)."""
    _check_shape(F)
    _check_method(method)
    Fv = ufl.variable(F)
    if method == "series":
        return ufl.diff(strain_energy_density(Fv, params), Fv)
    P_rest = ufl.diff(_rest_energy(Fv, params), Fv)
    dd = _DividedDifference(F.T * F, 0.5 * params.alpha)
    return F * _embed(2.0 * _energy_scale(params) * dd.dg_dC()) + P_rest


def stress_increment(
    F: ufl.core.expr.Expr,
    dF: ufl.core.expr.Expr,
    params: HyperfoamParameters,
    method: str = "series",
) -> ufl.core.expr.Expr:
    """Directional derivative ``dP[dF] = A : dF`` of the stress (3x3) [Pa]."""
    _check_shape(F)
    _check_method(method)
    if method == "series":
        A = material_tangent(F, params, method)
        i, j, k, l = ufl.indices(4)
        return ufl.as_tensor(A[i, j, k, l] * dF[k, l], (i, j))
    Fv = ufl.variable(F)
    A_rest = ufl.diff(ufl.diff(_rest_energy(Fv, params), Fv), Fv)
    i, j, k, l = ufl.indices(4)
    dP_rest = ufl.as_tensor(A_rest[i, j, k, l] * dF[k, l], (i, j))
    C = F.T * F
    dd = _DividedDifference(C, 0.5 * params.alpha)
    scale = 2.0 * _energy_scale(params)
    S_g = _embed(scale * dd.dg_dC())
    dC = dF.T * F + F.T * dF
    dC2 = ufl.as_matrix([[dC[0, 0], dC[0, 1]], [dC[1, 0], dC[1, 1]]])
    dS_g = _embed(scale * dd.d2g_dC2(dC2))
    return dF * S_g + F * dS_g + dP_rest


def material_tangent(
    F: ufl.core.expr.Expr, params: HyperfoamParameters, method: str = "series"
) -> ufl.core.expr.Expr:
    """``A_iJkL = d^2W/dF_iJ dF_kL`` [Pa].

    ``series``: UFL automatic differentiation of the series energy.
    ``divided_difference``: explicit in-plane tangent from the divided-difference
    coefficients plus automatic differentiation of the smooth out-of-plane and
    volumetric part. Only plane-strain components (in-plane and ``F33``) are
    represented in either case.
    """
    _check_shape(F)
    _check_method(method)
    if method == "series":
        Fv = ufl.variable(F)
        P = ufl.diff(strain_energy_density(Fv, params), Fv)
        return ufl.diff(P, Fv)

    def unit(k: int, l: int) -> ufl.core.expr.Expr:
        return ufl.as_matrix(
            [[float((a, b) == (k, l)) for b in range(3)] for a in range(3)]
        )

    dP = [
        [stress_increment(F, unit(k, l), params, method) for l in range(3)]
        for k in range(3)
    ]
    return ufl.as_tensor(
        [
            [[[dP[k][l][a, b] for l in range(3)] for k in range(3)] for b in range(3)]
            for a in range(3)
        ]
    )


def _energy_scale(params: HyperfoamParameters) -> float:
    return 2.0 * params.mu / params.alpha**2


def _embed(M2: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    return ufl.as_matrix([[M2[0, 0], M2[0, 1], 0], [M2[1, 0], M2[1, 1], 0], [0, 0, 0]])


def _log_ratio(u: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """``ln(u)/(u - 1)`` (= ``log1p(z)/z`` with ``u = 1 + z``), Kahan's evaluation.

    Accurate to a few ulps for ``u`` near 1 because the rounding error of
    ``u`` cancels between numerator and denominator; ``u == 1`` -> 1.
    """
    one = ufl.eq(u, 1.0)
    den = ufl.conditional(one, 1.0, u - 1.0)
    return ufl.conditional(one, 1.0, ufl.ln(u) / den)


def _expm1_ratio(y: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """``expm1(y)/y`` by Kahan's ``(w - 1)/ln(w)``, ``w = exp(y)``; ``w == 1`` -> 1."""
    w = ufl.exp(y)
    one = ufl.eq(w, 1.0)
    den = ufl.conditional(one, 1.0, ufl.ln(w))
    return ufl.conditional(one, 1.0, (w - 1.0) / den)


class _DividedDifference:
    """In-plane stretch sum ``g = q+^p + q-^p`` and its (s, D) derivatives.

    ``q+- = s +- sqrt(D)`` are the eigenvalues of the in-plane block ``C2``,
    ``s = tr(C2)/2``, ``D = x^2 + y^2`` with ``x = (C11 - C22)/2``, ``y = C12``,
    ``Dev = C2 - s I = [[x, y], [y, -x]]``. With ``f(q) = q^p``::

        g_s  = f'(q+) + f'(q-)             g_ss = f''(q+) + f''(q-)
        g_D  = f'[q+, q-]                  g_sD = f''[q+, q-]
        K    = g_DD * D = (g_ss - 2 g_D)/4

    where ``h[a, b] = (h(a) - h(b))/(a - b)``. The divided difference of
    ``q^m`` is evaluated without cancellation as::

        (q+^m - q-^m)/(q+ - q-) = m q+^(m-1) * Lr(u) * Er(m ln u),   u = q-/q+,

    ``Lr(u) = ln(u)/(u - 1)`` and ``Er(y) = expm1(y)/y`` (Kahan's tricks).
    ``g_DD`` alone is never formed: it enters the tangent only as
    ``g_DD (Dev:dC) Dev = K (N:dC) N`` with ``N = Dev/sqrt(D)``, and ``K`` is
    accurate to ``O(eps |g_ss|)`` absolutely, which is the accuracy the tangent
    needs. Only exact-equality guards are used (``u == 1``, ``w == 1``,
    ``D == 0`` for the bounded direction ``N``); no series and no threshold.
    """

    def __init__(self, C: ufl.core.expr.Expr, p: float) -> None:
        self.p = p
        a, b, c = C[0, 0], C[1, 1], C[0, 1]
        self.s = 0.5 * (a + b)
        x = 0.5 * (a - b)
        y = c
        self.D = x**2 + y**2
        self.Dev = ufl.as_matrix([[x, y], [y, -x]])
        delta = ufl.sqrt(self.D)
        qp, qm = self.s + delta, self.s - delta
        self.qp, self.qm = qp, qm
        u = qm / qp
        self._lnu = ufl.ln(u)
        self._lr = _log_ratio(u)
        f1 = lambda q: p * q ** (p - 1.0)
        f2 = lambda q: p * (p - 1.0) * q ** (p - 2.0)
        self.g = qp**p + qm**p
        self.g_s = f1(qp) + f1(qm)
        self.g_ss = f2(qp) + f2(qm)
        self.g_D = p * self._power_divided_difference(p - 1.0)
        self.g_sD = p * (p - 1.0) * self._power_divided_difference(p - 2.0)
        self.K = 0.25 * self.g_ss - 0.5 * self.g_D
        delta_safe = ufl.conditional(ufl.eq(self.D, 0.0), 1.0, delta)
        self.N = self.Dev / delta_safe

    def _power_divided_difference(self, m: float) -> ufl.core.expr.Expr:
        if m == 0.0:
            return ufl.as_ufl(0.0)
        return m * self.qp ** (m - 1.0) * self._lr * _expm1_ratio(m * self._lnu)

    def dg_dC(self) -> ufl.core.expr.Expr:
        """``dg/dC2 = (g_s/2) I + g_D Dev`` (2x2, symmetric-tensor derivative)."""
        return 0.5 * self.g_s * ufl.Identity(2) + self.g_D * self.Dev

    def d2g_dC2(self, dC: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """``d(dg/dC2)[dC]`` for a symmetric 2x2 increment ``dC``."""
        I2 = ufl.Identity(2)
        tr = ufl.tr(dC)
        dev_dC = dC - 0.5 * tr * I2
        return (
            0.25 * self.g_ss * tr * I2
            + 0.5 * self.g_sD * (ufl.inner(self.Dev, dC) * I2 + tr * self.Dev)
            + self.K * ufl.inner(self.N, dC) * self.N
            + self.g_D * dev_dC
        )


def _region_measures(
    materials: FoamMaterialMap,
    width: WidthField,
    cell_tags: mesh.MeshTags | None,
) -> list[tuple[HyperfoamParameters, ufl.Measure]]:
    if not isinstance(materials, FoamMaterialMap):
        materials = FoamMaterialMap(materials)
    if cell_tags is None:
        if len(materials) != 1:
            raise ValueError(
                "cell_tags are required when more than one foam region is given"
            )
        (params,) = materials.values()
        return [(params, width.dx)]
    domain = width.domain
    tdim = domain.topology.dim
    if cell_tags.dim != tdim:
        raise ValueError("cell_tags must tag cells of the width-field mesh")
    imap = domain.topology.index_map(tdim)
    n_owned = imap.size_local
    owned = cell_tags.indices < n_owned
    local_tags = set(np.unique(cell_tags.values[owned]).tolist())
    n_tagged = int(np.count_nonzero(owned))
    all_tags = set().union(*domain.comm.allgather(local_tags))
    total_tagged = domain.comm.allreduce(n_tagged, op=MPI.SUM)
    total_cells = imap.size_global
    missing = sorted(all_tags - set(materials))
    if missing:
        raise ValueError(f"no hyperfoam parameters for tagged foam region(s) {missing}")
    if total_tagged != total_cells:
        raise ValueError(
            f"{total_cells - total_tagged} cell(s) carry no foam-region tag; every cell "
            "must belong to a foam region"
        )
    dx = width.dx(subdomain_data=cell_tags)
    return [(materials[tag], dx(tag)) for tag in sorted(all_tags)]


def stored_energy(
    u: fem.Function,
    materials: FoamMaterialMap,
    width: WidthField,
    cell_tags: mesh.MeshTags | None = None,
    method: str = "series",
) -> ufl.Form:
    """Width-weighted elastic stored energy ``Pi = sum_r int b(X) W_r(F(u)) dA`` [J].

    ``u`` is the reference-mesh displacement [m]. With ``cell_tags`` each
    foam region tag must have an entry in ``materials`` and every cell must be
    tagged; without tags, ``materials`` must contain exactly one region.
    Integration uses the width field's quadrature rule, so ``b`` is sampled at
    the integration points.

    Residual and consistent Jacobian::

        R = ufl.derivative(Pi, u, v);   K = ufl.derivative(R, u, du)

    This holds for ``method="series"`` only. For ``"divided_difference"``
    the energy is the closed form and must not be differentiated by UFL
    (``sqrt(D)`` is not differentiable at repeated stretches); use
    :func:`internal_virtual_work` and :func:`tangent_form` instead.
    """
    F = ps.deformation_gradient(u)
    b = width.function
    terms = [
        strain_energy_density(F, params, method) * b * dx
        for params, dx in _region_measures(materials, width, cell_tags)
    ]
    return sum(terms[1:], terms[0])


def internal_virtual_work(
    u: fem.Function,
    v: ufl.Argument | fem.Function,
    materials: FoamMaterialMap,
    width: WidthField,
    cell_tags: mesh.MeshTags | None = None,
    method: str = "series",
) -> ufl.Form:
    """``sum_r int b(X) P_r(F(u)) : dF(v) dA`` with ``dF = Grad v`` embedded (in-plane).

    Equal to ``derivative(stored_energy(...), u, v)``; provided so the
    stress-based residual can be compared with the energy-based one.
    """
    F = ps.deformation_gradient(u)
    dF = _embed(ufl.grad(v))
    b = width.function
    terms = [
        ufl.inner(first_piola_kirchhoff_stress(F, params, method), dF) * b * dx
        for params, dx in _region_measures(materials, width, cell_tags)
    ]
    return sum(terms[1:], terms[0])


def tangent_form(
    u: fem.Function,
    du: ufl.Argument,
    v: ufl.Argument,
    materials: FoamMaterialMap,
    width: WidthField,
    cell_tags: mesh.MeshTags | None = None,
    method: str = "series",
) -> ufl.Form:
    """Consistent Jacobian ``sum_r int b(X) (A_r : dF(du)) : dF(v) dA`` of
    :func:`internal_virtual_work`.

    ``series``: ``ufl.derivative`` of the residual. ``divided_difference``:
    the explicit stress increment :func:`stress_increment`.
    """
    _check_method(method)
    if method == "series":
        return ufl.derivative(
            internal_virtual_work(u, v, materials, width, cell_tags, method), u, du
        )
    F = ps.deformation_gradient(u)
    dF_u, dF_v = _embed(ufl.grad(du)), _embed(ufl.grad(v))
    b = width.function
    terms = [
        ufl.inner(stress_increment(F, dF_u, params, method), dF_v) * b * dx
        for params, dx in _region_measures(materials, width, cell_tags)
    ]
    return sum(terms[1:], terms[0])


def minimum_jacobian(u: fem.Function, degree: int) -> float:
    """Global minimum of ``J = det F(u)`` over quadrature points of the given degree."""
    domain = u.function_space.mesh
    tdim = domain.topology.dim
    n_owned = domain.topology.index_map(tdim).size_local
    cell_type = basix.CellType[domain.topology.cell_name()]
    points, _ = basix.make_quadrature(cell_type, degree)
    expr = fem.Expression(ps.jacobian(ps.deformation_gradient(u)), points)
    local = expr.eval(domain, np.arange(n_owned, dtype=np.int32))
    if local.size == 0:
        local_min = math.inf
    elif not np.all(np.isfinite(local)):
        local_min = -math.inf  # non-finite J is treated as inadmissible
    else:
        local_min = float(np.min(local))
    return float(domain.comm.allreduce(local_min, op=MPI.MIN))


def check_admissible_deformation(u: fem.Function, degree: int) -> float:
    """Return ``min J``; raise :class:`InadmissibleDeformationError` if ``min J <= 0``.

    Collective over the mesh communicator (every rank raises together).
    """
    jmin = minimum_jacobian(u, degree)
    if not jmin > 0.0:
        raise InadmissibleDeformationError(
            f"inadmissible deformation: min J = det F = {jmin:.6g} <= 0 at a quadrature "
            f"point (degree {degree}); inverted elements are outside the hyperfoam domain"
        )
    return jmin
