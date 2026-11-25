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
    "STRETCH_SERIES_THRESHOLD",
    "check_admissible_deformation",
    "first_piola_kirchhoff_stress",
    "internal_virtual_work",
    "material_tangent",
    "minimum_jacobian",
    "stored_energy",
    "strain_energy_density",
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


def strain_energy_density(
    F: ufl.core.expr.Expr, params: HyperfoamParameters
) -> ufl.core.expr.Expr:
    """``W(F)`` [J/m^3] for an embedded plane-strain ``F`` (``C13 = C23 = 0``)."""
    _check_shape(F)
    mu, alpha, beta = params.mu, params.alpha, params.beta
    C = F.T * F
    J = ufl.det(F)
    p = 0.5 * alpha
    lam_sum = _inplane_power_sum(C, p) + C[2, 2] ** p
    return 2.0 * mu / alpha**2 * (lam_sum - 3.0 + _volumetric_term(J, alpha, beta))


def first_piola_kirchhoff_stress(
    F: ufl.core.expr.Expr, params: HyperfoamParameters
) -> ufl.core.expr.Expr:
    """``P = dW/dF`` [Pa] (3x3; ``P33`` is the plane-strain transverse stress)."""
    _check_shape(F)
    Fv = ufl.variable(F)
    return ufl.diff(strain_energy_density(Fv, params), Fv)


def material_tangent(
    F: ufl.core.expr.Expr, params: HyperfoamParameters
) -> ufl.core.expr.Expr:
    """``A_iJkL = d^2W/dF_iJ dF_kL`` [Pa] by automatic differentiation."""
    _check_shape(F)
    Fv = ufl.variable(F)
    P = ufl.diff(strain_energy_density(Fv, params), Fv)
    return ufl.diff(P, Fv)


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
) -> ufl.Form:
    """Width-weighted elastic stored energy ``Pi = sum_r int b(X) W_r(F(u)) dA`` [J].

    ``u`` is the reference-mesh displacement [m]. With ``cell_tags`` each
    foam region tag must have an entry in ``materials`` and every cell must be
    tagged; without tags, ``materials`` must contain exactly one region.
    Integration uses the width field's quadrature rule, so ``b`` is sampled at
    the integration points.

    Residual and consistent Jacobian::

        R = ufl.derivative(Pi, u, v);   K = ufl.derivative(R, u, du)
    """
    F = ps.deformation_gradient(u)
    b = width.function
    terms = [
        strain_energy_density(F, params) * b * dx
        for params, dx in _region_measures(materials, width, cell_tags)
    ]
    return sum(terms[1:], terms[0])


def internal_virtual_work(
    u: fem.Function,
    v: ufl.Argument | fem.Function,
    materials: FoamMaterialMap,
    width: WidthField,
    cell_tags: mesh.MeshTags | None = None,
) -> ufl.Form:
    """``sum_r int b(X) P_r(F(u)) : dF(v) dA`` with ``dF = Grad v`` embedded (in-plane).

    Equal to ``derivative(stored_energy(...), u, v)``; provided so the
    stress-based residual can be compared with the energy-based one.
    """
    F = ps.deformation_gradient(u)
    gv = ufl.grad(v)
    dF = ufl.as_matrix([[gv[0, 0], gv[0, 1], 0], [gv[1, 0], gv[1, 1], 0], [0, 0, 0]])
    b = width.function
    terms = [
        ufl.inner(first_piola_kirchhoff_stress(F, params), dF) * b * dx
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
