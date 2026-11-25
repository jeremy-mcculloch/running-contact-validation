"""Plane-strain finite-strain kinematics in UFL (requires FEniCSx).

UFL counterpart of :mod:`running_contact_validation.fe.kinematics`. The
returned 3x3 tensors are the embedded plane-strain kinematic quantities on
which a full three-dimensional constitutive potential can later be evaluated
(``docs/scientific_specification.md`` Sections 22.1 D28, 22.2, 24.2)::

    F = [[1 + u_x,X,     u_x,Y,  0],
         [    u_y,X, 1 + u_y,Y,  0],
         [        0,         0,  1]],   C = F^T F,   E = (C - I)/2.

Gradients are taken with respect to the coordinates of the mesh on which
``u`` is defined, which must therefore be the **undeformed reference** mesh
(total-Lagrangian formulation; the mesh geometry is never moved).

No constitutive law is defined here.
"""

from __future__ import annotations

import ufl

__all__ = [
    "deformation_gradient",
    "green_lagrange_strain",
    "jacobian",
    "right_cauchy_green",
]


def deformation_gradient(u: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """Embedded 3x3 plane-strain deformation gradient of a 2D displacement ``u``."""
    shape = ufl.shape(u)
    if shape != (2,):
        raise ValueError(f"plane-strain displacement must have shape (2,); got {shape}")
    H = ufl.grad(u)
    return ufl.as_matrix(
        [
            [1 + H[0, 0], H[0, 1], 0],
            [H[1, 0], 1 + H[1, 1], 0],
            [0, 0, 1],
        ]
    )


def right_cauchy_green(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """``C = F^T F``."""
    return F.T * F


def green_lagrange_strain(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """``E = (F^T F - I)/2``."""
    return 0.5 * (right_cauchy_green(F) - ufl.Identity(ufl.shape(F)[0]))


def jacobian(F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """``J = det F``."""
    return ufl.det(F)
