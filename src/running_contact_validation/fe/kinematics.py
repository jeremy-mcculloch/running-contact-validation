"""Plane-strain finite-strain kinematics (pure NumPy reference implementation).

This module contains **kinematics only**; it has no constitutive content and
no FEniCSx dependency, so it can be used for unit tests, quadrature-point
post-processing, and as an independent reference for the UFL implementation
in :mod:`running_contact_validation.fe.plane_strain`.

Conventions (``docs/scientific_specification.md``, Sections 3, 19.2, 22.1 D28,
24.2):

* Reference (material) coordinates ``X = (X, Y)`` in the sagittal plane, in
  metres; ``X`` is the reference heel-to-toe (forward) direction and ``Y`` is
  upward. The out-of-plane direction is ``Z``.
* Plane strain: the out-of-plane displacement and all out-of-plane
  displacement-gradient components vanish, so the embedded 3x3 deformation
  gradient is::

      F = [[1 + u_x,X,     u_x,Y,  0],
           [    u_y,X, 1 + u_y,Y,  0],
           [        0,         0,  1]]

* ``C = F^T F`` (right Cauchy-Green) and ``E = (C - I)/2`` (Green-Lagrange).

All functions operate on the trailing two axes and broadcast over leading
axes (e.g. arrays of quadrature points).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "green_lagrange_strain",
    "jacobian",
    "plane_strain_deformation_gradient",
    "right_cauchy_green",
]

FloatArray = npt.NDArray[np.float64]


def plane_strain_deformation_gradient(grad_u: npt.ArrayLike) -> FloatArray:
    """Return the embedded 3x3 plane-strain deformation gradient.

    Parameters
    ----------
    grad_u:
        In-plane reference displacement gradient ``Grad_X u`` with shape
        ``(..., 2, 2)`` and components ``grad_u[..., i, J] = du_i/dX_J``
        (dimensionless).

    Returns
    -------
    numpy.ndarray
        ``F`` with shape ``(..., 3, 3)``; ``F[..., 2, 2] == 1`` and the
        remaining out-of-plane components are zero.
    """
    H = np.asarray(grad_u, dtype=np.float64)
    if H.shape[-2:] != (2, 2):
        raise ValueError(f"grad_u must have trailing shape (2, 2); got {H.shape}")
    F = np.zeros(H.shape[:-2] + (3, 3), dtype=np.float64)
    F[..., :2, :2] = H
    F[..., 0, 0] += 1.0
    F[..., 1, 1] += 1.0
    F[..., 2, 2] = 1.0
    return F


def right_cauchy_green(F: npt.ArrayLike) -> FloatArray:
    """Return ``C = F^T F`` for ``F`` of shape ``(..., 3, 3)``."""
    F = np.asarray(F, dtype=np.float64)
    return np.einsum("...kI,...kJ->...IJ", F, F)


def green_lagrange_strain(F: npt.ArrayLike) -> FloatArray:
    """Return the Green-Lagrange strain ``E = (F^T F - I)/2``."""
    C = right_cauchy_green(F)
    return 0.5 * (C - np.eye(C.shape[-1]))


def jacobian(F: npt.ArrayLike) -> FloatArray:
    """Return ``J = det F``. For plane strain, ``J`` equals the in-plane determinant."""
    return np.linalg.det(np.asarray(F, dtype=np.float64))
