"""FE-01 analytic tests of the NumPy plane-strain kinematics."""

import numpy as np
import pytest

from running_contact_validation.fe.kinematics import (
    green_lagrange_strain,
    jacobian,
    plane_strain_deformation_gradient,
    right_cauchy_green,
)

ATOL = 1e-14


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_zero_deformation():
    F = plane_strain_deformation_gradient(np.zeros((2, 2)))
    np.testing.assert_allclose(F, np.eye(3), atol=ATOL)
    np.testing.assert_allclose(right_cauchy_green(F), np.eye(3), atol=ATOL)
    np.testing.assert_allclose(green_lagrange_strain(F), np.zeros((3, 3)), atol=ATOL)
    assert jacobian(F) == pytest.approx(1.0)


@pytest.mark.parametrize("lx, ly", [(1.2, 1.0), (0.7, 1.0), (1.1, 0.6)])
def test_uniform_extension(lx, ly):
    # u = ((lx-1) X, (ly-1) Y): F = diag(lx, ly, 1)
    grad_u = np.diag([lx - 1.0, ly - 1.0])
    F = plane_strain_deformation_gradient(grad_u)
    np.testing.assert_allclose(F, np.diag([lx, ly, 1.0]), atol=ATOL)
    np.testing.assert_allclose(
        right_cauchy_green(F), np.diag([lx**2, ly**2, 1.0]), atol=ATOL
    )
    E = green_lagrange_strain(F)
    np.testing.assert_allclose(
        E, np.diag([(lx**2 - 1) / 2, (ly**2 - 1) / 2, 0.0]), atol=ATOL
    )
    assert jacobian(F) == pytest.approx(lx * ly)


@pytest.mark.parametrize("gamma", [0.05, 0.5, -1.3])
def test_simple_shear(gamma):
    # u = (gamma Y, 0)
    grad_u = np.array([[0.0, gamma], [0.0, 0.0]])
    F = plane_strain_deformation_gradient(grad_u)
    expected_C = np.array(
        [[1.0, gamma, 0.0], [gamma, 1.0 + gamma**2, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(right_cauchy_green(F), expected_C, atol=ATOL)
    expected_E = np.array(
        [[0.0, gamma / 2, 0.0], [gamma / 2, gamma**2 / 2, 0.0], [0, 0, 0]]
    )
    np.testing.assert_allclose(green_lagrange_strain(F), expected_E, atol=ATOL)
    assert jacobian(F) == pytest.approx(1.0)


@pytest.mark.parametrize("theta", [1e-3, np.pi / 6, np.pi / 2, 2.5])
def test_rigid_rotation_is_strain_free(theta):
    # u = (R - I) X: F = R embedded, C = I, E = 0 for finite rotations
    R = rotation(theta)
    F = plane_strain_deformation_gradient(R - np.eye(2))
    np.testing.assert_allclose(F[:2, :2], R, atol=ATOL)
    np.testing.assert_allclose(right_cauchy_green(F), np.eye(3), atol=ATOL)
    np.testing.assert_allclose(green_lagrange_strain(F), np.zeros((3, 3)), atol=ATOL)
    assert jacobian(F) == pytest.approx(1.0)


def test_rotation_superposed_on_strain_leaves_E_unchanged():
    rng = np.random.default_rng(0)
    grad_u = 0.3 * rng.standard_normal((2, 2))
    F = plane_strain_deformation_gradient(grad_u)
    Q = np.eye(3)
    Q[:2, :2] = rotation(0.9)
    np.testing.assert_allclose(
        green_lagrange_strain(Q @ F), green_lagrange_strain(F), atol=1e-13
    )


def test_plane_strain_structure_and_broadcasting():
    rng = np.random.default_rng(1)
    grad_u = 0.2 * rng.standard_normal((4, 5, 2, 2))
    F = plane_strain_deformation_gradient(grad_u)
    assert F.shape == (4, 5, 3, 3)
    np.testing.assert_array_equal(F[..., 2, 2], 1.0)
    np.testing.assert_array_equal(F[..., 2, :2], 0.0)
    np.testing.assert_array_equal(F[..., :2, 2], 0.0)
    E = green_lagrange_strain(F)
    np.testing.assert_allclose(E, np.swapaxes(E, -1, -2), atol=ATOL)
    np.testing.assert_allclose(E[..., 2, :], 0.0, atol=ATOL)
    # Pointwise consistency with explicit formula E = (H + H^T + H^T H)/2
    H = grad_u
    E2 = 0.5 * (H + np.swapaxes(H, -1, -2) + np.einsum("...kI,...kJ->...IJ", H, H))
    np.testing.assert_allclose(E[..., :2, :2], E2, atol=ATOL)
    np.testing.assert_allclose(jacobian(F), np.linalg.det(np.eye(2) + H), atol=ATOL)


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        plane_strain_deformation_gradient(np.zeros((3, 3)))
