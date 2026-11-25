"""FE-01 benchmarks: UFL plane-strain kinematics on DOLFINx meshes.

Analytic homogeneous deformations (zero, uniform extension, simple shear,
finite rigid rotation) are represented exactly by P1 displacements, so the
FE deformation gradient, right Cauchy-Green tensor and Green-Lagrange strain
must match the analytic tensors to round-off at every quadrature point. A
quadratic, non-homogeneous P2 field checks pointwise agreement with the
independent NumPy implementation at quadrature points.
"""

import basix
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from running_contact_validation.fe import kinematics as npk
from running_contact_validation.fe import plane_strain as ps
from running_contact_validation.fe.variable_width import (
    assemble_total,
    reference_quadrature_points,
)

L, H = 0.28, 0.03  # shoe-like reference section [m]
CELL_TYPES = [mesh.CellType.triangle, mesh.CellType.quadrilateral]


def make_domain(cell_type, n=(8, 3)):
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, H])],
        list(n),
        cell_type=cell_type,
    )


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


HOMOGENEOUS_CASES = {
    "zero": np.zeros((2, 2)),
    "uniform_extension": np.diag([0.15, -0.25]),
    "simple_shear": np.array([[0.0, 0.4], [0.0, 0.0]]),
    "rigid_rotation": rotation(np.pi / 5) - np.eye(2),
}


def interpolate_linear(domain, grad_u, degree=1):
    V = fem.functionspace(domain, ("Lagrange", degree, (2,)))
    u = fem.Function(V)
    G = np.asarray(grad_u)
    u.interpolate(lambda x: G @ x[:2])
    return u


def eval_at_quadrature(domain, expr, degree):
    cell_type = basix.CellType[domain.topology.cell_name()]
    ref_points, _ = basix.make_quadrature(cell_type, degree)
    tdim = domain.topology.dim
    imap = domain.topology.index_map(tdim)
    cells = np.arange(imap.size_local, dtype=np.int32)
    values = fem.Expression(expr, ref_points).eval(domain, cells)
    return values.reshape(len(cells), ref_points.shape[0], *ufl.shape(expr))


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("case", list(HOMOGENEOUS_CASES))
def test_homogeneous_kinematics_match_analytic(case, cell_type):
    grad_u = HOMOGENEOUS_CASES[case]
    domain = make_domain(cell_type)
    u = interpolate_linear(domain, grad_u)

    F = ps.deformation_gradient(u)
    C = ps.right_cauchy_green(F)
    E = ps.green_lagrange_strain(F)

    F_ex = npk.plane_strain_deformation_gradient(grad_u)
    C_ex = npk.right_cauchy_green(F_ex)
    E_ex = npk.green_lagrange_strain(F_ex)

    for name, num, exact in (("F", F, F_ex), ("C", C, C_ex), ("E", E, E_ex)):
        vals = eval_at_quadrature(domain, num, degree=2)
        np.testing.assert_allclose(
            vals, np.broadcast_to(exact, vals.shape), atol=1e-12, err_msg=name
        )
        # Integrated squared error over the domain (global, MPI-safe)
        diff = num - ufl.as_matrix(exact.tolist())
        err = assemble_total(ufl.inner(diff, diff) * ufl.dx(domain=domain))
        assert err < 1e-24 * L * H, (name, err)

    J = eval_at_quadrature(domain, ps.jacobian(F), degree=2)
    np.testing.assert_allclose(J, npk.jacobian(F_ex), atol=1e-12)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_rigid_rotation_strain_energy_proxy_vanishes(cell_type):
    """Finite rigid rotation: integral of E:E is round-off, while small-strain
    measure sym(grad u) would be O(theta^2) wrong."""
    theta = np.pi / 3
    domain = make_domain(cell_type)
    u = interpolate_linear(domain, rotation(theta) - np.eye(2))
    E = ps.green_lagrange_strain(ps.deformation_gradient(u))
    dx = ufl.dx(domain=domain)
    assert assemble_total(ufl.inner(E, E) * dx) < 1e-26
    eps = ufl.sym(ufl.grad(u))
    assert assemble_total(ufl.inner(eps, eps) * dx) > 1e-3 * L * H


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_nonhomogeneous_field_pointwise_matches_numpy(cell_type):
    """P2 displacement u = (a X^2 + c Y^2, b X Y) is reproduced exactly; its
    UFL kinematics at quadrature points must match the NumPy reference."""
    a, b, c = 2.0, -3.0, 50.0
    domain = make_domain(cell_type)
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = fem.Function(V)
    u.interpolate(lambda x: np.vstack([a * x[0] ** 2 + c * x[1] ** 2, b * x[0] * x[1]]))

    degree = 4
    points, _ = reference_quadrature_points(domain, degree)
    imap = domain.topology.index_map(domain.topology.dim)
    points = points[: imap.size_local]
    X, Y = points[..., 0], points[..., 1]
    grad_u = np.empty(X.shape + (2, 2))
    grad_u[..., 0, 0] = 2 * a * X
    grad_u[..., 0, 1] = 2 * c * Y
    grad_u[..., 1, 0] = b * Y
    grad_u[..., 1, 1] = b * X
    F_ex = npk.plane_strain_deformation_gradient(grad_u)

    F = ps.deformation_gradient(u)
    np.testing.assert_allclose(eval_at_quadrature(domain, F, degree), F_ex, atol=1e-11)
    np.testing.assert_allclose(
        eval_at_quadrature(domain, ps.green_lagrange_strain(F), degree),
        npk.green_lagrange_strain(F_ex),
        atol=1e-11,
    )


def test_deformation_gradient_rejects_non_2d_displacement():
    domain = make_domain(mesh.CellType.triangle, n=(2, 1))
    V = fem.functionspace(domain, ("Lagrange", 1))
    with pytest.raises(ValueError):
        ps.deformation_gradient(fem.Function(V))
