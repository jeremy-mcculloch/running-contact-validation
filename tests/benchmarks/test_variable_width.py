"""FE-01 benchmarks: variable effective-width quadrature and mass integration.

Analytic references on the rectangle Omega0 = [0, L] x [0, H]:

* constant width b0:          M = rho0 b0 L H
* linear width b0 -> b1:      M = rho0 H L (b0 + b1)/2
* quadratic test width        b = b0 (1 + k (X/L)^2):  M = rho0 b0 H L (1 + k/3)

Units: rho0 [kg/m^3] * b [m] * dA [m^2] -> kg.
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from running_contact_validation.fe import plane_strain as ps
from running_contact_validation.fe.variable_width import WidthField, assemble_total
from running_contact_validation.fe.width import ConstantWidth, LinearWidth, WidthProfile

L, H = 0.28, 0.03  # m
RHO0 = 220.0  # kg/m^3, illustrative foam density (not a material parameter choice)
CELL_TYPES = [mesh.CellType.triangle, mesh.CellType.quadrilateral]
RTOL = 1e-12


def make_domain(cell_type, n=(8, 3)):
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, H])],
        list(n),
        cell_type=cell_type,
    )


class QuadraticTestWidth(WidthProfile):
    """Test-only nonlinear profile; not a proposed production width."""

    def __init__(self, b0, k):
        self.b0, self.k = b0, k

    @property
    def X_range(self):
        return (0.0, L)

    def _formula(self, X):
        return self.b0 * (1 + self.k * (X / L) ** 2)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("n", [(1, 1), (8, 3), (32, 6)])
def test_constant_width_mass(cell_type, n):
    b0 = 0.095
    field = WidthField(make_domain(cell_type, n), ConstantWidth(b0), degree=2)
    assert field.total_mass(RHO0) == pytest.approx(RHO0 * b0 * L * H, rel=RTOL)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("n", [(1, 1), (8, 3), (32, 6)])
def test_linear_width_mass(cell_type, n):
    b0, b1 = 0.080, 0.110
    field = WidthField(make_domain(cell_type, n), LinearWidth(0.0, b0, L, b1), degree=2)
    expected = RHO0 * H * L * (b0 + b1) / 2
    assert field.total_mass(RHO0) == pytest.approx(expected, rel=RTOL)
    # first moment: integral b X dA = H (b0 L^2/2 + slope L^3/3), quadratic integrand
    X = ufl.SpatialCoordinate(field.domain)[0]
    slope = (b1 - b0) / L
    assert field.integrate(X) == pytest.approx(
        H * (b0 * L**2 / 2 + slope * L**3 / 3), rel=RTOL
    )


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_width_is_sampled_at_quadrature_points_not_interpolated(cell_type):
    """A nonlinear b(X) is integrated exactly (degree-2 integrand, degree-2
    rule) on a coarse mesh; nodal P1 interpolation of b would not be."""
    b0, k = 0.09, 0.8
    profile = QuadraticTestWidth(b0, k)
    domain = make_domain(cell_type, (2, 1))
    field = WidthField(domain, profile, degree=2)
    exact = RHO0 * b0 * H * L * (1 + k / 3)
    assert field.total_mass(RHO0) == pytest.approx(exact, rel=RTOL)

    # stored values are b evaluated at the physical quadrature points
    np.testing.assert_allclose(field.values, profile(field.points[..., 0]), rtol=1e-15)

    P1 = fem.functionspace(domain, ("Lagrange", 1))
    b_nodal = fem.Function(P1)
    b_nodal.interpolate(lambda x: profile(x[0]))
    nodal_mass = assemble_total(RHO0 * b_nodal * field.dx)
    assert abs(nodal_mass - exact) / exact > 1e-3


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize(
    "profile",
    [
        ConstantWidth(0.1),
        LinearWidth(0.0, 0.08, L, 0.11),
        QuadraticTestWidth(0.09, 0.8),
    ],
)
def test_quadrature_field_matches_symbolic_width(cell_type, profile):
    domain = make_domain(cell_type)
    field = WidthField(domain, profile, degree=3)
    b_ufl = profile(ufl.SpatialCoordinate(domain)[0])
    symbolic = assemble_total(RHO0 * b_ufl * field.dx)
    assert field.total_mass(RHO0) == pytest.approx(symbolic, rel=RTOL)


def test_doubling_width_doubles_mass_and_is_applied_once():
    domain = make_domain(mesh.CellType.triangle)
    lin = LinearWidth(0.0, 0.08, L, 0.11)
    lin2 = LinearWidth(0.0, 0.16, L, 0.22)
    m1 = WidthField(domain, lin, 2).total_mass(RHO0)
    m2 = WidthField(domain, lin2, 2).total_mass(RHO0)
    assert m2 == pytest.approx(2 * m1, rel=RTOL)
    # Unit-width (2D, per-width) mass is rho0 * area [kg/m]; b0 = 1 m recovers it
    unit = WidthField(domain, ConstantWidth(1.0), 2).total_mass(RHO0)
    assert unit == pytest.approx(RHO0 * L * H, rel=RTOL)


def test_mass_units_with_realistic_values():
    """~0.28 m x 0.03 m section, ~0.095 m wide, 220 kg/m^3 -> ~0.18 kg."""
    field = WidthField(
        make_domain(mesh.CellType.quadrilateral), ConstantWidth(0.095), 2
    )
    assert field.total_mass(RHO0) == pytest.approx(0.175560, rel=1e-6)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_width_weighted_kinematic_integrand(cell_type):
    """integral b(X) E_xx dA under uniform extension = E_xx * integral b dA,
    exercising the integrand pattern used by future constitutive forms."""
    lx = 1.2
    domain = make_domain(cell_type)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    u.interpolate(lambda x: np.vstack([(lx - 1) * x[0], 0 * x[1]]))
    E = ps.green_lagrange_strain(ps.deformation_gradient(u))
    field = WidthField(domain, LinearWidth(0.0, 0.08, L, 0.11), degree=2)
    area_b = H * L * (0.08 + 0.11) / 2
    assert field.integrate(E[0, 0]) == pytest.approx((lx**2 - 1) / 2 * area_b, rel=RTOL)
    assert abs(field.integrate(E[2, 2])) < 1e-16


def test_profile_not_covering_mesh_is_rejected():
    domain = make_domain(mesh.CellType.triangle)
    with pytest.raises(ValueError, match="outside"):
        WidthField(domain, LinearWidth(0.0, 0.08, 0.5 * L, 0.11), degree=2)


def test_nonpositive_width_at_quadrature_point_is_rejected():
    domain = make_domain(mesh.CellType.triangle)
    with pytest.raises(ValueError, match="> 0"):
        WidthField(domain, QuadraticTestWidth(0.09, -1.5), degree=2)
