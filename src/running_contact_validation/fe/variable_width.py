"""Variable effective-width quadrature for 2D sagittal FE models (requires FEniCSx).

Implements the reduced reference-volume integration convention of
``docs/scientific_specification.md`` Section 16.2: for a 2D integrand ``g``
(per unit out-of-plane width), the whole-shoe quantity is

    G = integral_Omega0 b(X) g dA,

with ``b`` evaluated **at each element quadrature point in reference
coordinates**. The width is applied exactly once, inside the integrand;
assembled totals must never be multiplied by a width again.

The width is stored in a quadrature-element function (one value per
quadrature point, no interpolation onto a nodal basis). That function must be
used with the matching measure from :func:`width_measure`, so the form is
integrated at the same points at which ``b`` was sampled.

Assumptions:

* The mesh is the undeformed reference configuration (its geometry is never
  moved) and its first coordinate axis is the reference heel-to-toe
  direction ``X``.
* DOLFINx treats a 2D problem as unit thickness; all out-of-plane weighting
  enters only through ``b``.
"""

from __future__ import annotations

import basix
import basix.ufl
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from running_contact_validation.fe.width import WidthProfile, check_width_values

__all__ = [
    "WidthField",
    "assemble_total",
    "reference_quadrature_points",
    "width_measure",
]


def width_measure(domain: mesh.Mesh, degree: int) -> ufl.Measure:
    """Cell measure with a fixed quadrature degree, matching a :class:`WidthField`."""
    return ufl.Measure(
        "dx",
        domain=domain,
        metadata={"quadrature_degree": degree, "quadrature_rule": "default"},
    )


def reference_quadrature_points(
    domain: mesh.Mesh, degree: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Physical (reference-configuration) coordinates of cell quadrature points.

    Returns ``(points, cells)`` where ``points`` has shape
    ``(num_cells, num_qp, gdim)`` for the local cells (owned and ghost)
    ``cells``, using the default basix rule of the given degree.
    """
    tdim = domain.topology.dim
    cell_map = domain.topology.index_map(tdim)
    cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    cell_type = basix.CellType[domain.topology.cell_name()]
    ref_points, _ = basix.make_quadrature(cell_type, degree)
    expr = fem.Expression(ufl.SpatialCoordinate(domain), ref_points)
    gdim = domain.geometry.dim
    values = expr.eval(domain, cells).reshape(len(cells), ref_points.shape[0], gdim)
    return values, cells


class WidthField:
    """Effective width ``b(X_q)`` sampled at the quadrature points of ``domain``.

    Parameters
    ----------
    domain:
        Reference-configuration 2D mesh (coordinates in metres).
    profile:
        Positive width profile ``b(X)`` [m] of the reference coordinate ``X``.
    degree:
        Quadrature degree. Forms using :attr:`function` must be integrated
        with :attr:`dx` (same degree and rule); a different rule would mean
        ``b`` is not evaluated at the integration points.
    """

    def __init__(self, domain: mesh.Mesh, profile: WidthProfile, degree: int) -> None:
        if domain.topology.dim != 2:
            raise ValueError("WidthField requires a 2D sagittal mesh")
        self.domain = domain
        self.profile = profile
        self.degree = int(degree)

        element = basix.ufl.quadrature_element(
            domain.topology.cell_name(), value_shape=(), degree=self.degree
        )
        self.space = fem.functionspace(domain, element)
        self.function = fem.Function(self.space, name="effective_width")

        points, cells = reference_quadrature_points(domain, self.degree)
        self.points = points
        # Evaluate at the reference heel-to-toe coordinate X. Validation is
        # collective so an invalid width on any rank raises on every rank
        # (avoids deadlock in later collective operations).
        error = ""
        try:
            b = profile(points[..., 0])
            check_width_values(b)
        except ValueError as exc:
            error = str(exc)
        errors = [e for e in domain.comm.allgather(error) if e]
        if errors:
            raise ValueError(errors[0])
        dofs = self.space.dofmap.list[cells]
        if dofs.shape != b.shape:
            raise RuntimeError(
                f"quadrature dof layout {dofs.shape} does not match points {b.shape}"
            )
        self.function.x.array[dofs.reshape(-1)] = b.reshape(-1)
        self.function.x.scatter_forward()
        self.dx = width_measure(domain, self.degree)

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Local width values at quadrature points, shape ``(num_cells, num_qp)``."""
        return self.function.x.array[self.space.dofmap.list[: self.points.shape[0]]]

    def total_mass(self, rho0: float | ufl.core.expr.Expr) -> float:
        """``M = integral rho_0 b(X) dA`` [kg] for reference density ``rho0`` [kg/m^3]."""
        return assemble_total(rho0 * self.function * self.dx)

    def integrate(self, integrand: ufl.core.expr.Expr) -> float:
        """``integral b(X) g dA`` for a scalar per-unit-width integrand ``g``."""
        return assemble_total(self.function * integrand * self.dx)


def assemble_total(form: ufl.Form) -> float:
    """Assemble a scalar form and sum over MPI ranks."""
    compiled = fem.form(form)
    local = fem.assemble_scalar(compiled)
    return float(compiled.mesh.comm.allreduce(local, op=MPI.SUM))
