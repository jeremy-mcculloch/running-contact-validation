"""FE-00 environment smoke tests (no physical model)."""

import numpy as np
from dolfinx import mesh
from mpi4py import MPI


def test_fe_stack_imports():
    import basix
    import dolfinx
    import ffcx
    import petsc4py
    import ufl
    from petsc4py import PETSc

    for module in (basix, dolfinx, ffcx, petsc4py, ufl):
        assert module.__version__
    assert PETSc.Sys.getVersion()


def test_package_importable_in_container():
    import running_contact_validation

    assert running_contact_validation is not None


def test_fenicsx_mesh():
    width = 8
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        width,
        width,
        cell_type=mesh.CellType.triangle,
    )

    assert domain.topology.dim == 2
    assert domain.geometry.dim == 2

    coordinates = domain.geometry.x
    assert np.all(np.isfinite(coordinates))

    # Global counts are partition-independent:
    # 2*width*width triangles and (width+1)^2 vertices.
    tdim = domain.topology.dim
    domain.topology.create_entities(0)
    num_cells = domain.topology.index_map(tdim).size_global
    num_vertices = domain.topology.index_map(0).size_global
    assert num_cells == 2 * width * width
    assert num_vertices == (width + 1) ** 2
