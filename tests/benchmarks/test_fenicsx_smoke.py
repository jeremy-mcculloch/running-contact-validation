from mpi4py import MPI
from dolfinx import mesh
import numpy as np


def test_fenicsx_mesh():
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        8,
        8,
        cell_type=mesh.CellType.triangle,
    )

    assert domain.topology.dim == 2
    assert domain.geometry.dim == 2

    coordinates = domain.geometry.x

    assert np.all(np.isfinite(coordinates))
    assert len(coordinates) > 0
