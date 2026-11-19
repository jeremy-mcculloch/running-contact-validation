"""Print the versions of the FEniCSx stack in a fixed format (FE-00)."""

import platform

import basix
import dolfinx
import ffcx
import mpi4py
import numpy
import petsc4py
import ufl
from mpi4py import MPI
from petsc4py import PETSc


def main() -> None:
    petsc = ".".join(str(v) for v in PETSc.Sys.getVersion())
    rows = [
        ("python", platform.python_version()),
        ("dolfinx", dolfinx.__version__),
        ("ufl", ufl.__version__),
        ("basix", basix.__version__),
        ("ffcx", ffcx.__version__),
        ("petsc", petsc),
        ("petsc4py", petsc4py.__version__),
        ("petsc_scalar", numpy.dtype(PETSc.ScalarType).name),
        ("mpi4py", mpi4py.__version__),
        ("mpi_library", MPI.Get_library_version().splitlines()[0].strip()),
        ("numpy", numpy.__version__),
    ]
    for name, version in rows:
        print(f"{name:<13} {version}")


if __name__ == "__main__":
    main()
