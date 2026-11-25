# running-contact-validation

Predicts running ground-reaction forces from prescribed joint kinematics and independently characterized shoe mechanics. See `docs/scientific_specification.md` (authoritative modeling decisions) and `docs/AGENTS.md` (contributor/agent rules).

## Development workflow

Development is issue-driven: each independently reviewable task is tracked as one GitHub issue, implemented on one feature branch, and submitted as one pull request targeting `main`.

- **Eligibility.** Issues labeled `agent-ready` by a maintainer are eligible for automated implementation by a coding agent. Issues without this label are not picked up automatically.
- **Feature branches.** Coding agents start from the latest `main` and work on a dedicated feature branch for the issue. They never push directly to `main`; the pull request references the issue (e.g. `Closes #<issue-number>`) and records the verification evidence and known limitations.
- **Continuous integration.** GitHub Actions (`.github/workflows/tests.yml`) runs on every pull request to `main`: the `Python tests` job runs Ruff and the unit tests, and the `FEniCSx tests` job runs the FE benchmarks in the pinned FEniCSx container (see below).
- **Human approval.** Scientific decisions (changes to approved physical models, equations, assumptions, verification tolerances, or selection of methods the specification marks unresolved) require explicit approval from the human researcher. Agents may propose such changes in a pull request but must not treat them as approved. Pull requests are reviewed and merged only by a human; agents do not merge.

Agent conventions and scientific boundaries are defined in `docs/AGENTS.md` and `CLAUDE.md`; approved physics is defined in `docs/scientific_specification.md`.

## Quick checks (no FEniCSx required)

Requires [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

`uv run pytest` runs the unit tests. The FEniCSx benchmarks in `tests/benchmarks/` are not collected when DOLFINx is not installed; these commands do **not** exercise FE code.

## FEniCSx environment (Docker)

FE code and benchmarks run in a container built from a digest-pinned `dolfinx/dolfinx` image (see `Dockerfile`). The repository is bind-mounted at `/workspace`, and `src/` is on `PYTHONPATH`.

Build the image:

```bash
docker compose build fe
```

Print the installed FEniCSx/PETSc versions:

```bash
docker compose run --rm fe python3 scripts/report_fe_versions.py
```

Run the FE benchmarks:

```bash
docker compose run --rm fe python3 -m pytest tests/benchmarks -v
```

Open an interactive shell in the container:

```bash
docker compose run --rm fe
```

Inside the container `RCV_REQUIRE_FENICSX=1` is set, so the benchmarks fail rather than skip if DOLFINx cannot be imported.

GitHub Actions (`.github/workflows/tests.yml`) runs the same build, version report and benchmark commands in the `FEniCSx tests` job, and the quick checks in the `Python tests` job.

## Pinned FE environment

| Component | Version |
|---|---|
| Base image | `dolfinx/dolfinx:stable@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8` |
| Python (container) | 3.12.3 |
| DOLFINx | 0.11.0.post0 |
| UFL | 2026.1.0 |
| Basix | 0.11.0 |
| FFCx | 0.11.0 |
| PETSc / petsc4py | 3.25.1 / 3.25.1 (real, float64; `PETSC_ARCH=linux-gnu-real64-32`) |
| mpi4py / MPI | 4.1.2 / MPICH 5.0.1 |
| NumPy (container) | 2.4.6 |

These are the versions printed by `scripts/report_fe_versions.py`, which the "Report FE versions" CI step also runs. To change the image, update the digest in `Dockerfile`, re-run the FE tests, and update this table.

## FE constitutive model: one-term hyperfoam (FE-02)

`fe/hyperfoam.py` (NumPy reference, no FEniCSx) and `fe/hyperfoam_ufl.py` (UFL) implement the approved one-term compressible Ogden/Storakers hyperfoam of `docs/scientific_specification.md` Section 24.2 (Abaqus convention, full principal stretches):

    W = (2 mu/alpha^2) [ l1^alpha + l2^alpha + l3^alpha - 3 + (J^(-alpha beta) - 1)/beta ],   J > 0

- Units: `mu` [Pa] (initial shear modulus), `alpha` [-], `beta` [-]; `W` in J/m^3 of reference volume. Small-strain bulk modulus `K0 = 2 mu (beta + 1/3)`, `lambda0 = 2 mu beta`, `nu0 = beta/(1 + 2 beta)`.
- Admissible parameters (checked by `HyperfoamParameters`): finite, `mu > 0`, `alpha != 0`, `beta > -1/3`. These are necessary small-strain conditions only; check stability over the strain range reached for each identified parameter set. For `beta < 0`, `W >= 0` is not guaranteed at finite strain.
- `beta = 0` is evaluated through the continuous limit `-alpha ln J`.
- Plane strain: the full 3D energy is evaluated on the embedded `F` with `F33 = 1`, so the transverse stress `P33` is retained (generally nonzero). The UFL functions require this block structure.
- `P = dW/dF` and the tangent come from UFL automatic differentiation; the residual and Jacobian are derivatives of `Pi = sum_r int b(X) W_r dA` (`stored_energy`), with parameters per foam-region cell tag (`FoamMaterialMap`).
- Call `check_admissible_deformation(u, degree)` before assembling: `J <= 0` raises `InadmissibleDeformationError`.

No viscous stress yet (FE-03). Coefficients in tests are synthetic and illustrative.
