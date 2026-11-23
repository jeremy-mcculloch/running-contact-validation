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
