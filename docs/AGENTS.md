# Coding-agent instructions — running-contact-validation

## Mission and authority

Build a reproducible scientific software framework that predicts running ground-reaction forces (GRFs) from prescribed human joint kinematics and independently characterized shoe mechanics, then compares predictions with withheld experimental measurements.

**The sole scientific source of truth is `docs/scientific_specification.md`.** Read the relevant sections before changing physics code. The specification contains successive decision updates: later explicitly approved decisions supersede earlier provisional recommendations. If two approved statements conflict, stop and ask for clarification. Do not create or maintain `docs/decisions.md`; use Git commits and pull requests for change history. Propose specification changes in a pull request and obtain human approval before treating them as approved physics.

## Non-negotiable scientific boundaries

- Never use held-out measured GRFs, moments, or force-plate COP to drive, initialize, calibrate, optimize, or correct predictions. Do not use force-plate contact events to prescribe simulated touchdown or takeoff.
- Shoe constitutive parameters must come from independent material/component tests or explicitly documented literature sources. Keep calibration and validation datasets separate.
- No artificial forces, acceleration clipping, unexplained displacement corrections, or force rescaling to obtain a periodic solution or improve force agreement.
- Preserve consistent SI units, coordinate frames, mass accounting, action–reaction, and linear/angular momentum balance. Document any numerical regularization and demonstrate convergence.
- Report nonzero periodicity residuals and solver failures rather than hiding them.
- Do not claim that code has been tested unless the stated tests actually ran.

## Approved initial physical model

- Start in a 2D sagittal plane, with a design that permits later 3D extension. World `x` is forward, `y` is upward, and positive rotation is counterclockwise.
- Use a full-body rigid-segment human model with prescribed, smoothed joint kinematics. OpenSim supplies biomechanical geometry and inertial properties; no explicit muscle activation or internal biological dissipation model initially.
- Use COM-centered translation and angular-momentum balance. For the final distributed FE shoe, account for **total human-plus-shoe mass, momentum, and inertia consistently**; do not blindly apply the simplified six-state equations to a massive deforming shoe.
- Begin numerical verification with synthetic trajectories and simple Hunt–Crossley-like contact before integrating the detailed shoe.
- Final shoe: distributed dynamic FE deformation in **FEniCSx/DOLFINx with PETSc**; 2D continuum foam, multiple perfectly bonded foam regions, longitudinally variable effective width from measured midsole width, and initial plane-strain kinematics.
- Initial elastic material: published **one-term compressible Ogden hyperfoam**. Use the exact published potential and parameter conventions in the scientific specification; do not substitute an incompressible Ogden rubber law.
- Initial dissipation: **state-free Green–Lagrange strain-rate viscosity**, one viscosity coefficient per foam material. Verify frame objectivity and nonnegative dissipation; do not add fractional-memory or Prony states without approval.
- Carbon plate: beam elements with **whole-plate** stiffness, perfectly bonded to foam initially. Avoid multiplying whole-plate stiffness by effective foam width a second time.
- Foot–shoe attachment: rigid initially. Ground contact: regularized unilateral **penalty normal contact** on the exterior shoe boundary, without an explicit deformable outsole layer initially. Penalty stiffness is a numerical enforcement parameter, not a material property.
- Support stationary overground and moving-belt treadmill contact with correct relative velocities and frame-dependent periodicity conditions.
- Use least-squares periodic shooting initially, retaining and reporting residuals; distributed FE displacement/velocity states must be handled consistently. Single shooting, multiple shooting, and implicit integration details beyond approved choices remain subject to verification and review.

## Decisions that are NOT yet authorized for autonomous selection

Do not silently choose a production tangential friction law, detailed beam–solid coupling, time integrator, FE-state periodic initialization method, contact smoothing/penalty tolerances, or material coefficients. These require a documented proposal and human approval. Consult the latest unresolved-decision section of `docs/scientific_specification.md`.

## Repository layout and dependency boundaries

Python package: `src/running_contact_validation/` (do not create a second `running_contact` package).

- `fe/`: continuum and beam elements, materials, assembly, FE solvers.
- `biomechanics/`: OpenSim adapters and prescribed kinematics.
- `contact/`: normal and tangential contact interfaces.
- `dynamics/`: global momentum and coupled-body dynamics.
- `solvers/`: time integration and periodic-orbit methods.
- `tests/unit/`, `tests/integration/`, `tests/benchmarks/`: progressively stronger verification.
- `docs/scientific_specification.md`: approved model and open scientific questions.

Keep pure-Python core tests runnable without OpenSim or FEniCSx. Put FEniCSx-dependent tests in a documented, reproducible environment; pin compatible versions. Do not commit participant-identifiable data, large raw mocap files, or proprietary shoe geometry without explicit approval.

## Workflow for each agent task

1. Read this file and the relevant scientific-specification sections.
2. State the governing equations, assumptions, inputs/outputs, and acceptance tests before implementation.
3. Work on one narrow issue/branch; avoid unrelated refactors.
4. Add unit tests or analytical benchmarks with the implementation.
5. Run available checks, record exact commands and results, and identify any tests not run.
6. Submit a pull request with files changed, mathematical interpretation, verification evidence, limitations, and any approval requests.

Preferred lightweight checks from the repository root:

```bash
uv run pytest
uv run ruff check .
```

Run FE tests in the pinned FEniCSx environment when applicable. Never claim these lightweight commands validate FE code they did not execute.

## Required verification principles

- Analytic free-flight translation and conserved whole-body angular momentum.
- COM and foot kinematic reconstruction, coordinate invariance, and correct mass accounting.
- Plane-strain and variable-width patch/mass tests; hyperfoam stress and consistent tangent checked against energy derivatives.
- Frame objectivity, nonnegative viscous dissipation, and elastic/viscous limiting cases.
- Bonded-material displacement continuity and traction balance; beam bending and beam–solid moment transfer.
- Non-tensile normal contact, contact separation, penetration convergence, and moving-belt relative motion.
- Mesh/time-step convergence, momentum/energy accounting, and periodic-orbit recovery on a known synthetic benchmark.

## Initial issue sequence

- **FE-00:** Pin and verify reproducible FEniCSx/PETSc environment, with a smoke test.
- **FE-01:** Implement plane-strain kinematics and variable-width integration; pass analytic patch and mass tests.
- **FE-02:** Implement one-term published Ogden hyperfoam; test stress, tangent, and admissible deformations.
- **FE-03:** Implement state-free objective viscous stress; verify nonnegative dissipation.
- **FE-04:** Bonded multi-material foam benchmark.
- **FE-05:** Unilateral penalty normal-contact benchmark.

Implement these as separate, reviewable pull requests. Do not start the full coupled runner–shoe solver until the underlying mechanics are independently verified.
