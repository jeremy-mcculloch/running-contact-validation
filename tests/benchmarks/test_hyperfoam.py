"""FE-02 benchmarks: UFL one-term Ogden/Storakers hyperfoam (Section 24.2).

Independent references:

* the NumPy implementation (:mod:`running_contact_validation.fe.hyperfoam`),
  which evaluates the 3D energy through ``eigh`` of ``C`` and the stress via
  the closed-form matrix function ``S = (2mu/a)[C^((a-2)/2) - J^(-ab) C^-1]``,
  i.e. without the in-plane symmetric-sum/series construction used in UFL;
* central finite differences of energy (for stress) and of stress (for the
  tangent);
* closed-form homogeneous plane-strain solutions.

All coefficients are SYNTHETIC, ILLUSTRATIVE values; they are not identified
foam properties. Units: m, Pa, J.
"""

import math
import time

import numpy as np
import pytest
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

from running_contact_validation.fe import hyperfoam as nph
from running_contact_validation.fe import hyperfoam_ufl as hf
from running_contact_validation.fe import plane_strain as ps
from running_contact_validation.fe.hyperfoam import (
    FoamMaterialMap,
    HyperfoamParameters,
    InadmissibleDeformationError,
)
from running_contact_validation.fe.variable_width import WidthField, assemble_total
from running_contact_validation.fe.width import ConstantWidth, LinearWidth

L, H = 0.28, 0.03  # m, shoe-like reference section
B0 = 0.09  # m, constant synthetic width
QDEG = 4

PARAMS = {
    "foam_a": HyperfoamParameters(mu=0.8e6, alpha=8.0, beta=0.15),
    "beta_zero": HyperfoamParameters(mu=0.5e6, alpha=4.0, beta=0.0),
    "neg_alpha": HyperfoamParameters(mu=0.3e6, alpha=-3.0, beta=0.4),
    "neg_beta": HyperfoamParameters(mu=1.2e6, alpha=2.5, beta=-0.2),
    "stiff_alpha": HyperfoamParameters(mu=0.2e6, alpha=20.0, beta=0.05),
}


def embed(H2):
    F = np.eye(3)
    F[:2, :2] += np.asarray(H2, float)
    return F


def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# In-plane displacement gradients. "near_iso_*" straddle the series switch
# r = D/s^2 = 1e-3 of the in-plane symmetric sum; "identity_eps" is deep
# inside the series branch.
def _near_iso(r):
    # F = diag(1+d, 1-d) -> s = 1+d^2, D = 4d^2 -> r ~ 4 d^2
    d = math.sqrt(r) / 2
    return np.diag([d, -d])


STATES = {
    "compress_0.25": np.diag([-0.75, 0.0]),
    "compress_biax": np.diag([-0.5, -0.4]),
    "extend_1.6": np.diag([0.6, 0.0]),
    "shear_0.8": np.array([[0.0, 0.8], [0.0, 0.0]]),
    "compress_shear": np.array([[-0.1, 0.5], [0.05, -0.6]]),
    "isotropic_0.6": np.diag([-0.4, -0.4]),
    "identity_eps": np.array([[1e-7, 2e-8], [-3e-8, 4e-8]]),
    "near_iso_below": _near_iso(0.98e-3),
    "near_iso_above": _near_iso(1.02e-3),
}

# components of F that preserve the plane-strain block structure
BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]


def one_cell_domain():
    return mesh.create_unit_square(MPI.COMM_SELF, 1, 1)


class ConstantF:
    """Evaluate UFL expressions of a constant 3x3 F (block structure) at one point."""

    def __init__(self, params):
        self.domain = one_cell_domain()
        self.Fc = fem.Constant(self.domain, np.eye(3))
        self.params = params
        self.W = hf.strain_energy_density(self.Fc, params)
        self.P = hf.first_piola_kirchhoff_stress(self.Fc, params)
        self.A = hf.material_tangent(self.Fc, params)
        pt = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        self._exprs = {k: fem.Expression(getattr(self, k), pt) for k in ("W", "P", "A")}

    def __call__(self, name, F):
        self.Fc.value[:] = F
        val = self._exprs[name].eval(self.domain, np.array([0], dtype=np.int32))
        return val.reshape(ufl.shape(getattr(self, name)) or ())


@pytest.fixture(scope="module")
def evaluators():
    return {k: ConstantF(p) for k, p in PARAMS.items()}


# ------------------------------------------------ pointwise constitutive checks


@pytest.mark.parametrize("name", list(PARAMS))
def test_reference_state(evaluators, name):
    ev = evaluators[name]
    assert abs(float(ev("W", np.eye(3)))) < 1e-9
    np.testing.assert_allclose(ev("P", np.eye(3)), 0.0, atol=1e-9)
    # tangent at reference = linear isotropic elasticity with lambda0 = 2 mu beta
    p = PARAMS[name]
    A = ev("A", np.eye(3))
    I = np.eye(3)
    A_lin = p.lame_lambda * np.einsum("ij,kl->ijkl", I, I) + p.mu * (
        np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I)
    )
    # compare the plane-strain block only: out-of-plane shear tangent entries
    # (e.g. A_1313) are not used by the plane-strain residual and are not
    # represented (documented in hyperfoam_ufl)
    for i, j in BLOCK:
        for k, l in BLOCK:
            assert A[i, j, k, l] == pytest.approx(A_lin[i, j, k, l], abs=1e-8 * p.mu)


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("state", list(STATES))
def test_ufl_energy_and_stress_match_numpy(evaluators, name, state):
    ev, p = evaluators[name], PARAMS[name]
    F = embed(STATES[state])
    W_ref = nph.strain_energy_density(F, p)
    assert float(ev("W", F)) == pytest.approx(W_ref, rel=1e-11, abs=1e-12 * p.mu)
    P_ref = nph.first_piola_kirchhoff_stress(F, p)
    scale = max(np.abs(P_ref).max(), 1e-6 * p.mu)
    np.testing.assert_allclose(ev("P", F), P_ref, atol=1e-10 * scale + 1e-12 * p.mu)


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("state", list(STATES))
def test_ufl_stress_matches_fd_of_energy(evaluators, name, state):
    """P = dW/dF against central differences of the UFL and NumPy energies."""
    ev, p = evaluators[name], PARAMS[name]
    F = embed(STATES[state])
    P = ev("P", F)
    h = 1e-6
    scale = max(np.abs(P).max(), 1e-3 * p.mu)
    for i, j in BLOCK:
        dF = np.zeros((3, 3))
        dF[i, j] = h
        fd_ufl = (float(ev("W", F + dF)) - float(ev("W", F - dF))) / (2 * h)
        fd_np = (
            nph.strain_energy_density(F + dF, p) - nph.strain_energy_density(F - dF, p)
        ) / (2 * h)
        assert P[i, j] == pytest.approx(fd_ufl, abs=2e-6 * scale), (i, j)
        assert P[i, j] == pytest.approx(fd_np, abs=2e-6 * scale), (i, j)
    # out-of-plane shear stresses vanish on the plane-strain manifold
    for i, j in [(0, 2), (2, 0), (1, 2), (2, 1)]:
        assert abs(P[i, j]) < 1e-12 * scale


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("state", list(STATES))
def test_ufl_tangent_matches_fd_of_stress(evaluators, name, state):
    """A = dP/dF (UFL AD) against central differences of the independent NumPy
    stress and of the UFL stress, over all plane-strain components."""
    ev, p = evaluators[name], PARAMS[name]
    F = embed(STATES[state])
    A = ev("A", F)
    h = 1e-6
    scale = max(np.abs(A).max(), p.mu)
    for k, l in BLOCK:
        dF = np.zeros((3, 3))
        dF[k, l] = h
        fd_np = (
            nph.first_piola_kirchhoff_stress(F + dF, p)
            - nph.first_piola_kirchhoff_stress(F - dF, p)
        ) / (2 * h)
        fd_ufl = (ev("P", F + dF) - ev("P", F - dF)) / (2 * h)
        for i, j in BLOCK:
            assert A[i, j, k, l] == pytest.approx(fd_np[i, j], abs=1e-6 * scale), (
                i,
                j,
                k,
                l,
            )
            assert A[i, j, k, l] == pytest.approx(fd_ufl[i, j], abs=1e-6 * scale), (
                i,
                j,
                k,
                l,
            )
    # major symmetry (hyperelastic)
    Ab = np.array([[A[i, j, k, l] for k, l in BLOCK] for i, j in BLOCK])
    np.testing.assert_allclose(Ab, Ab.T, atol=1e-10 * scale)


@pytest.mark.parametrize("name", ["foam_a", "neg_alpha", "stiff_alpha"])
def test_stretch_series_switch_is_continuous(evaluators, name):
    """W, P and A are continuous across r = D/s^2 = 1e-3 (series <-> closed form)."""
    ev, p = evaluators[name], PARAMS[name]
    for r in [0.999e-3, 1.0e-3, 1.001e-3]:
        F = embed(_near_iso(r) + np.diag([-0.3, -0.3]))
        assert float(ev("W", F)) == pytest.approx(
            nph.strain_energy_density(F, p), rel=1e-11
        )
        np.testing.assert_allclose(
            ev("P", F),
            nph.first_piola_kirchhoff_stress(F, p),
            rtol=1e-9,
            atol=1e-9 * p.mu,
        )
    # states 1e-9 apart in r on either side of the switch: the tangent jump
    # must be at round-off level (the true change in A is ~1e-10 relative)
    lo = embed(_near_iso(1e-3 * (1 - 1e-9)) + np.diag([-0.3, -0.3]))
    hi = embed(_near_iso(1e-3 * (1 + 1e-9)) + np.diag([-0.3, -0.3]))
    A_lo, A_hi = ev("A", lo), ev("A", hi)
    np.testing.assert_allclose(A_lo, A_hi, atol=1e-8 * np.abs(A_lo).max())


@pytest.mark.parametrize("alpha", [8.0, -3.0])
def test_beta_zero_limit_continuous_in_ufl(alpha):
    """Volumetric term across beta = 0 and the series threshold |beta| = 1e-4."""
    F = embed(STATES["compress_shear"])
    for beta in [0.0, 1e-12, -1e-12, 0.99e-4, 1.01e-4, -0.99e-4, -1.01e-4, 1e-3]:
        p = HyperfoamParameters(mu=1e6, alpha=alpha, beta=beta)
        ev = ConstantF(p)
        assert float(ev("W", F)) == pytest.approx(
            nph.strain_energy_density(F, p), rel=1e-12
        )
        np.testing.assert_allclose(
            ev("P", F), nph.first_piola_kirchhoff_stress(F, p), rtol=1e-10, atol=1e-6
        )
    W0 = float(ConstantF(HyperfoamParameters(1e6, alpha, 0.0))("W", F))
    W1 = float(ConstantF(HyperfoamParameters(1e6, alpha, 1e-9))("W", F))
    assert W1 == pytest.approx(W0, rel=1e-7)


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("state", ["compress_shear", "extend_1.6", "shear_0.8"])
def test_frame_indifference_ufl(evaluators, name, state):
    """In-plane superposed rotations Q (sagittal plane): W(QF) = W(F), P(QF) = Q P(F)."""
    ev, p = evaluators[name], PARAMS[name]
    F = embed(STATES[state])
    W, P = float(ev("W", F)), ev("P", F)
    for theta in [0.4, math.pi / 2, 2.8]:
        Q = np.eye(3)
        Q[:2, :2] = rot2(theta)
        assert float(ev("W", Q @ F)) == pytest.approx(W, rel=1e-11, abs=1e-10 * p.mu)
        np.testing.assert_allclose(
            ev("P", Q @ F), Q @ P, atol=1e-9 * max(np.abs(P).max(), p.mu)
        )


def test_plane_strain_transverse_stress_nonzero(evaluators):
    p = PARAMS["foam_a"]
    lam = 0.4
    F = embed(np.diag([lam - 1, 0.0]))
    P = evaluators["foam_a"]("P", F)
    expected = 2 * p.mu / p.alpha * (1 - lam ** (-p.alpha * p.beta))
    assert P[2, 2] == pytest.approx(expected, rel=1e-12)
    assert P[2, 2] < -0.1 * p.mu


@pytest.mark.parametrize(
    "state, lam_range",
    [("uniaxial", (0.2, 2.0)), ("shear", (0.0, 1.5))],
)
def test_strong_ellipticity_over_documented_range(evaluators, state, lam_range):
    """Diagnostic for the documented deformation range of the synthetic 'foam_a'
    set: the in-plane acoustic tensor Q_ik = A_iJkL N_J N_L is positive definite
    for all sampled directions N. Not a guarantee for other parameter sets."""
    ev = evaluators["foam_a"]
    thetas = np.linspace(0, np.pi, 37)
    for t in np.linspace(*lam_range, 25):
        H2 = (
            np.diag([t - 1, 0.0])
            if state == "uniaxial"
            else np.array([[0, t], [0, 0.0]])
        )
        A = ev("A", embed(H2))[:2, :2, :2, :2]
        for th in thetas:
            N = np.array([math.cos(th), math.sin(th)])
            Qac = np.einsum("iJkL,J,L->ik", A, N, N)
            assert np.linalg.eigvalsh(0.5 * (Qac + Qac.T)).min() > 0


# ---------------------------------------------------- assembled FE forms


def make_domain(cell_type, n=(10, 3), comm=MPI.COMM_WORLD):
    return mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([L, H])], list(n), cell_type=cell_type
    )


CELL_TYPES = [mesh.CellType.triangle, mesh.CellType.quadrilateral]


def nonhomogeneous_u(V):
    u = fem.Function(V)
    u.interpolate(
        lambda x: np.vstack(
            [
                -0.15 * x[0] + 2.0 * x[1] ** 2 + 0.3 * x[1],
                -6.0 * x[1] ** 2 * (1 + x[0]) + 0.05 * x[0],
            ]
        )
    )
    return u


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_homogeneous_energy_integral(cell_type):
    """Affine u: Pi = W(F) int b dA exactly for linear b(X)."""
    domain = make_domain(cell_type)
    width = WidthField(domain, LinearWidth(0.0, 0.07, L, 0.11), QDEG)
    p = PARAMS["foam_a"]
    materials = FoamMaterialMap({0: p})
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    G = STATES["compress_shear"]
    u = fem.Function(V)
    u.interpolate(lambda x: G @ x[:2])
    Pi = assemble_total(hf.stored_energy(u, materials, width))
    area_b = H * L * (0.07 + 0.11) / 2
    assert Pi == pytest.approx(
        nph.strain_energy_density(embed(G), p) * area_b, rel=1e-11
    )


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_residual_energy_and_stress_forms_agree_and_match_fd(cell_type):
    """R = d Pi/du (AD) equals int b P:dF(v); both match FD of the assembled energy;
    K = dR/du matches FD of the assembled residual."""
    if MPI.COMM_WORLD.size > 1:
        pytest.skip("matrix-vector FD check written for serial runs")
    domain = make_domain(cell_type, n=(6, 2))
    width = WidthField(domain, LinearWidth(0.0, 0.07, L, 0.11), QDEG)
    materials = FoamMaterialMap({0: PARAMS["foam_a"]})
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = nonhomogeneous_u(V)
    jmin = hf.check_admissible_deformation(u, QDEG)
    assert 0.2 < jmin < 1.0
    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    Pi = hf.stored_energy(u, materials, width)
    R_form = fem.form(ufl.derivative(Pi, u, v))
    R_alt_form = fem.form(hf.internal_virtual_work(u, v, materials, width))
    K_form = fem.form(ufl.derivative(ufl.derivative(Pi, u, v), u, du))

    def assemble_R(form=R_form):
        b = fem.assemble_vector(form)
        b.scatter_reverse(la.InsertMode.add)
        return b.array[:n_owned].copy()

    n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    R = assemble_R()
    np.testing.assert_allclose(assemble_R(R_alt_form), R, atol=1e-10 * np.abs(R).max())

    # directional FD of energy along random perturbation w
    rng = np.random.default_rng(11)
    w = rng.standard_normal(n_owned) * 1e-3
    u0 = u.x.array.copy()

    def set_u(arr_owned):
        u.x.array[:n_owned] = arr_owned
        u.x.scatter_forward()

    h = 1e-4
    vals = []
    for sgn in (+1, -1):
        set_u(u0[:n_owned] + sgn * h * w)
        vals.append(assemble_total(Pi))
    set_u(u0[:n_owned])
    dPi_fd = (vals[0] - vals[1]) / (2 * h)
    dPi = domain.comm.allreduce(float(R @ w), op=MPI.SUM)
    assert dPi == pytest.approx(dPi_fd, rel=1e-7)

    # Jacobian vs FD of residual: K w ~ (R(u+hw) - R(u-hw)) / 2h
    A = fem.assemble_matrix(K_form)
    A.scatter_reverse()
    Ad = A.to_scipy().tocsr()  # owned rows; columns include ghosts (zero in serial)
    Kw = Ad[:n_owned, :] @ np.concatenate([w, np.zeros(Ad.shape[1] - n_owned)])
    Rp, Rm = [], []
    for sgn, store in ((+1, Rp), (-1, Rm)):
        set_u(u0[:n_owned] + sgn * h * w)
        store.append(assemble_R())
    set_u(u0[:n_owned])
    Kw_fd = (Rp[0] - Rm[0]) / (2 * h)
    np.testing.assert_allclose(Kw, Kw_fd, atol=1e-6 * np.abs(Kw).max())


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("name", ["foam_a", "neg_alpha", "beta_zero"])
@pytest.mark.parametrize("lam", [0.3, 0.7, 1.4])
def test_newton_uniaxial_plane_strain_lateral_free(cell_type, name, lam):
    """Homogeneous FE solution with Newton on the AD tangent.

    BCs: u_x = 0 at X = 0, u_x = (lam - 1) L at X = L, u_y = 0 at Y = 0; top free.
    Exact solution: lateral stretch from P22 = 0,
        lam2 = lam^(-beta/(1+beta)),  J = lam lam2,
        P11 = (2mu/a)(lam^(a-1) - J^(-ab)/lam),  P33 = (2mu/a)(1 - J^(-ab)).
    Reaction on X = L: f_x = P11 H B0 (constant width).
    """
    p = PARAMS[name]
    a, b, mu = p.alpha, p.beta, p.mu
    domain = make_domain(cell_type, n=(8, 2))
    width = WidthField(domain, ConstantWidth(B0, X_range=(0.0, L)), QDEG)
    materials = FoamMaterialMap({3: p})
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    fdim = domain.topology.dim - 1

    def dofs(marker, comp):
        facets = mesh.locate_entities_boundary(domain, fdim, marker)
        return fem.locate_dofs_topological(V.sub(comp), fdim, facets)

    left = dofs(lambda x: np.isclose(x[0], 0.0), 0)
    right = dofs(lambda x: np.isclose(x[0], L), 0)
    bottom = dofs(lambda x: np.isclose(x[1], 0.0), 1)
    ux_right = fem.Constant(domain, 0.0)
    bcs = [
        fem.dirichletbc(0.0, left, V.sub(0)),
        fem.dirichletbc(ux_right, right, V.sub(0)),
        fem.dirichletbc(0.0, bottom, V.sub(1)),
    ]
    Pi = hf.stored_energy(u, materials, width)
    R = ufl.derivative(Pi, u, v)
    problem = NonlinearProblem(
        R,
        u,
        bcs=bcs,
        petsc_options_prefix=f"fe02_{name}_{cell_type.name}_{str(lam).replace('.', '_')}_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-9,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    t0 = time.perf_counter()
    n_steps = 10  # load stepping in prescribed stretch
    its = []
    for k in range(1, n_steps + 1):
        lam_k = 1 + (lam - 1) * k / n_steps
        ux_right.value = (lam_k - 1) * L
        problem.solve()
        reason = problem.solver.getConvergedReason()
        assert reason > 0, f"SNES diverged (reason {reason}) at step {k}"
        its.append(problem.solver.getIterationNumber())
        hf.check_admissible_deformation(u, QDEG)
    runtime = time.perf_counter() - t0

    lam2 = lam ** (-b / (1 + b))
    J = lam * lam2
    P11 = 2 * mu / a * (lam ** (a - 1) - J ** (-a * b) / lam)
    P33 = 2 * mu / a * (1 - J ** (-a * b))
    F_ex = np.diag([lam, lam2, 1.0])
    # nodal displacement is the exact affine field
    x = V.tabulate_dof_coordinates()
    n = V.dofmap.index_map.size_local
    uu = u.x.array.reshape(-1, 2)[:n]
    np.testing.assert_allclose(uu[:, 0], (lam - 1) * x[:n, 0], atol=1e-10 * L)
    np.testing.assert_allclose(uu[:, 1], (lam2 - 1) * x[:n, 1], atol=1e-10 * L)
    # transverse (out-of-plane) stress resultant, nonzero unless beta = 0
    F = ps.deformation_gradient(u)
    P = hf.first_piola_kirchhoff_stress(F, p)
    area = assemble_total(width.function * width.dx)
    P33_avg = assemble_total(P[2, 2] * width.function * width.dx) / area
    assert P33_avg == pytest.approx(P33, rel=1e-8, abs=1e-6 * mu)
    if b != 0:
        assert abs(P33) > 1e-3 * mu
    # reaction on X = L via energy conjugacy: f_x = sum of residual at right ux dofs
    v_right = fem.Function(V)
    v_right.x.array[:] = 0.0
    fem.dirichletbc(1.0, right, V.sub(0)).set(v_right.x.array)
    reaction = assemble_total(ufl.replace(R, {v: v_right}))
    assert reaction == pytest.approx(P11 * H * B0, rel=1e-8)
    np.testing.assert_allclose(
        nph.first_piola_kirchhoff_stress(F_ex, p)[0, 0], P11, rtol=1e-12
    )
    print(
        f"[FE-02 Newton] {name} {cell_type.name} lam={lam}: P2, mesh 8x2, qdeg={QDEG}, "
        f"{n_steps} load steps, Newton its/step={its}, runtime {runtime:.2f}s, "
        f"P11={P11:.6e} Pa, P33={P33:.6e} Pa"
    )
    assert max(its) <= 8  # quadratic convergence of the consistent tangent


# --------------------------------------------------- multiple foam regions


def test_region_tagged_energy_two_materials():
    domain = make_domain(mesh.CellType.quadrilateral, n=(10, 4))
    width = WidthField(domain, ConstantWidth(B0, X_range=(0.0, L)), QDEG)
    tdim = domain.topology.dim
    n_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(n_local, dtype=np.int32)
    mid = mesh.compute_midpoints(domain, tdim, cells)
    tags_v = np.where(mid[:, 0] < L / 2, 1, 7).astype(np.int32)
    tags = mesh.meshtags(domain, tdim, cells, tags_v)
    pa, pb = PARAMS["foam_a"], PARAMS["neg_alpha"]
    materials = FoamMaterialMap({1: pa, 7: pb})
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    G = STATES["compress_biax"]
    u = fem.Function(V)
    u.interpolate(lambda x: G @ x[:2])
    Pi = assemble_total(hf.stored_energy(u, materials, width, tags))
    Fe = embed(G)
    half = B0 * H * L / 2
    expected = half * (
        nph.strain_energy_density(Fe, pa) + nph.strain_energy_density(Fe, pb)
    )
    assert Pi == pytest.approx(expected, rel=1e-11)

    with pytest.raises(ValueError, match=r"region\(s\) \[7\]"):
        hf.stored_energy(u, FoamMaterialMap({1: pa}), width, tags)
    partial = mesh.meshtags(domain, tdim, cells[tags_v == 1], tags_v[tags_v == 1])
    with pytest.raises(ValueError, match="no foam-region tag"):
        hf.stored_energy(u, materials, width, partial)
    with pytest.raises(ValueError, match="cell_tags are required"):
        hf.stored_energy(u, materials, width)


# ------------------------------------------------------ inadmissible states


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_inverted_deformation_rejected(cell_type):
    domain = make_domain(cell_type, n=(4, 2))
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    u.interpolate(lambda x: np.vstack([-1.5 * x[0], 0 * x[1]]))  # F11 = -0.5
    with pytest.raises(InadmissibleDeformationError, match="min J"):
        hf.check_admissible_deformation(u, QDEG)
    u.interpolate(lambda x: np.vstack([-1.0 * x[0], 0 * x[1]]))  # J = 0
    with pytest.raises(InadmissibleDeformationError):
        hf.check_admissible_deformation(u, QDEG)
    u.x.array[:] = np.nan
    with pytest.raises(InadmissibleDeformationError):
        hf.check_admissible_deformation(u, QDEG)
    u.interpolate(lambda x: np.vstack([-0.9 * x[0], 0 * x[1]]))  # J = 0.1, admissible
    assert hf.check_admissible_deformation(u, QDEG) == pytest.approx(0.1)


def test_rejects_non_3x3_F():
    domain = one_cell_domain()
    with pytest.raises(ValueError, match="3x3"):
        hf.strain_energy_density(fem.Constant(domain, np.eye(2)), PARAMS["foam_a"])
