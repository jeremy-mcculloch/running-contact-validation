"""Repeated in-plane stretches: divided-difference vs series evaluation (PR #20 review).

Checks, for both UFL methods (``series`` and ``divided_difference``):

* exactly repeated stretches (``F = I``, equal biaxial, equal biaxial with a
  superposed in-plane rotation);
* approach to a repeated state along 16 directions in ``(C11 - C22, C12)``
  with ``t = sqrt(D)/s`` from 1e-1 down to 1e-14;
* accuracy of W and P against the independent NumPy eigenvalue
  implementation, of the tangent against a fourth-order central difference of
  the NumPy stress, agreement between the two methods, and ``O(t)``
  continuity of W, P and A toward the repeated state;
* the UFL (Kahan) divided-difference coefficients against the NumPy
  (``log1p``/``expm1``) coefficients, which are verified against an
  80-digit reference in ``tests/unit/test_hyperfoam_divided_difference.py``.

Coefficients are SYNTHETIC, ILLUSTRATIVE values.
"""

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from running_contact_validation.fe import hyperfoam as nph
from running_contact_validation.fe import hyperfoam_ufl as hf
from running_contact_validation.fe.hyperfoam import HyperfoamParameters

PARAMS = {
    "foam_a": HyperfoamParameters(mu=0.8e6, alpha=8.0, beta=0.15),
    "beta_zero": HyperfoamParameters(mu=0.5e6, alpha=4.0, beta=0.0),
    "neg_alpha": HyperfoamParameters(mu=0.3e6, alpha=-3.0, beta=0.4),
    "alpha_two": HyperfoamParameters(mu=0.5e6, alpha=2.0, beta=0.2),
    "stiff_alpha": HyperfoamParameters(mu=0.2e6, alpha=20.0, beta=0.05),
    "frac_alpha": HyperfoamParameters(mu=0.6e6, alpha=1.37, beta=0.3),
    "neg_frac_alpha": HyperfoamParameters(mu=0.4e6, alpha=-2.71, beta=0.08),
    "small_frac_alpha": HyperfoamParameters(mu=0.6e6, alpha=0.5, beta=0.1),
}
BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
THETAS = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
T_SWEEP = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]


def embed(F2):
    F = np.eye(3)
    F[:2, :2] = F2
    return F


def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def F_with_C2(s, t, theta, R=None):
    """Embedded F = R sqrt(C2), C2 = s (I + t [[cos, sin], [sin, -cos]])."""
    C2 = s * (
        np.eye(2)
        + t
        * np.array(
            [[math.cos(theta), math.sin(theta)], [math.sin(theta), -math.cos(theta)]]
        )
    )
    R = np.eye(2) if R is None else R
    w, Q = np.linalg.eigh(C2)
    return embed(R @ (Q @ np.diag(np.sqrt(w)) @ Q.T))


def fd4_stress(F, p, h=1e-4):
    """Fourth-order central difference of the NumPy stress (BLOCK columns)."""
    A = np.zeros((3, 3, 3, 3))
    for k, l in BLOCK:
        E = np.zeros((3, 3))
        E[k, l] = h

        def P(x, E=E):
            return nph.first_piola_kirchhoff_stress(F + x * E, p)

        A[:, :, k, l] = (-P(2) + 8 * P(1) - 8 * P(-1) + P(-2)) / (12 * h)
    return A


class Evaluator:
    def __init__(self, params, method):
        self.domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
        self.Fc = fem.Constant(self.domain, np.eye(3))
        exprs = {
            "W": hf.strain_energy_density(self.Fc, params, method),
            "P": hf.first_piola_kirchhoff_stress(self.Fc, params, method),
            "A": hf.material_tangent(self.Fc, params, method),
        }
        pt = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        self.shapes = {k: ufl.shape(e) for k, e in exprs.items()}
        self.exprs = {k: fem.Expression(e, pt) for k, e in exprs.items()}

    def __call__(self, name, F):
        self.Fc.value[:] = F
        val = self.exprs[name].eval(self.domain, np.array([0], dtype=np.int32))
        return val.reshape(self.shapes[name] or ())


@pytest.fixture(scope="module")
def ev():
    return {(k, m): Evaluator(p, m) for k, p in PARAMS.items() for m in hf.METHODS}


def block(A):
    return np.array([[A[i, j, k, l] for k, l in BLOCK] for i, j in BLOCK])


def repeated_states():
    out = {"identity": np.eye(3)}
    for lam in (0.4, 0.8, 1.3):
        out[f"equibiax_{lam}"] = embed(lam * np.eye(2))
        out[f"equibiax_rot_{lam}"] = embed(rot2(0.7) @ (lam * np.eye(2)))
    return out


REPEATED = repeated_states()


@pytest.mark.parametrize("method", hf.METHODS)
@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("state", list(REPEATED))
def test_exactly_repeated_stretches(ev, method, name, state):
    p, F = PARAMS[name], REPEATED[state]
    e = ev[name, method]
    W, P, A = float(e("W", F)), e("P", F), e("A", F)
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(A))
    assert W == pytest.approx(
        nph.strain_energy_density(F, p), rel=1e-12, abs=1e-12 * p.mu
    )
    P_ref = nph.first_piola_kirchhoff_stress(F, p)
    sP = max(np.abs(P_ref).max(), p.mu)
    np.testing.assert_allclose(P, P_ref, atol=1e-12 * sP)
    A_ref = fd4_stress(F, p)
    sA = max(np.abs(block(A_ref)).max(), p.mu)
    np.testing.assert_allclose(block(A), block(A_ref), atol=1e-8 * sA)
    # major symmetry
    np.testing.assert_allclose(block(A), block(A).T, atol=1e-12 * sA)
    # the two methods agree to round-off at exactly repeated stretches
    other = "series" if method == "divided_difference" else "divided_difference"
    np.testing.assert_allclose(
        block(A), block(ev[name, other]("A", F)), atol=1e-11 * sA
    )


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("s", [0.36, 1.0, 1.69])
def test_directional_approach_to_repeated_state(ev, name, s):
    """Approach C2 = s I along 16 directions; W, P, A accurate and O(t)-continuous."""
    p = PARAMS[name]
    F0 = embed(math.sqrt(s) * np.eye(2))
    dd, se = ev[name, "divided_difference"], ev[name, "series"]
    W0, P0, A0 = float(dd("W", F0)), dd("P", F0), block(dd("A", F0))
    sP = max(np.abs(P0).max(), p.mu)
    sA = max(np.abs(A0).max(), p.mu)
    worst = {"P_np": 0.0, "A_series": 0.0, "A_fd": 0.0}
    for theta in THETAS:
        for t in T_SWEEP:
            F = F_with_C2(s, t, theta)
            W, P, A = float(dd("W", F)), dd("P", F), block(dd("A", F))
            # accuracy against the independent NumPy implementation
            W_ref = nph.strain_energy_density(F, p)
            assert W == pytest.approx(W_ref, rel=1e-12, abs=1e-12 * p.mu)
            P_ref = nph.first_piola_kirchhoff_stress(F, p)
            err = np.abs(P - P_ref).max() / sP
            worst["P_np"] = max(worst["P_np"], err)
            assert err < 1e-12, (theta, t)
            # agreement with the series method (AD of the series energy)
            err = np.abs(A - block(se("A", F))).max() / sA
            worst["A_series"] = max(worst["A_series"], err)
            assert err < 1e-10, (theta, t)
            # continuity: |X(t) - X(0)| = O(t) (smooth in C)
            c = 50.0 * max(1.0, abs(p.alpha)) ** 2
            assert abs(W - W0) <= c * t * p.mu + 1e-12 * p.mu
            assert np.abs(P - P0).max() <= c * t * sP + 1e-12 * sP
            assert np.abs(A - A0).max() <= c * t * sA + 1e-11 * sA
        # FD tangent check along each direction at a few magnitudes
        for t in (1e-2, 1e-6, 1e-12):
            F = F_with_C2(s, t, theta)
            err = np.abs(block(dd("A", F)) - block(fd4_stress(F, p))).max() / sA
            worst["A_fd"] = max(worst["A_fd"], err)
            assert err < 1e-8, (theta, t)
    print(f"[FE-02 repeated] {name} s={s}: worst rel. errors {worst}")


@pytest.mark.parametrize("name", ["foam_a", "neg_frac_alpha", "frac_alpha"])
def test_objectivity_and_p33_divided_difference(ev, name):
    p = PARAMS[name]
    dd = ev[name, "divided_difference"]
    for F in (
        F_with_C2(0.6, 1e-9, 1.1),
        F_with_C2(0.6, 0.3, 2.0),
        embed(0.7 * np.eye(2)),
    ):
        W, P = float(dd("W", F)), dd("P", F)
        for th in (0.3, 2.5):
            Q = embed(rot2(th))
            assert float(dd("W", Q @ F)) == pytest.approx(
                W, rel=1e-12, abs=1e-12 * p.mu
            )
            np.testing.assert_allclose(
                dd("P", Q @ F), Q @ P, atol=1e-11 * max(p.mu, np.abs(P).max())
            )
        J = np.linalg.det(F)
        P33 = 2 * p.mu / p.alpha * (1 - J ** (-p.alpha * p.beta))
        assert P[2, 2] == pytest.approx(P33, rel=1e-12, abs=1e-12 * p.mu)


@pytest.mark.parametrize("p_exp", [4.0, 1.0, 10.0, -1.5, 0.685, -1.355, 0.25])
def test_ufl_coefficients_match_numpy(p_exp):
    """Kahan ln(u)/(u-1), (w-1)/ln(w) in UFL vs log1p/expm1 in NumPy."""
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    Cc = fem.Constant(domain, np.eye(3))
    dd = hf._DividedDifference(Cc, p_exp)
    keys = ["g", "g_s", "g_D", "g_ss", "g_sD", "K"]
    expr = fem.Expression(
        ufl.as_vector([getattr(dd, k) for k in keys]), np.array([[0.3, 0.3]])
    )
    for s in (0.3, 1.0, 2.5):
        for t in [0.0, 1e-300, 1e-16, 1e-12, 1e-8, 1e-5, 1e-3, 0.1, 0.5]:
            for theta in (0.0, 1.0, 2.5):
                x, y = t * s * math.cos(theta), t * s * math.sin(theta)
                Cc.value[:] = np.array([[s + x, y, 0], [y, s - x, 0], [0, 0, 1.0]])
                got = expr.eval(domain, np.array([0], dtype=np.int32)).ravel()
                D = x * x + y * y
                ref = nph.inplane_stretch_sum_coefficients(s, D, p_exp)
                scale2 = (abs(p_exp * (p_exp - 1)) + abs(p_exp)) * s ** (p_exp - 2)
                for k, g in zip(keys, got):
                    r = float(ref[k])
                    tol = 1e-14 * (abs(r) if k != "K" else scale2) + 1e-14 * scale2 * (
                        k == "g_sD"
                    )
                    assert abs(g - r) <= tol + 1e-300, (k, s, t, theta, g, r)
