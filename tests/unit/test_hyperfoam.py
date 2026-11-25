"""FE-02 unit tests: NumPy one-term Ogden/Storakers hyperfoam (Section 24.2).

All coefficients below are SYNTHETIC, ILLUSTRATIVE values chosen to exercise
the formulas (positive/negative alpha, zero/positive/negative beta). They are
not identified foam properties.
"""

import math

import numpy as np
import pytest

from running_contact_validation.fe.hyperfoam import (
    FoamMaterialMap,
    HyperfoamParameters,
    InadmissibleDeformationError,
    cauchy_stress,
    check_admissible_jacobian,
    first_piola_kirchhoff_stress,
    principal_stretches,
    second_piola_kirchhoff_stress,
    strain_energy_density,
    volumetric_term,
)
from running_contact_validation.fe.kinematics import plane_strain_deformation_gradient

PARAMS = {
    "foam_a": HyperfoamParameters(mu=0.8e6, alpha=8.0, beta=0.15),
    "beta_zero": HyperfoamParameters(mu=0.5e6, alpha=4.0, beta=0.0),
    "neg_alpha": HyperfoamParameters(mu=0.3e6, alpha=-3.0, beta=0.4),
    "neg_beta": HyperfoamParameters(mu=1.2e6, alpha=2.5, beta=-0.2),
    "stiff_alpha": HyperfoamParameters(mu=0.2e6, alpha=20.0, beta=0.05),
}


def rot3(axis, theta):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def ps(H):
    return plane_strain_deformation_gradient(np.asarray(H, float))


STATES = {
    "compress_0.3": ps(np.diag([-0.7, 0.0])),
    "compress_biax": ps(np.diag([-0.5, -0.4])),
    "extend_1.6": ps(np.diag([0.6, 0.0])),
    "shear_0.8": ps([[0.0, 0.8], [0.0, 0.0]]),
    "compress_shear": ps([[-0.1, 0.5], [0.05, -0.6]]),
    "near_identity": ps([[1e-5, 2e-6], [-3e-6, 4e-6]]),
    "general_3d": np.array([[1.1, 0.3, -0.1], [0.05, 0.7, 0.2], [0.1, -0.15, 0.9]]),
}


def fd_stress(F, params, h=1e-6):
    P = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dF = np.zeros((3, 3))
            dF[i, j] = h
            P[i, j] = (
                strain_energy_density(F + dF, params)
                - strain_energy_density(F - dF, params)
            ) / (2 * h)
    return P


# ---------------------------------------------------------------- parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu": 0.0, "alpha": 2.0, "beta": 0.1},
        {"mu": -1.0, "alpha": 2.0, "beta": 0.1},
        {"mu": 1.0, "alpha": 0.0, "beta": 0.1},
        {"mu": 1.0, "alpha": 2.0, "beta": -1.0 / 3.0},
        {"mu": 1.0, "alpha": 2.0, "beta": -0.5},
        {"mu": math.nan, "alpha": 2.0, "beta": 0.1},
        {"mu": 1.0, "alpha": math.inf, "beta": 0.1},
    ],
)
def test_parameter_admissibility_rejects(kwargs):
    with pytest.raises(ValueError):
        HyperfoamParameters(**kwargs)


@pytest.mark.parametrize("bad", ["1.0", None, True])
def test_parameter_type_rejects(bad):
    with pytest.raises(TypeError):
        HyperfoamParameters(mu=bad, alpha=2.0, beta=0.1)


def test_derived_small_strain_moduli():
    p = HyperfoamParameters(mu=2.0e5, alpha=6.0, beta=0.25)
    assert p.shear_modulus == 2.0e5
    assert p.lame_lambda == pytest.approx(2 * 2.0e5 * 0.25)
    assert p.bulk_modulus == pytest.approx(2 * 2.0e5 * (0.25 + 1 / 3))
    assert p.poisson_ratio == pytest.approx(0.25 / 1.5)
    # beta = nu / (1 - 2 nu)
    nu = p.poisson_ratio
    assert nu / (1 - 2 * nu) == pytest.approx(0.25)
    # K = lambda + 2 mu / 3
    assert p.bulk_modulus == pytest.approx(p.lame_lambda + 2 * p.mu / 3)


def test_material_map():
    a, b = PARAMS["foam_a"], PARAMS["neg_alpha"]
    m = FoamMaterialMap({2: b, 1: a})
    assert list(m) == [1, 2]
    assert m[1] is a and m[2] is b and len(m) == 2
    with pytest.raises(KeyError, match="region 3"):
        m[3]
    with pytest.raises(ValueError):
        FoamMaterialMap({})
    with pytest.raises(ValueError):
        FoamMaterialMap({-1: a})
    with pytest.raises(TypeError):
        FoamMaterialMap({0: (1.0, 2.0, 0.1)})


# ------------------------------------------------------- reference state


@pytest.mark.parametrize("name", list(PARAMS))
def test_reference_state_energy_and_stress_vanish(name):
    p = PARAMS[name]
    I = np.eye(3)
    assert strain_energy_density(I, p) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(first_piola_kirchhoff_stress(I, p), 0.0, atol=1e-9)
    np.testing.assert_allclose(second_piola_kirchhoff_stress(I, p), 0.0, atol=1e-9)


# ----------------------------------------------- stress = dW/dF (FD check)


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("state", list(STATES))
def test_stress_matches_finite_difference_of_energy(name, state):
    p, F = PARAMS[name], STATES[state]
    P = first_piola_kirchhoff_stress(F, p)
    P_fd = fd_stress(F, p)
    scale = max(np.max(np.abs(P)), p.mu * 1e-3)
    np.testing.assert_allclose(P, P_fd, atol=1e-6 * scale, rtol=1e-6)


@pytest.mark.parametrize("name", list(PARAMS))
def test_random_states_energy_stress_consistency(name):
    p = PARAMS[name]
    rng = np.random.default_rng(42)
    for _ in range(20):
        F = np.eye(3) + 0.35 * rng.standard_normal((3, 3))
        if np.linalg.det(F) < 0.2:
            continue
        P = first_piola_kirchhoff_stress(F, p)
        scale = max(np.max(np.abs(P)), p.mu * 1e-3)
        np.testing.assert_allclose(P, fd_stress(F, p), atol=2e-6 * scale)


# ------------------------------------------------------ small strain


@pytest.mark.parametrize("name", list(PARAMS))
def test_small_strain_limit_is_linear_isotropic(name):
    """W ~ mu eps:eps + (lambda0/2)(tr eps)^2,  sigma ~ 2 mu eps + lambda0 tr(eps) I."""
    p = PARAMS[name]
    rng = np.random.default_rng(7)
    G = rng.standard_normal((3, 3))
    eps_size = 1e-5
    H = eps_size * G
    eps = 0.5 * (H + H.T)
    F = np.eye(3) + H
    W_lin = p.mu * np.sum(eps * eps) + 0.5 * p.lame_lambda * np.trace(eps) ** 2
    assert strain_energy_density(F, p) == pytest.approx(W_lin, rel=1e-3)
    sigma_lin = 2 * p.mu * eps + p.lame_lambda * np.trace(eps) * np.eye(3)
    np.testing.assert_allclose(
        cauchy_stress(F, p), sigma_lin, atol=1e-3 * np.max(np.abs(sigma_lin))
    )
    # Hydrostatic: pressure = -K dV/V
    e = 1e-6
    sigma = cauchy_stress(np.eye(3) * (1 + e), p)
    assert np.trace(sigma) / 3 == pytest.approx(p.bulk_modulus * 3 * e, rel=1e-4)


# --------------------------------------------- homogeneous analytic tests


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("lam", [0.2, 0.5, 0.9, 1.3, 2.0])
def test_plane_strain_uniaxial_analytic(name, lam):
    """F = diag(lam, 1, 1):
    W   = (2mu/a^2)[lam^a - 1 + (lam^(-ab) - 1)/b]
    P11 = (2mu/a)(lam^(a-1) - lam^(-ab-1)),  P22 = P33 = (2mu/a)(1 - lam^(-ab))."""
    p = PARAMS[name]
    a, b, mu = p.alpha, p.beta, p.mu
    F = ps(np.diag([lam - 1.0, 0.0]))
    vol = -a * math.log(lam) if b == 0 else (lam ** (-a * b) - 1) / b
    assert strain_energy_density(F, p) == pytest.approx(
        2 * mu / a**2 * (lam**a - 1 + vol), rel=1e-12, abs=1e-6
    )
    P = first_piola_kirchhoff_stress(F, p)
    lat = 2 * mu / a * (1 - lam ** (-a * b))
    expected = np.diag([2 * mu / a * (lam ** (a - 1) - lam ** (-a * b - 1)), lat, lat])
    np.testing.assert_allclose(P, expected, rtol=1e-11, atol=1e-6)


def test_plane_strain_transverse_stress_nonzero_and_beta_zero_limit():
    lam = 0.5
    F = ps(np.diag([lam - 1.0, 0.0]))
    p = PARAMS["foam_a"]
    P33 = first_piola_kirchhoff_stress(F, p)[2, 2]
    # Compression with beta > 0: transverse plane-strain constraint is compressive
    assert P33 < -0.01 * p.mu
    assert P33 == pytest.approx(2 * p.mu / p.alpha * (1 - lam ** (-p.alpha * p.beta)))
    # beta = 0: zero lateral stress in uniaxial strain (nu_0 = 0)
    P0 = first_piola_kirchhoff_stress(F, PARAMS["beta_zero"])
    assert abs(P0[1, 1]) < 1e-6 and abs(P0[2, 2]) < 1e-6


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("lam", [0.4, 0.8, 1.25])
def test_hydrostatic_analytic(name, lam):
    """F = lam I:  P = (2mu/a)(lam^(a-1) - lam^(-3ab-1)) I."""
    p = PARAMS[name]
    a, b = p.alpha, p.beta
    P = first_piola_kirchhoff_stress(lam * np.eye(3), p)
    expected = 2 * p.mu / a * (lam ** (a - 1) - lam ** (-3 * a * b - 1))
    np.testing.assert_allclose(P, expected * np.eye(3), rtol=1e-11, atol=1e-6)


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("gamma", [0.1, 0.8, 2.0])
def test_simple_shear_analytic(name, gamma):
    """Isochoric simple shear: stretches L, 1/L, 1 with L = g/2 + sqrt(1 + g^2/4);
    sigma12 = (2mu/a)(L^a - L^-a)/sqrt(4 + g^2), sigma33 = 0, W indep. of beta."""
    p = PARAMS[name]
    a, mu = p.alpha, p.mu
    F = ps([[0.0, gamma], [0.0, 0.0]])
    L = gamma / 2 + math.sqrt(1 + gamma**2 / 4)
    np.testing.assert_allclose(
        principal_stretches(F), sorted([1 / L, 1.0, L]), rtol=1e-12
    )
    W = 2 * mu / a**2 * (L**a + L ** (-a) - 2)
    assert strain_energy_density(F, p) == pytest.approx(W, rel=1e-11)
    sigma = cauchy_stress(F, p)
    s12 = 2 * mu / a * (L**a - L ** (-a)) / math.sqrt(4 + gamma**2)
    assert sigma[0, 1] == pytest.approx(s12, rel=1e-10)
    assert sigma[1, 0] == pytest.approx(s12, rel=1e-10)
    assert abs(sigma[2, 2]) < 1e-8 * mu


# ------------------------------------------------------- beta -> 0 limit


@pytest.mark.parametrize("alpha", [8.0, -3.0])
def test_beta_zero_limit_is_continuous(alpha):
    J = np.array([0.1, 0.5, 0.99, 1.0, 1.7, 5.0])
    ref = -alpha * np.log(J)
    np.testing.assert_allclose(volumetric_term(J, alpha, 0.0), ref, atol=1e-15)
    for beta in [1e-14, -1e-14, 1e-9, -1e-6]:
        # (J^(-ab)-1)/b = -a lnJ (1 + x/2 + ...), x = -a b lnJ
        x = -alpha * beta * np.log(J)
        np.testing.assert_allclose(
            volumetric_term(J, alpha, beta),
            ref * (1 + x / 2 + x**2 / 6),
            rtol=1e-13,
            atol=1e-15,
        )
    F = STATES["compress_shear"]
    W0 = strain_energy_density(F, HyperfoamParameters(1e6, alpha, 0.0))
    for beta in [1e-12, -1e-12]:
        W = strain_energy_density(F, HyperfoamParameters(1e6, alpha, beta))
        assert W == pytest.approx(W0, rel=1e-10)


# ------------------------------------------------------- objectivity


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("state", ["compress_shear", "general_3d", "extend_1.6"])
def test_frame_indifference_and_isotropy(name, state):
    p, F = PARAMS[name], STATES[state]
    W = strain_energy_density(F, p)
    P = first_piola_kirchhoff_stress(F, p)
    sig = cauchy_stress(F, p)
    for Q in [rot3([0, 0, 1], 0.7), rot3([1, 2, -0.5], 2.1)]:
        QF = Q @ F
        assert strain_energy_density(QF, p) == pytest.approx(W, rel=1e-12, abs=1e-6)
        np.testing.assert_allclose(
            first_piola_kirchhoff_stress(QF, p),
            Q @ P,
            atol=1e-9 * max(p.mu, np.abs(P).max()),
        )
        np.testing.assert_allclose(
            cauchy_stress(QF, p), Q @ sig @ Q.T, atol=1e-9 * p.mu
        )
        # material isotropy
        assert strain_energy_density(F @ Q, p) == pytest.approx(W, rel=1e-12, abs=1e-6)
    # rigid rotation alone is stress free
    Q = rot3([0, 0, 1], 1.3)
    assert strain_energy_density(Q, p) == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(first_piola_kirchhoff_stress(Q, p), 0.0, atol=1e-8)


def test_cauchy_stress_symmetric():
    for p in PARAMS.values():
        sig = cauchy_stress(STATES["general_3d"], p)
        np.testing.assert_allclose(sig, sig.T, atol=1e-9 * p.mu)


# ------------------------------------------------- inadmissible states


@pytest.mark.parametrize(
    "F",
    [
        np.diag([-1.0, 1.0, 1.0]),  # reflection, J = -1
        ps(np.diag([-1.5, 0.0])),  # inverted plane-strain compression
        ps(np.diag([-1.0, 0.0])),  # J = 0
        np.full((3, 3), np.nan),
    ],
)
def test_inadmissible_states_rejected(F):
    p = PARAMS["foam_a"]
    for fn in (
        strain_energy_density,
        first_piola_kirchhoff_stress,
        second_piola_kirchhoff_stress,
        cauchy_stress,
    ):
        with pytest.raises(InadmissibleDeformationError, match="J = det F must be > 0"):
            fn(F, p)


def test_batch_rejection_and_broadcasting():
    p = PARAMS["foam_a"]
    Fs = np.stack([STATES["shear_0.8"], STATES["compress_0.3"]])
    W = strain_energy_density(Fs, p)
    assert W.shape == (2,)
    assert first_piola_kirchhoff_stress(Fs, p).shape == (2, 3, 3)
    Fs_bad = np.stack([STATES["shear_0.8"], np.diag([1.0, -1.0, 1.0])])
    with pytest.raises(InadmissibleDeformationError, match="1 invalid"):
        check_admissible_jacobian(Fs_bad)
    with pytest.raises(ValueError):
        check_admissible_jacobian(np.eye(2))


def test_energy_nonnegative_on_sampled_states():
    """W >= 0 for beta >= 0: W = (2mu/a^2)[sum_i (lam_i^a - 1 - a ln lam_i)
    + (e^y - 1 - y)/beta] with y = -a beta ln J, and both brackets are >= 0.
    (For beta < 0 the volumetric bracket is <= 0, so W >= 0 is NOT guaranteed.)"""
    rng = np.random.default_rng(3)
    for p in (q for q in PARAMS.values() if q.beta >= 0):
        for _ in range(50):
            F = np.eye(3) + 0.4 * rng.standard_normal((3, 3))
            if np.linalg.det(F) <= 0.05:
                continue
            assert strain_energy_density(F, p) >= -1e-9 * p.mu
