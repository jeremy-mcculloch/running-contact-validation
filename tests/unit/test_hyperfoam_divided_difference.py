"""Divided-difference coefficients of the in-plane stretch sum (PR #20 review).

``g(s, D) = q+^p + q-^p`` with ``q+- = s +- sqrt(D)``. The NumPy evaluation
(:func:`inplane_stretch_sum_coefficients`) is compared with an 80-digit
``decimal`` reference along ``t = sqrt(D)/s -> 0``, including ``t = 0``
exactly, for integer, fractional and negative non-integer ``p = alpha/2``.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from running_contact_validation.fe.hyperfoam import inplane_stretch_sum_coefficients

# p = alpha / 2 (alpha = 8, 4, 2, 2.5, 20, -3, 1.37, -2.71, 0.5)
P_VALUES = [4.0, 2.0, 1.0, 1.25, 10.0, -1.5, 0.685, -1.355, 0.25]
S_VALUES = [0.3, 1.0, 2.5]
T_VALUES = [0.0, 1e-300, 1e-16, 1e-12, 1e-8, 1e-5, 1e-3, 0.1, 0.5, 0.9]


def _falling(p, k):
    out = Decimal(1)
    for j in range(k):
        out *= p - j
    return out


def reference(s, D, p):
    """High-precision coefficients from the defining formulas (cancellation is
    harmless at 80 digits for the t used here); exact limits at D = 0."""
    with localcontext() as ctx:
        ctx.prec = 80
        s, D, p = Decimal(s), Decimal(D), Decimal(p)

        def f(q, k):
            return _falling(p, k) * ((p - k) * q.ln()).exp()

        if D == 0:
            g = 2 * f(s, 0)
            g_s, g_ss = 2 * f(s, 1), 2 * f(s, 2)
            g_D, g_sD = f(s, 2), f(s, 3)
            K = Decimal(0)
        else:
            d = D.sqrt()
            qp, qm = s + d, s - d
            g = f(qp, 0) + f(qm, 0)
            g_s = f(qp, 1) + f(qm, 1)
            g_ss = f(qp, 2) + f(qm, 2)
            g_D = (f(qp, 1) - f(qm, 1)) / (2 * d)
            g_sD = (f(qp, 2) - f(qm, 2)) / (2 * d)
            g_DD = (d * (f(qp, 2) + f(qm, 2)) - (f(qp, 1) - f(qm, 1))) / (4 * d**3)
            K = g_DD * D
        return {
            k: float(v)
            for k, v in {
                "g": g,
                "g_s": g_s,
                "g_D": g_D,
                "g_ss": g_ss,
                "g_sD": g_sD,
                "K": K,
            }.items()
        }


@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("s", S_VALUES)
@pytest.mark.parametrize("t", T_VALUES)
def test_coefficients_match_high_precision_reference(p, s, t):
    D = (t * s) ** 2
    got = inplane_stretch_sum_coefficients(s, D, p)
    ref = reference(s, D, p)
    # Input conditioning: q- = s - sqrt(D) carries a relative rounding error
    # ~ eps s/q-, which a power q-^m amplifies by |m|; grows only as t -> 1.
    rel = 4e-15 * (1.0 + (abs(p) + 2.0) * t / (1.0 - t))
    # natural scale of second-derivative coefficients, for zero-valued cases
    scale2 = (abs(p * (p - 1.0)) + abs(p)) * s ** (p - 2.0)
    scale3 = scale2 / s
    for key, scale in [
        ("g", s**p),
        ("g_s", abs(p) * s ** (p - 1)),
        ("g_ss", scale2),
        ("g_D", scale2),
        ("g_sD", scale3),
    ]:
        assert got[key] == pytest.approx(ref[key], rel=rel, abs=rel * scale), key
    # K = g_DD D = g_ss/4 - g_D/2 is O(t^2) as t -> 0; its accuracy is absolute
    # with respect to the tangent scale |g_ss| + |g_D| (the terms it combines)
    k_scale = abs(ref["g_ss"]) + abs(ref["g_D"]) + scale2 * 1e-300
    assert got["K"] == pytest.approx(ref["K"], abs=rel * k_scale)


@pytest.mark.parametrize("p", P_VALUES)
def test_exact_coincident_limits(p):
    s = 0.7
    c = inplane_stretch_sum_coefficients(s, 0.0, p)
    assert c["g"] == pytest.approx(2 * s**p, rel=1e-15)
    assert c["g_s"] == pytest.approx(2 * p * s ** (p - 1), rel=1e-15)
    assert c["g_D"] == pytest.approx(p * (p - 1) * s ** (p - 2), rel=1e-15, abs=1e-300)
    assert c["g_ss"] == pytest.approx(
        2 * p * (p - 1) * s ** (p - 2), rel=1e-15, abs=1e-300
    )
    assert c["g_sD"] == pytest.approx(
        p * (p - 1) * (p - 2) * s ** (p - 3), rel=1e-15, abs=1e-300
    )
    assert abs(c["K"]) <= 1e-15 * abs(p * (p - 1)) * s ** (p - 2)


def test_naive_divided_differences_lose_digits():
    """Sensitivity check: the textbook formulas lose ~log10(1/t) (g_D) digits,
    so the reference comparison above is a meaningful accuracy test."""
    p, s, t = -1.355, 1.0, 1e-8
    d = t * s
    qp, qm = s + d, s - d
    naive_gD = p * (qp ** (p - 1) - qm ** (p - 1)) / (2 * d)
    ref = reference(s, d * d, p)["g_D"]
    assert abs(naive_gD - ref) / abs(ref) > 1e-10
    good = inplane_stretch_sum_coefficients(s, d * d, p)["g_D"]
    assert abs(good - ref) / abs(ref) < 4e-15


def test_vectorized():
    s = np.array([0.5, 1.0, 2.0])
    D = np.array([0.0, 1e-20, 0.1])
    c = inplane_stretch_sum_coefficients(s, D, 1.37)
    assert all(v.shape == (3,) for v in c.values())
    assert np.all(np.isfinite(np.stack(list(c.values()))))
