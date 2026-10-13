"""The MPC against report Table 3."""

import math

from dock_control.mpc import augment, f_c, linearise, predict_matrices
import numpy as np
import pytest

X_C = np.array([-0.4, 0.05, 0.2])
U = np.array([0.2, -0.05, 0.1])


def _central_differences(fun, v, h=1e-6):
    cols = []
    for i in range(v.size):
        e = np.zeros_like(v)
        e[i] = h
        cols.append((fun(v + e) - fun(v - e)) / (2.0 * h))
    return np.column_stack(cols)


def test_linearisation_matches_finite_differences(p):
    """A_k and B_k agree with central differences of report eq. 9."""
    Ts = p['mpc']['Ts']
    A, B = linearise(X_C, U, Ts)
    assert np.allclose(A, _central_differences(lambda x: f_c(x, U, Ts), X_C), atol=1e-6)
    assert np.allclose(B, _central_differences(lambda u: f_c(X_C, u, Ts), U), atol=1e-6)


def test_prediction_matrices_match_brute_force_stepping(p):
    """F and Phi reproduce stepping the incremental model of report eq. 11 by hand."""
    Np, Nc = p['mpc']['Np'], p['mpc']['Nc']
    rng = np.random.default_rng(0)
    A = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    B = 0.05 * rng.normal(size=(3, 3))
    F, Phi = predict_matrices(*augment(A, B), Np, Nc)
    assert F.shape == (3 * Np, 6) and Phi.shape == (3 * Np, 3 * Nc)
    x_p, dU = rng.normal(size=6), rng.normal(size=3 * Nc)
    dx, x, Y = x_p[:3].copy(), x_p[3:].copy(), np.empty(3 * Np)
    for i in range(Np):
        du = dU[3 * i:3 * i + 3] if i < Nc else np.zeros(3)
        dx = A @ dx + B @ du
        x = x + dx
        Y[3 * i:3 * i + 3] = x
    assert np.allclose(F @ x_p + Phi @ dU, Y, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('solver', ['daqp', 'osqp'])
def test_unconstrained_tick_matches_closed_form(p, solver):
    """
    Check that, with the limits removed, a tick equals the closed form (report Table 3).

    dU = (Phi' Q_bar Phi + R_bar)^-1 Phi' Q_bar (Y_r - F x_p), to the solver tolerance.
    """
    pytest.importorskip(solver)
    from dock_control.mpc import Mpc
    for key in ('a_max', 'alpha_max', 'w_max'):
        p['lim'][key] = math.inf
    p['mpc']['contact_band'] = p['mpc']['th_max_deg'] = math.inf
    Ts, Np, Nc = p['mpc']['Ts'], p['mpc']['Np'], p['mpc']['Nc']
    x_hat, r_gyro = np.array([-1.0, 0.1, 0.05, 0.1, 0.0]), 0.02
    u_prev, y_ref = np.array([0.1, 0.0, 0.02]), np.array([-0.3, 0.0, 0.0])

    res = Mpc(p, solver).step(x_hat, r_gyro, u_prev, y_ref, math.inf, math.inf)

    F, Phi = predict_matrices(*augment(*linearise(x_hat[:3], u_prev, Ts)), Np, Nc)
    x_p = np.concatenate((Ts * np.array((x_hat[3], x_hat[4], r_gyro)), x_hat[:3]))
    q_bar = np.tile(np.asarray(p['mpc']['Qy'], dtype=float), Np)
    r_bar = np.tile(np.asarray(p['mpc']['rw'], dtype=float) * np.asarray(p['mpc']['Ru']), Nc)
    dU = np.linalg.solve(Phi.T @ (q_bar[:, None] * Phi) + np.diag(r_bar),
                         Phi.T @ (q_bar * (np.tile(y_ref, Np) - F @ x_p)))
    assert not res.fallback and res.slack == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(res.dU, dU, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize('solver', ['daqp', 'osqp'])
def test_constrained_tick_respects_every_limit(p, solver):
    """Check a tick from 0.5 m out at the cap: 320 rows, the cap active, every limit held."""
    pytest.importorskip(solver)
    from dock_control.mpc import Mpc
    mpc = Mpc(p, solver)
    Nc, tol = mpc.Nc, 1e-3
    v_max, r_max = p['lim']['v_approach'], p['lim']['r_max']
    u_prev = np.array([v_max, 0.0, 0.0])

    res = mpc.step(np.array([-0.5, 0.0, 0.0, v_max, 0.0]), 0.0, u_prev, (0.0, 0.0, 0.0),
                   v_max, r_max)

    assert mpc.m == 20 * Nc + 3 * mpc.Np and not res.fallback
    dU = res.dU.reshape(Nc, 3)
    U = u_prev + np.cumsum(dU, axis=0)
    speed = np.abs(U[:, 0]) + np.abs(U[:, 1])
    assert np.all(np.abs(dU) <= mpc.du_max + tol)
    assert np.all(speed <= v_max + tol) and np.all(np.abs(U[:, 2]) <= r_max + tol)
    assert np.all(np.abs(U @ mpc.J_inv.T) <= mpc.w_max + tol)
    assert np.all(res.y_pred[:, 0] <= mpc.x_max + res.slack + tol)
    assert np.all(np.abs(res.y_pred[:, 2]) <= mpc.th_max + res.slack + tol)
    assert speed.max() == pytest.approx(v_max, abs=tol)
