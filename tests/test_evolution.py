"""The diffrax Schrödinger solver against exact propagators."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control.evolution import evolve, piecewise_constant, time_independent

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.diag([1.0 + 0j, -1.0])
PSI0 = jnp.array([1, 0], dtype=jnp.complex128)


def test_time_independent_evolution_matches_expm():
    h_drift = 0.8 * SIGMA_Z + 0.5 * SIGMA_X
    no_controls = jnp.zeros((0, 2, 2), dtype=jnp.complex128)
    psi = evolve(PSI0, jnp.asarray(h_drift), no_controls, time_independent(), 0.0, 3.0)
    np.testing.assert_allclose(psi, expm(-3.0j * h_drift)[:, 0], atol=1e-6)

    times = jnp.linspace(0.0, 3.0, 7)
    traj = evolve(
        PSI0, jnp.asarray(h_drift), no_controls, time_independent(), 0.0, 3.0, saveat=times
    )
    assert traj.shape == (7, 2)
    for t, psi_t in zip(np.asarray(times), traj, strict=True):
        np.testing.assert_allclose(psi_t, expm(-1j * t * h_drift)[:, 0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(traj, axis=-1), 1.0, atol=1e-6)


def test_piecewise_constant_coefficients():
    pulses = jnp.asarray(np.arange(8.0).reshape(4, 2))
    coeffs = piecewise_constant(pulses, 1.0, 3.0)  # four bins of width 0.5
    for t, row in ((1.0, 0), (1.49, 0), (1.5, 1), (2.6, 3), (3.0, 3)):
        np.testing.assert_array_equal(coeffs(t), pulses[row])


def test_piecewise_constant_evolution_matches_propagator_product():
    rng = np.random.default_rng(0)
    n_steps, t1 = 6, 1.2
    pulses = rng.normal(size=(n_steps, 2))
    h_drift = 0.3 * SIGMA_Z
    controls = np.stack([SIGMA_X, SIGMA_Y])

    psi = evolve(
        PSI0,
        jnp.asarray(h_drift),
        jnp.asarray(controls),
        piecewise_constant(jnp.asarray(pulses), 0.0, t1),
        0.0,
        t1,
        rtol=1e-10,
        atol=1e-12,
    )
    expected = np.asarray(PSI0)
    for row in pulses:
        h = h_drift + row[0] * SIGMA_X + row[1] * SIGMA_Y
        expected = expm(-1j * (t1 / n_steps) * h) @ expected
    assert abs(np.vdot(expected, np.asarray(psi))) ** 2 == pytest.approx(1.0, abs=1e-4)
