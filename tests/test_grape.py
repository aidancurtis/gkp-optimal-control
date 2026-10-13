"""GRAPE on a driven qubit: both optimizer front ends, the sweep, and pulse I/O."""

import jax.numpy as jnp
import numpy as np
import pytest

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control.grape import (
    FourierBand,
    Penalties,
    System,
    TimeGrid,
    bandwidth_sweep,
    forward_evolve,
    load_pulse,
    run_grape,
    run_grape_batched,
    save_pulse,
)

SIGMA_X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
SIGMA_Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
SIGMA_Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


@pytest.fixture(scope="module")
def problem():
    system = System(
        H_drift=0.3 * SIGMA_Z,
        H_controls=jnp.stack([SIGMA_X, SIGMA_Y]),
        psi_init=jnp.array([1, 0], dtype=jnp.complex128),
        psi_targ=jnp.array([0, 1], dtype=jnp.complex128),
    )
    return system, TimeGrid(T=2.0, n_steps=40), Penalties(amp=1.0, eps_max=5.0, boundary=0.1)


def test_fourier_band_counts_allowed_modes(problem):
    system, grid, _ = problem
    band = FourierBand(f_max=2.0)
    freqs = np.fft.fftfreq(grid.n_steps, d=grid.dt)
    assert band.n_allowed(grid) == int(np.sum(np.abs(freqs) <= 2.0))
    assert band.param_shape(system, grid) == (2, band.n_allowed(grid), 2)
    assert FourierBand(f_max=2.0, f_min=0.1).n_allowed(grid) == band.n_allowed(grid) - 1


def test_forward_evolve_matches_closed_form(problem):
    system, grid, _ = problem
    # Constant resonant-free X drive of area pi/2 per unit amplitude: a pi pulse.
    amplitude = np.pi / (2 * grid.T)
    pulse = jnp.asarray(np.stack([np.full(grid.n_steps, amplitude), np.zeros(grid.n_steps)]))
    psi_f, history = forward_evolve(
        pulse, grid.dt, system.psi_init, 0.0 * SIGMA_Z, system.H_controls, return_history=True
    )
    assert history.shape == (grid.n_steps, 2)
    np.testing.assert_allclose(history[-1], psi_f)
    np.testing.assert_allclose(np.linalg.norm(history, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(psi_f, [0, -1j], atol=1e-12)
    np.testing.assert_allclose(
        forward_evolve(pulse, grid.dt, system.psi_init, 0.0 * SIGMA_Z, system.H_controls), psi_f
    )


def test_run_grape_reaches_target(problem):
    system, grid, penalties = problem
    band = FourierBand(f_max=2.0)
    result, diag, params_to_pulse = run_grape(
        system, grid, band, penalties, maxiter=200, verbose=False, progress_every=0
    )
    assert diag["F"] > 0.9999
    assert diag["pulse"].shape == (2, grid.n_steps)
    assert np.abs(diag["pulse"]).max() <= 5.0 + 1e-3
    assert diag["amp_penalty"] < 1e-6

    # The returned closure and diagnostics describe the same pulse and state.
    params = jnp.asarray(result.x.reshape(band.param_shape(system, grid)))
    np.testing.assert_allclose(params_to_pulse(params), diag["pulse"], atol=1e-12)
    psi_f = forward_evolve(
        jnp.asarray(diag["pulse"]), grid.dt, system.psi_init, system.H_drift, system.H_controls
    )
    np.testing.assert_allclose(psi_f, diag["psi_final"], atol=1e-12)
    assert abs(np.vdot(system.psi_targ, psi_f)) ** 2 == pytest.approx(diag["F"])

    # Band limit: the pulse has no spectral weight outside the allowed modes.
    spectrum = np.fft.fft(diag["pulse"], axis=-1)
    assert np.abs(spectrum[:, ~np.asarray(band.mask(grid))]).max() < 1e-9


def test_run_grape_respects_explicit_start(problem):
    system, grid, penalties = problem
    band = FourierBand(f_max=2.0)
    params0 = np.zeros(band.param_shape(system, grid))
    _, diag, _ = run_grape(
        system, grid, band, penalties, params0=params0, maxiter=0, verbose=False, progress_every=0
    )
    np.testing.assert_allclose(diag["pulse"], 0.0, atol=1e-15)
    assert diag["F"] == pytest.approx(0.0, abs=1e-12)  # drift alone never flips the qubit


def test_run_grape_batched_shapes_and_fidelity(problem):
    system, grid, penalties = problem
    bands = [FourierBand(f_max=2.0), FourierBand(f_max=2.2)]  # same number of modes
    assert bands[0].n_allowed(grid) == bands[1].n_allowed(grid)
    out = run_grape_batched(
        system, grid, bands, penalties, n_seeds_per_band=2, maxiter=200, verbose=False
    )
    n_allowed = bands[0].n_allowed(grid)
    assert out["F"].shape == (2, 2)
    assert out["pulse"].shape == (2, 2, 2, grid.n_steps)
    assert out["params"].shape == (2, 2, 2, n_allowed, 2)
    assert out["best_pulse"].shape == (2, 2, grid.n_steps)
    np.testing.assert_allclose(out["best_F"], out["F"].max(axis=1))
    assert out["best_F"].min() > 0.999

    psi_f = forward_evolve(
        jnp.asarray(out["best_pulse"][0]), grid.dt, system.psi_init, system.H_drift,
        system.H_controls,
    )  # fmt: skip
    assert abs(np.vdot(system.psi_targ, psi_f)) ** 2 == pytest.approx(out["best_F"][0], abs=1e-9)

    with pytest.raises(ValueError, match="same number"):
        run_grape_batched(
            system, grid, [FourierBand(f_max=1.0), FourierBand(f_max=3.0)], penalties,
            verbose=False,
        )  # fmt: skip


def test_bandwidth_sweep_groups_unequal_bands(problem):
    system, grid, penalties = problem
    f_max = [1.0, 2.0, 3.0]
    out = bandwidth_sweep(system, grid, f_max, penalties, n_seeds=1, maxiter=150, verbose=False)
    np.testing.assert_allclose(out["f_max"], f_max)
    assert out["F"].shape == (3, 1)
    assert out["best_pulse"].shape == (3, 2, grid.n_steps)
    assert out["best_F"].min() > 0.99


def test_pulse_save_load_round_trip(tmp_path):
    pulse = np.random.default_rng(0).normal(size=(2, 10))
    path = tmp_path / "pulse.npz"
    save_pulse(str(path), pulse, {"fidelity": 0.5, "T": 2.0})
    loaded, meta = load_pulse(str(path))
    np.testing.assert_array_equal(loaded, pulse)
    assert float(meta["fidelity"]) == 0.5 and float(meta["T"]) == 2.0
