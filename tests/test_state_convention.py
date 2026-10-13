"""One state representation across the package.

Constructors return 1-D complex128 JAX arrays, and every function that takes a
state accepts that, a NumPy array, a column vector or a ``jaxquantum.Qarray``
and gives the same answer.
"""

import jax
import jax.numpy as jnp
import jaxquantum as jqt
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import gkp_optimal_control as gkp  # noqa: E402
from gkp_optimal_control import device, pulse_simulation, pulses  # noqa: E402
from gkp_optimal_control.evolution import evolve, time_independent  # noqa: E402
from gkp_optimal_control.grape import System, forward_evolve  # noqa: E402
from gkp_optimal_control.states import coherent_state_vectors  # noqa: E402


def _close(actual, desired):
    """Equal up to round-off; ``Qarray.create`` zeroes entries below ~1e-15."""
    np.testing.assert_allclose(actual, desired, rtol=1e-10, atol=1e-12)


N_FOCK = 24
ALPHA, BETA = np.sqrt(np.pi / 2), 1j * np.sqrt(np.pi / 2)

CONSTRUCTED = {
    "fock": lambda: gkp.fock_state(N_FOCK, 2),
    "gkp_0": lambda: gkp.gkp_states(N_FOCK, ALPHA, BETA, 0.4, 6)[0],
    "gkp_1": lambda: gkp.gkp_states(N_FOCK, ALPHA, BETA, 0.4, 6)[1],
    "cat_even": lambda: gkp.cat_states(N_FOCK, 1.2)[0],
    "cat_odd": lambda: gkp.cat_states(N_FOCK, 1.2)[1],
    "squeezed": lambda: gkp.squeezed_vacuum(N_FOCK, 0.4),
    "coherent": lambda: coherent_state_vectors(N_FOCK, jnp.asarray([0.5 + 0.2j]))[0],
}


def _forms(ket) -> dict:
    """The same ket in every form a caller might hold it."""
    arr = np.asarray(ket)
    return {
        "jax": jnp.asarray(arr),
        "numpy": arr,
        "column": arr[:, None],
        "qarray": jqt.Qarray.create(jnp.asarray(arr)[:, None]),
    }


@pytest.mark.parametrize("name", list(CONSTRUCTED))
def test_constructors_return_normalized_1d_jax_arrays(name):
    state = CONSTRUCTED[name]()
    assert isinstance(state, jax.Array)
    assert state.shape == (N_FOCK,) and state.dtype == jnp.complex128
    assert float(jnp.linalg.norm(state)) == pytest.approx(1.0)
    # Usable directly with jax.numpy, without unwrapping.
    assert jnp.vdot(state, state).real == pytest.approx(1.0)
    assert state[-4:].shape == (4,) and state.astype(jnp.complex128) is not None


FORMS = ("jax", "numpy", "column", "qarray")


@pytest.fixture(scope="module")
def kets():
    return _forms(gkp.fock_state(N_FOCK, 0)), _forms(gkp.gkp_states(N_FOCK, ALPHA, BETA, 0.4, 6)[0])


@pytest.mark.parametrize("form", FORMS)
def test_coercion_helpers(kets, form):
    vac, target = kets
    _close(gkp.to_ket(target[form]), target["numpy"])
    _close(gkp.to_ket_batch(target[form], N_FOCK), target["numpy"][None])
    _close(gkp.to_qarray(target[form]).data[:, 0], target["numpy"])
    _close(gkp.to_ket_batch([vac[form], target["jax"]]), np.stack([vac["numpy"], target["numpy"]]))


@pytest.mark.parametrize("form", FORMS)
def test_brachistochrone_accepts_any_form(kets, form):
    vac, target = kets
    h_ref, t_ref = gkp.quantum_brachistochrone_hamiltonian(vac["jax"], target["jax"])
    h, t_min = gkp.quantum_brachistochrone_hamiltonian(vac[form], target[form])
    assert isinstance(h, jax.Array) and h.shape == (N_FOCK, N_FOCK)
    _close(h, h_ref)
    assert float(t_min) == pytest.approx(float(t_ref))


@pytest.mark.parametrize("form", FORMS)
def test_gate_optimization_accepts_any_form(kets, form):
    vac, _ = kets
    target = _forms(gkp.fock_state(N_FOCK, 1))
    config = gkp.OptimizerConfig(n_seeds=2, n_adam_iters=30, polish=False)
    run = lambda a, b: gkp.optimize_gate_sequence(  # noqa: E731
        gkp.snap_sequence(1, N_FOCK, n_snap=4), a, b, optimizer=config, verbose=False
    )
    reference, result = run(vac["jax"], target["jax"]), run(vac[form], target[form])
    np.testing.assert_array_equal(result.flat_params, reference.flat_params)
    assert isinstance(result.final_ket(), np.ndarray) and result.final_ket().shape == (N_FOCK,)
    _close(gkp.sequence_history(result, vac[form]), gkp.sequence_history(reference, vac["jax"]))


@pytest.mark.parametrize("form", FORMS)
def test_grape_and_evolution_accept_any_form(kets, form):
    vac, target = kets
    a = np.diag(np.sqrt(np.arange(1, N_FOCK)), 1).astype(complex)
    h_drift = jnp.asarray(0.2 * a.conj().T @ a)
    h_controls = jnp.asarray(np.stack([a + a.conj().T]))

    system = System(h_drift, h_controls, vac[form], target[form])
    assert system.psi_init.shape == system.psi_targ.shape == (N_FOCK,)
    _close(system.psi_targ, target["numpy"])

    pulse = jnp.asarray(0.3 * np.ones((1, 5)))
    ref = forward_evolve(pulse, 0.1, target["jax"], h_drift, h_controls)
    _close(forward_evolve(pulse, 0.1, target[form], h_drift, h_controls), ref)
    # Linear in the state: an unnormalized ket is not silently rescaled.
    _close(forward_evolve(pulse, 0.1, 2.0 * target["numpy"], h_drift, h_controls), 2.0 * ref)

    none = jnp.zeros((0, N_FOCK, N_FOCK), dtype=jnp.complex128)
    out = evolve(target[form], h_drift, none, time_independent(), 0.0, 0.5)
    _close(out, evolve(target["jax"], h_drift, none, time_independent(), 0.0, 0.5))
    assert out.shape == (N_FOCK,)


@pytest.mark.parametrize("form", FORMS)
def test_pulse_simulation_accepts_any_form(kets, form):
    _, target = kets
    params = device.SystemParams().replace(dt=10e-3, kappa=0.0, chi_prime=0.0, kerr=0.0)
    seq = pulses.compile_snap_sequence(np.zeros((1, 4)), [0.1, -0.1], params=params, n_drive=4)
    seq.eps, seq.omega = seq.eps[:40], seq.omega[:40]  # a short slice is enough here
    seq.gate_indices = np.array([40])
    reference = pulse_simulation.simulate_sequence(seq, psi0=target["jax"], n_fock=N_FOCK)
    result = pulse_simulation.simulate_sequence(seq, psi0=target[form], n_fock=N_FOCK)
    _close(result.states, reference.states)
    assert result.fidelity(target[form]) == pytest.approx(reference.fidelity(target["jax"]))


@pytest.mark.parametrize("form", FORMS)
def test_plotting_accepts_any_form(kets, form):
    import matplotlib.pyplot as plt

    _, target = kets
    _, _, ref = gkp.compute_wigner(target["jax"], 4.0, 4.0, 31)
    _close(gkp.compute_wigner(target[form], 4.0, 4.0, 31)[2], ref)
    fig, axes = plt.subplots(1, 2)
    gkp.plot_wigner(target[form], x_bound=4.0, y_bound=4.0, ax=axes[0], grid_points=31)
    gkp.plot_photon_number(target[form], ax=axes[1])
    plt.close("all")
