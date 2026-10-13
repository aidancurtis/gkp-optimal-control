"""Physics invariants of the quantum-brachistochrone Hamiltonian.

The defining property is that evolving ``initial_state`` under ``h_optimal``
for ``min_time`` lands on ``final_state`` up to a global phase. A sign or
phase-convention error in the construction leaves the Hamiltonian Hermitian
with the right norm and the right ``min_time``, so only the evolved-state
fidelity catches it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from gkp_optimal_control.brachistochrone import quantum_brachistochrone_hamiltonian
from gkp_optimal_control.states import cat_states, fock_state, gkp_states

N_FOCK = 80
GKP_ALPHA = np.sqrt(np.pi / 2)
GKP_BETA = 1j * np.sqrt(np.pi / 2)
GKP_DELTA = 0.3
GKP_CUTOFF = 30


def _ket(state) -> jnp.ndarray:
    """Flatten a Qarray or array to a normalized ``(dim,)`` ket."""
    data = jnp.asarray(getattr(state, "data", state)).reshape(-1)
    return data / jnp.linalg.norm(data)


def _random_ket(seed: int, dim: int = 12) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return _ket(rng.normal(size=dim) + 1j * rng.normal(size=dim))


def _evolve(h: jnp.ndarray, psi: jnp.ndarray, t: float) -> np.ndarray:
    """Apply exp(-i h t) to psi via the eigendecomposition of Hermitian h."""
    evals, evecs = np.linalg.eigh(np.asarray(h))
    return evecs @ (np.exp(-1j * evals * t) * (evecs.conj().T @ np.asarray(psi)))


def _fidelity(a, b) -> float:
    return float(np.abs(np.vdot(np.asarray(a), np.asarray(b))) ** 2)


def _cases() -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    gkp_0, gkp_1 = (_ket(s) for s in gkp_states(N_FOCK, GKP_ALPHA, GKP_BETA, GKP_DELTA, GKP_CUTOFF))
    cat_even, _ = cat_states(N_FOCK, 2.0)
    vac = _ket(fock_state(N_FOCK, 0))
    return {
        "vac_to_gkp0": (vac, gkp_0),
        "vac_to_gkp1": (vac, gkp_1),
        "gkp0_to_gkp1": (gkp_0, gkp_1),
        "vac_to_cat": (vac, _ket(cat_even)),
        # Nonzero overlap phase: the branch a real-overlap case never exercises.
        "vac_to_gkp0_phased": (vac, jnp.exp(0.7j) * gkp_0),
        "vac_to_gkp0_negated": (vac, -gkp_0),
        "random_complex": (_random_ket(0), _random_ket(1)),
    }


CASES = _cases()


@pytest.fixture(params=list(CASES), ids=list(CASES))
def states(request) -> tuple[jnp.ndarray, jnp.ndarray]:
    return CASES[request.param]


def test_reaches_target_at_min_time(states):
    psi_i, psi_f = states
    h, min_time = quantum_brachistochrone_hamiltonian(psi_i, psi_f)
    psi_t = _evolve(h, psi_i, float(min_time))
    assert _fidelity(psi_f, psi_t) == pytest.approx(1.0, abs=1e-10)


def test_follows_geodesic(states):
    """Fidelity with the target rises as cos^2(theta_B * (1 - t/T)) along the path."""
    psi_i, psi_f = states
    h, min_time = quantum_brachistochrone_hamiltonian(psi_i, psi_f)
    bures_angle = np.arccos(np.clip(np.abs(np.vdot(psi_i, psi_f)), 0.0, 1.0))
    for frac in (0.0, 0.25, 0.5, 0.75):
        psi_t = _evolve(h, psi_i, frac * float(min_time))
        expected = np.cos(bures_angle * (1.0 - frac)) ** 2
        assert _fidelity(psi_f, psi_t) == pytest.approx(expected, abs=1e-10)


def test_hamiltonian_is_hermitian_with_bounded_norm(states):
    psi_i, psi_f = states
    energy_bound = 2.5
    h, _ = quantum_brachistochrone_hamiltonian(psi_i, psi_f, energy_bound=energy_bound)
    h = np.asarray(h)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-12)
    assert np.linalg.norm(h, ord=2) == pytest.approx(energy_bound, abs=1e-10)


def test_min_time_is_bures_angle_over_energy(states):
    psi_i, psi_f = states
    bures_angle = np.arccos(np.clip(np.abs(np.vdot(psi_i, psi_f)), 0.0, 1.0))
    for energy_bound in (0.5, 1.0, 3.0):
        h, min_time = quantum_brachistochrone_hamiltonian(psi_i, psi_f, energy_bound=energy_bound)
        assert float(min_time) == pytest.approx(bures_angle / energy_bound, abs=1e-12)
        psi_t = _evolve(h, psi_i, float(min_time))
        assert _fidelity(psi_f, psi_t) == pytest.approx(1.0, abs=1e-10)
