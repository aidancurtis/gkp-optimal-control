"""State constructors: normalization, parity, and the conventions in their docstrings."""

import numpy as np
import pytest
from scipy.linalg import expm

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control import operators as op
from gkp_optimal_control.states import (
    cat_states,
    coherent_state_vectors,
    fock_state,
    gkp_states,
    squeezed_vacuum,
)

N_FOCK = 80
ALPHA = np.sqrt(np.pi / 2)
BETA = 1j * np.sqrt(np.pi / 2)


def _ket(state) -> np.ndarray:
    return np.asarray(state)


def _ladder(n_fock: int = N_FOCK) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n_fock)), 1).astype(complex)


def test_fock_state():
    ket = _ket(fock_state(N_FOCK, 3))
    assert ket.shape == (N_FOCK,)
    np.testing.assert_array_equal(ket, np.eye(N_FOCK)[3])
    for bad in (-1, N_FOCK):
        with pytest.raises(ValueError, match="outside the truncated space"):
            fock_state(N_FOCK, bad)


def test_coherent_state_vectors_match_displaced_vacuum():
    alphas = np.array([0.0, 0.7 - 0.3j, -1.2 + 0.4j])
    vectors = np.asarray(coherent_state_vectors(N_FOCK, alphas))
    assert vectors.shape == (3, N_FOCK)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), 1.0, atol=1e-12)
    for alpha, vec in zip(alphas, vectors, strict=True):
        np.testing.assert_allclose(
            vec, np.asarray(op.displacement(N_FOCK, alpha))[:, 0], atol=1e-10
        )


@pytest.mark.parametrize("alpha", [1.5, 0.8 + 1.1j])
def test_cat_states_have_definite_parity(alpha):
    even, odd = (_ket(s) for s in cat_states(N_FOCK, alpha))
    for ket in (even, odd):
        assert np.linalg.norm(ket) == pytest.approx(1.0)
    assert np.abs(even[1::2]).max() < 1e-12
    assert np.abs(odd[0::2]).max() < 1e-12
    assert abs(np.vdot(even, odd)) < 1e-12
    # Both are eigenstates of a^2 with eigenvalue alpha^2.
    a2 = _ladder() @ _ladder()
    for ket in (even, odd):
        np.testing.assert_allclose((a2 @ ket)[:40], alpha**2 * ket[:40], atol=1e-8)


@pytest.mark.parametrize("r, phi", [(0.5, 0.0), (0.9, 1.3)])
def test_squeezed_vacuum_matches_the_squeeze_operator(r, phi):
    ket = _ket(squeezed_vacuum(N_FOCK, r, phi))
    assert np.linalg.norm(ket) == pytest.approx(1.0)
    assert np.abs(ket[1::2]).max() == 0.0
    # S(xi)|0> with S = exp[(xi^* a^2 - xi a^dag^2) / 2], xi = r e^{i phi}.
    a = _ladder()
    xi = r * np.exp(1j * phi)
    reference = expm(0.5 * (np.conj(xi) * a @ a - xi * a.conj().T @ a.conj().T))[:, 0]
    np.testing.assert_allclose(ket[:30], reference[:30], atol=1e-8)
    if phi == 0.0:  # squeezed in x: Var(x) = e^{-2r} / 2
        x = (a + a.conj().T) / np.sqrt(2)
        assert np.real(np.vdot(ket, x @ x @ ket)) == pytest.approx(0.5 * np.exp(-2 * r), rel=1e-6)


@pytest.mark.parametrize("delta", [0.25, 0.35])
def test_gkp_states_are_normalized_and_nearly_orthogonal(delta):
    gkp_0, gkp_1 = (_ket(s) for s in gkp_states(N_FOCK, ALPHA, BETA, delta, 10))
    assert np.linalg.norm(gkp_0) == pytest.approx(1.0)
    assert np.linalg.norm(gkp_1) == pytest.approx(1.0)
    assert abs(np.vdot(gkp_0, gkp_1)) < 5e-3
    # Truncation is adequate: little weight at the top of the space.
    assert np.sum(np.abs(gkp_0[-8:]) ** 2) < 1e-3


def test_gkp_states_respect_the_lattice():
    """Stabilizers have expectation near +1 and logical Z separates the two states."""
    delta = 0.25
    gkp_0, gkp_1 = (_ket(s) for s in gkp_states(N_FOCK, ALPHA, BETA, delta, 10))
    stab_q = np.asarray(op.displacement(N_FOCK, 2 * ALPHA))
    stab_p = np.asarray(op.displacement(N_FOCK, 2 * BETA))
    logical_z = np.asarray(op.displacement(N_FOCK, BETA))
    for ket in (gkp_0, gkp_1):
        assert np.real(np.vdot(ket, stab_q @ ket)) > 0.8
        assert np.real(np.vdot(ket, stab_p @ ket)) > 0.8
    z0 = np.real(np.vdot(gkp_0, logical_z @ gkp_0))
    z1 = np.real(np.vdot(gkp_1, logical_z @ gkp_1))
    assert z0 > 0.8 and z1 < -0.8
    # The logical X displacement maps |0_L> onto |1_L>.
    logical_x = np.asarray(op.displacement(N_FOCK, ALPHA))
    assert abs(np.vdot(gkp_1, logical_x @ gkp_0)) > 0.9


def test_gkp_envelope_narrows_the_photon_distribution():
    n = np.arange(N_FOCK)
    mean_n = [
        float(np.sum(n * np.abs(_ket(gkp_states(N_FOCK, ALPHA, BETA, d, 10)[0])) ** 2))
        for d in (0.25, 0.35, 0.5)
    ]
    assert mean_n[0] > mean_n[1] > mean_n[2]
