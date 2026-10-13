"""State coercion and Wigner helpers."""

import jax.numpy as jnp
import jaxquantum as jqt
import numpy as np
import pytest

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control import operators as op
from gkp_optimal_control.states import fock_state
from gkp_optimal_control.utils import (
    compute_wigner,
    to_ket,
    to_ket_batch,
    to_qarray,
    wigner_trajectory,
)

N_FOCK = 30


def _fock(n: int) -> np.ndarray:
    return np.eye(N_FOCK, dtype=complex)[n]


def test_to_qarray_accepts_kets_and_density_matrices():
    qarray = jqt.basis(N_FOCK, 1)
    assert to_qarray(qarray) is qarray
    assert to_qarray(fock_state(N_FOCK, 1)).data.shape == (N_FOCK, 1)
    assert to_qarray(_fock(1)).data.shape == (N_FOCK, 1)
    assert to_qarray(_fock(1)[:, None]).data.shape == (N_FOCK, 1)
    assert to_qarray(np.outer(_fock(1), _fock(1))).data.shape == (N_FOCK, N_FOCK)
    with pytest.raises(ValueError, match="Ambiguous state shape"):
        to_qarray(np.zeros((2, N_FOCK)))  # an ECD qubit block
    with pytest.raises(ValueError, match="single state"):
        to_qarray(np.zeros((3, N_FOCK, N_FOCK)))


def test_to_ket_flattens_and_normalizes():
    for state in (
        fock_state(N_FOCK, 2),
        jqt.basis(N_FOCK, 2),
        3.0 * _fock(2),
        3.0 * _fock(2)[:, None],
    ):
        ket = to_ket(state)
        assert ket.shape == (N_FOCK,)
        np.testing.assert_allclose(ket, _fock(2))
    with pytest.raises(ValueError, match="density matrix"):
        to_ket(np.eye(N_FOCK))
    np.testing.assert_allclose(to_ket(3.0 * _fock(2)[:, None], normalize=False), 3.0 * _fock(2))


def test_to_ket_batch_shapes():
    cases = [
        (_fock(1), 1),
        (2.0 * _fock(1)[:, None], 1),
        (fock_state(N_FOCK, 1), 1),
        (jqt.basis(N_FOCK, 1), 1),
        (np.stack([_fock(0), 2j * _fock(3)]), 2),
        ([fock_state(N_FOCK, 0), jqt.basis(N_FOCK, 3), _fock(5)[:, None]], 3),
    ]
    for state, k in cases:
        batch = to_ket_batch(state, N_FOCK)
        assert batch.shape == (k, N_FOCK)
        np.testing.assert_allclose(np.linalg.norm(batch, axis=-1), 1.0)
    np.testing.assert_allclose(to_ket_batch([_fock(0), _fock(3)])[1], _fock(3))


def test_to_ket_batch_errors():
    with pytest.raises(ValueError, match="psi_targ has Fock dimension"):
        to_ket_batch(_fock(1), N_FOCK + 1, "psi_targ")
    with pytest.raises(ValueError, match="zero vector"):
        to_ket_batch(np.zeros(N_FOCK))


def test_wigner_of_vacuum_and_coherent_state():
    bound, grid = 5.0, 101
    xvec, yvec, wigner = compute_wigner(_fock(0), bound, bound, grid)
    assert wigner.shape == (grid, grid)
    dx = xvec[1] - xvec[0]
    assert wigner.sum() * dx * dx == pytest.approx(1.0, abs=1e-6)
    # Vacuum: W(x, p) = exp(-(x^2 + p^2)) / pi.
    x_grid, p_grid = np.meshgrid(xvec, yvec)
    np.testing.assert_allclose(wigner, np.exp(-(x_grid**2 + p_grid**2)) / np.pi, atol=1e-8)

    # D(alpha)|0> is centred at (sqrt(2) Re alpha, sqrt(2) Im alpha).
    alpha = 1.0 - 0.5j
    ket = np.asarray(op.displacement(N_FOCK, alpha))[:, 0]
    _, _, shifted = compute_wigner(ket, bound, bound, grid)
    row, col = np.unravel_index(shifted.argmax(), shifted.shape)
    assert xvec[col] == pytest.approx(np.sqrt(2) * alpha.real, abs=dx)
    assert yvec[row] == pytest.approx(np.sqrt(2) * alpha.imag, abs=dx)


def test_wigner_of_fock_one_is_negative_at_origin():
    xvec, _, wigner = compute_wigner(fock_state(N_FOCK, 1), 4.0, 4.0, 81)
    assert wigner[40, 40] == pytest.approx(-1 / np.pi, abs=1e-8)
    rho = np.outer(_fock(1), _fock(1))
    np.testing.assert_allclose(compute_wigner(rho, 4.0, 4.0, 81)[2], wigner, atol=1e-10)


def test_wigner_trajectory_matches_single_frames():
    kets = np.stack([_fock(0), _fock(1), _fock(2)])
    xvec, yvec, frames = wigner_trajectory(kets, 4.0, 4.0, 41)
    assert frames.shape == (3, 41, 41)
    for ket, frame in zip(kets, frames, strict=True):
        np.testing.assert_allclose(frame, compute_wigner(ket, 4.0, 4.0, 41)[2], atol=1e-12)

    _, _, picked = wigner_trajectory(list(kets), 4.0, 4.0, 41, slices=[2, 0])
    np.testing.assert_allclose(picked, frames[[2, 0]], atol=1e-12)
    batched = jqt.Qarray.create(jnp.asarray(kets)[..., None])
    np.testing.assert_allclose(wigner_trajectory(batched, 4.0, 4.0, 41)[2], frames, atol=1e-12)
