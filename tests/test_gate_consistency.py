"""Consistency of the ECD and SNAP gate sets across their implementations.

The JAX block-basis propagators in :mod:`gate_optimization` (two displacement
methods each) and the builders in :mod:`operators` are pinned to the
conventions written in their docstrings, using independent dense references
built here with ``scipy.linalg.expm``.

Truncated displacements only compose like the infinite-dimensional ones away
from the Fock cutoff, so identities that rely on ``D(a) D(b) ~ D(a + b)`` or on
two different factorizations of ``D`` are checked on the low-Fock block
``[:LOW]``, with amplitudes small enough that the block is exact to round-off.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control import operators as op
from gkp_optimal_control import sequences as sq

N_FOCK = 48
LOW = 16
DISP_METHODS = ("fixed", "exact")
ALPHA = 0.6 - 0.4j

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


# ---------------------------------------------------------------------------
# Independent references and parameter helpers
# ---------------------------------------------------------------------------


def _ref_displacement(alpha: complex, n_fock: int = N_FOCK) -> np.ndarray:
    a = np.diag(np.sqrt(np.arange(1, n_fock)), 1).astype(complex)
    return expm(alpha * a.conj().T - np.conj(alpha) * a)


def _ref_rotation(theta: float, phi: float) -> np.ndarray:
    return expm(-0.5j * theta * (np.cos(phi) * SIGMA_X + np.sin(phi) * SIGMA_Y))


def _ref_ecd_history(thetas, phis, betas, ket) -> np.ndarray:
    """Dense reference for ``R_{N+1} ECD_N R_N ... ECD_1 R_1`` on ``|g>|ket>``.

    ``ECD(beta) = D(beta/2)|e><g| + D(-beta/2)|g><e|``. Returns the block state
    ``(2, n_fock)``, index 0 = ``|g>``, at the input and after every gate.
    """
    psi = np.stack([np.asarray(ket, dtype=complex), np.zeros(len(ket), dtype=complex)])
    history = [psi]
    for i, (theta, phi) in enumerate(zip(thetas, phis, strict=True)):
        psi = _ref_rotation(theta, phi) @ psi
        history.append(psi)
        if i < len(betas):
            d_op = _ref_displacement(betas[i] / 2, len(ket))
            psi = np.stack([d_op.conj().T @ psi[1], d_op @ psi[0]])
            history.append(psi)
    return np.stack(history)


def _displacement_product(alphas) -> tuple[complex, float]:
    r"""``D(a_n) ... D(a_1) = e^{i Phi} D(sum a)`` with ``Phi = sum_{i<j} Im(a_j conj(a_i))``."""
    alphas = np.asarray(alphas, dtype=complex)
    phase = sum(
        float(np.imag(alphas[j] * np.conj(alphas[i])))
        for i in range(len(alphas))
        for j in range(i + 1, len(alphas))
    )
    return complex(alphas.sum()), phase


def _low_kets(seed: int, k: int = 2, n_levels: int = 6) -> np.ndarray:
    """``k`` random normalized kets supported on the lowest ``n_levels`` levels."""
    rng = np.random.default_rng(seed)
    psi = np.zeros((k, N_FOCK), dtype=complex)
    psi[:, :n_levels] = rng.normal(size=(k, n_levels)) + 1j * rng.normal(size=(k, n_levels))
    return psi / np.linalg.norm(psi, axis=-1, keepdims=True)


def _ecd_params(n_gates: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, np.pi, n_gates + 1)
    phis = rng.uniform(-np.pi, np.pi, n_gates + 1)
    betas = 0.5 * (rng.normal(size=n_gates) + 1j * rng.normal(size=n_gates))
    return thetas, phis, betas


def _ecd_flat(thetas, phis, betas) -> jnp.ndarray:
    """Flat layout ``[(theta, phi) x (N+1), (Re beta, Im beta) x N]``."""
    betas = np.asarray(betas, dtype=complex)
    return jnp.asarray(
        np.concatenate(
            [np.stack([thetas, phis], -1).ravel(), np.stack([betas.real, betas.imag], -1).ravel()]
        )
    )


def _snap_params(n_gates: int, n_snap: int = N_FOCK, seed: int = 0):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (n_gates, n_snap))
    alphas = 0.4 * (rng.normal(size=n_gates + 1) + 1j * rng.normal(size=n_gates + 1))
    return phases, alphas


def _snap_flat(phases, alphas) -> jnp.ndarray:
    """Flat layout ``[phases x N, (Re alpha, Im alpha) x (N+1)]``."""
    alphas = np.asarray(alphas, dtype=complex)
    return jnp.asarray(
        np.concatenate([np.ravel(phases), np.stack([alphas.real, alphas.imag], -1).ravel()])
    )


def _run(seq, flat, psi_cav) -> np.ndarray:
    psi, *_ = seq.propagate(flat, seq.lift(jnp.asarray(psi_cav)))
    return np.asarray(psi)


# ---------------------------------------------------------------------------
# Displacements
# ---------------------------------------------------------------------------


def _matrix_free_displacement(alpha: complex) -> np.ndarray:
    """The matrix that ``make_displacement_apply`` applies, built column by column."""
    apply = op.make_displacement_apply(N_FOCK)
    return np.asarray(apply(jnp.eye(N_FOCK, dtype=jnp.complex128), alpha)).T


def test_exact_displacement_matches_reference():
    d_op = np.asarray(op.make_displacement(N_FOCK)(ALPHA))
    np.testing.assert_allclose(d_op, _ref_displacement(ALPHA), atol=1e-12)
    np.testing.assert_allclose(d_op.conj().T @ d_op, np.eye(N_FOCK), atol=1e-12)
    np.testing.assert_allclose(op.make_displacement(N_FOCK)(0.0), np.eye(N_FOCK), atol=1e-15)
    np.testing.assert_allclose(op.displacement(N_FOCK, ALPHA), d_op)


def test_matrix_free_displacement_is_unitary_and_matches_away_from_cutoff():
    d_op = _matrix_free_displacement(ALPHA)
    np.testing.assert_allclose(d_op.conj().T @ d_op, np.eye(N_FOCK), atol=1e-12)
    # A different factorization of D: equal to the exact one only away from the cutoff.
    ref = _ref_displacement(ALPHA)
    np.testing.assert_allclose(d_op[:LOW, :LOW], ref[:LOW, :LOW], atol=1e-12)
    assert np.abs(d_op - ref).max() > 1e-3


def test_matrix_free_displacement_adjoint():
    d_op = _matrix_free_displacement(ALPHA)
    apply = op.make_displacement_apply(N_FOCK)
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(3, N_FOCK)) + 1j * rng.normal(size=(3, N_FOCK))
    np.testing.assert_allclose(apply(jnp.asarray(psi), ALPHA), psi @ d_op.T, atol=1e-12)
    np.testing.assert_allclose(
        apply(jnp.asarray(psi), ALPHA, adjoint=True), psi @ d_op.conj(), atol=1e-12
    )


@pytest.mark.parametrize("build", [lambda: op.make_displacement(N_FOCK)(ALPHA),
                                   lambda: _matrix_free_displacement(ALPHA)],
                         ids=["exact", "matrix_free"])  # fmt: skip
def test_displaced_vacuum_is_coherent_state(build):
    d_op = np.asarray(build())
    n = np.arange(LOW)
    log_fact = np.cumsum(np.log(np.maximum(n, 1)))
    coherent = np.exp(-(abs(ALPHA) ** 2) / 2) * ALPHA**n / np.exp(0.5 * log_fact)
    np.testing.assert_allclose(d_op[:LOW, 0], coherent, atol=1e-12)


@pytest.mark.parametrize("r, phi", [(0.4, 0.7), (0.8, -2.0)])
def test_matrix_free_squeeze_matches_reference(r, phi):
    a = np.diag(np.sqrt(np.arange(1, N_FOCK)), 1).astype(complex)
    gen = 0.5 * r * (a @ a * np.exp(-1j * phi) - (a @ a).conj().T * np.exp(1j * phi))
    apply = op.make_squeeze_apply(N_FOCK)
    s_op = np.asarray(apply(jnp.eye(N_FOCK, dtype=jnp.complex128), r, phi)).T
    np.testing.assert_allclose(s_op, expm(gen), atol=1e-10)
    np.testing.assert_allclose(s_op.conj().T @ s_op, np.eye(N_FOCK), atol=1e-12)
    # Two phases at once: one per qubit block.
    psi = np.random.default_rng(5).normal(size=(2, 2, N_FOCK)) + 0j
    both = np.asarray(apply(jnp.asarray(psi), r, jnp.array([phi, phi + 1.0])))
    np.testing.assert_allclose(both[:, 0], psi[:, 0] @ s_op.T, atol=1e-10)
    assert not np.allclose(both[:, 1], psi[:, 1] @ s_op.T)


def test_displacement_product_matches_matrix_product():
    alphas = [0.3 + 0.2j, -0.25 + 0.4j, 0.1 - 0.35j]
    total, phase = _displacement_product(alphas)
    product = np.eye(N_FOCK, dtype=complex)
    for alpha in alphas:  # D(alpha_n) ... D(alpha_1)
        product = _ref_displacement(alpha) @ product
    merged = np.exp(1j * phase) * _ref_displacement(total)
    np.testing.assert_allclose(product[:LOW, :LOW], merged[:LOW, :LOW], atol=1e-12)


# ---------------------------------------------------------------------------
# Qubit rotations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta, phi", [(0.0, 0.3), (np.pi / 2, 0.0), (1.1, -2.0), (np.pi, 0.7)])
def test_qubit_rotation_matches_convention(theta, phi):
    np.testing.assert_allclose(op.qubit_rotation(theta, phi), _ref_rotation(theta, phi), atol=1e-14)


# ---------------------------------------------------------------------------
# ECD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", DISP_METHODS)
def test_ecd_gate_matches_definition(method):
    """ECD(beta) = D(beta/2)|e><g| + D(-beta/2)|g><e|, with identity rotations."""
    beta = 0.7 - 0.5j
    seq = sq.ecd_sequence(1, N_FOCK, disp_method=method)
    flat = _ecd_flat([0.0, 0.0], [0.0, 0.0], [beta])
    psi_g, psi_e = _low_kets(1)
    psi_in = jnp.asarray(np.stack([psi_g, psi_e])[None])  # (1, 2, n_fock)
    out = np.asarray(seq.propagate(flat, psi_in)[0])[0]
    np.testing.assert_allclose(out[1], _ref_displacement(beta / 2) @ psi_g, atol=1e-10)
    np.testing.assert_allclose(out[0], _ref_displacement(-beta / 2) @ psi_e, atol=1e-10)


@pytest.mark.parametrize("method", DISP_METHODS)
def test_echoed_gate_is_qubit_flip_of_bare_conditional(method):
    thetas, phis, betas = _ecd_params(1, seed=4)
    flat = _ecd_flat([thetas[0], 0.0], [phis[0], 0.0], betas)
    psi0 = _low_kets(2)
    echoed = _run(sq.ecd_sequence(1, N_FOCK, disp_method=method, echoed=True), flat, psi0)
    bare = _run(sq.ecd_sequence(1, N_FOCK, disp_method=method, echoed=False), flat, psi0)
    np.testing.assert_allclose(echoed, bare[:, ::-1, :], atol=1e-12)


@pytest.mark.parametrize("method", DISP_METHODS)
@pytest.mark.parametrize("n_gates", [1, 2, 4])
def test_ecd_sequence_matches_dense_reference(n_gates, method):
    thetas, phis, betas = _ecd_params(n_gates, seed=n_gates)
    seq = sq.ecd_sequence(n_gates, N_FOCK, disp_method=method)
    psi0 = _low_kets(5)
    out = _run(seq, _ecd_flat(thetas, phis, betas), psi0)
    assert out.shape == (len(psi0), 2, N_FOCK)
    for k, ket in enumerate(psi0):
        mirror = _ref_ecd_history(thetas, phis, betas, ket)[-1]
        np.testing.assert_allclose(out[k], mirror, atol=1e-10)


@pytest.mark.parametrize("method", DISP_METHODS)
def test_ecd_history_matches_reference_and_conserves_norm(method):
    n_gates = 3
    thetas, phis, betas = _ecd_params(n_gates, seed=6)
    seq = sq.ecd_sequence(n_gates, N_FOCK, disp_method=method)
    flat = _ecd_flat(thetas, phis, betas)
    psi0 = _low_kets(7)
    hist = np.asarray(seq.history(flat, seq.lift(jnp.asarray(psi0))))
    assert hist.shape == (2 * n_gates + 2, len(psi0), 2, N_FOCK)
    np.testing.assert_allclose(hist[-1], _run(seq, flat, psi0), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(hist, axis=(-2, -1)), 1.0, atol=1e-12)
    for k, ket in enumerate(psi0):
        mirror = _ref_ecd_history(thetas, phis, betas, ket)
        np.testing.assert_allclose(hist[:, k], mirror, atol=1e-10)


def test_ecd_parameter_layout():
    n_gates = 3
    thetas, phis, betas = _ecd_params(n_gates, seed=30)
    seq = sq.ecd_sequence(n_gates, N_FOCK)
    flat = _ecd_flat(thetas, phis, betas)
    assert seq.n_params == 4 * n_gates + 2 == flat.shape[0]
    assert seq.init(jax.random.PRNGKey(0)).shape == (seq.n_params,)

    described = seq.describe(flat)
    np.testing.assert_allclose(described["thetas"], thetas)
    np.testing.assert_allclose(described["phis"], phis)
    np.testing.assert_allclose(described["betas"], betas)
    np.testing.assert_allclose(seq.rotation_angles(flat), thetas)

    _, _, strengths, amps = seq.propagate(flat, seq.lift(jnp.asarray(_low_kets(9))))
    np.testing.assert_allclose(strengths, np.abs(betas), atol=1e-14)
    assert amps is None


def test_ecd_ground_overlap_scores_only_the_ground_block():
    seq = sq.ecd_sequence(2, N_FOCK)
    psi0 = _low_kets(11)
    psi = seq.propagate(_ecd_flat(*_ecd_params(2, seed=31)), seq.lift(jnp.asarray(psi0)))[0]
    targets = jnp.asarray(_low_kets(12))
    expected = np.einsum("kd,kd->k", np.conj(targets), np.asarray(psi)[:, 0, :])
    np.testing.assert_allclose(seq.overlaps(psi, targets), expected, atol=1e-14)


# ---------------------------------------------------------------------------
# SNAP
# ---------------------------------------------------------------------------


def _snap_reference(phases, alphas, psi0) -> np.ndarray:
    """Dense ``D S D S ... D`` applied to row kets, phases zero-padded to N_FOCK."""
    out = np.asarray(psi0, dtype=complex) @ _ref_displacement(alphas[0]).T
    for theta, alpha in zip(phases, alphas[1:], strict=True):
        diag = np.ones(N_FOCK, dtype=complex)
        diag[: len(theta)] = np.exp(1j * np.asarray(theta))
        out = (out * diag) @ _ref_displacement(alpha).T
    return out


@pytest.mark.parametrize("method", DISP_METHODS)
@pytest.mark.parametrize("n_gates", [1, 2, 3])
def test_snap_sequence_matches_dense_product(n_gates, method):
    phases, alphas = _snap_params(n_gates, seed=n_gates)
    seq = sq.snap_sequence(n_gates, N_FOCK, disp_method=method)
    psi0 = _low_kets(40)
    out = _run(seq, _snap_flat(phases, alphas), psi0)
    assert out.shape == psi0.shape
    np.testing.assert_allclose(out, _snap_reference(phases, alphas, psi0), atol=1e-10)


@pytest.mark.parametrize("method", DISP_METHODS)
def test_snap_gate_is_diagonal_phase(method):
    """With zero displacements a SNAP multiplies each Fock amplitude by its phase."""
    phases, _ = _snap_params(1, seed=41)
    seq = sq.snap_sequence(1, N_FOCK, disp_method=method)
    rng = np.random.default_rng(42)
    psi0 = rng.normal(size=(2, N_FOCK)) + 1j * rng.normal(size=(2, N_FOCK))
    psi0 /= np.linalg.norm(psi0, axis=-1, keepdims=True)
    out = _run(seq, _snap_flat(phases, [0.0, 0.0]), psi0)
    np.testing.assert_allclose(out, psi0 * np.exp(1j * phases[0]), atol=1e-12)
    np.testing.assert_allclose(np.abs(out) ** 2, np.abs(psi0) ** 2, atol=1e-12)


def test_truncated_snap_equals_zero_padded_full_snap():
    n_gates, n_snap = 2, 6
    phases, alphas = _snap_params(n_gates, n_snap=n_snap, seed=43)
    padded = np.zeros((n_gates, N_FOCK))
    padded[:, :n_snap] = phases
    psi0 = _low_kets(44)

    short = sq.snap_sequence(n_gates, N_FOCK, n_snap=n_snap)
    full = sq.snap_sequence(n_gates, N_FOCK)
    assert short.n_params == n_gates * n_snap + 2 * (n_gates + 1)
    assert full.n_params == n_gates * N_FOCK + 2 * (n_gates + 1)
    np.testing.assert_allclose(
        _run(short, _snap_flat(phases, alphas), psi0),
        _run(full, _snap_flat(padded, alphas), psi0),
        atol=1e-12,
    )


@pytest.mark.parametrize("method", DISP_METHODS)
def test_snap_with_zero_phases_collapses_to_one_displacement(method):
    n_gates = 3
    _, alphas = _snap_params(n_gates, seed=45)
    seq = sq.snap_sequence(n_gates, N_FOCK, disp_method=method)
    psi0 = _low_kets(46)
    out = _run(seq, _snap_flat(np.zeros((n_gates, N_FOCK)), alphas), psi0)
    total, phase = _displacement_product(alphas)
    expected = np.exp(1j * phase) * (psi0 @ _ref_displacement(total).T)
    np.testing.assert_allclose(out, expected, atol=1e-10)


@pytest.mark.parametrize("method", DISP_METHODS)
def test_snap_history_matches_gate_by_gate_reference(method):
    n_gates = 2
    phases, alphas = _snap_params(n_gates, seed=47)
    seq = sq.snap_sequence(n_gates, N_FOCK, disp_method=method)
    flat = _snap_flat(phases, alphas)
    psi0 = _low_kets(48)
    hist = np.asarray(seq.history(flat, seq.lift(jnp.asarray(psi0))))
    assert hist.shape == (2 * n_gates + 2, len(psi0), N_FOCK)
    np.testing.assert_allclose(hist[0], psi0, atol=1e-14)
    np.testing.assert_allclose(hist[-1], _run(seq, flat, psi0), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(hist, axis=-1), 1.0, atol=1e-12)
    # Odd entries follow a displacement, even entries follow a SNAP.
    for i in range(n_gates + 1):
        expected = _snap_reference(phases[:i], alphas[: i + 1], psi0)
        np.testing.assert_allclose(hist[2 * i + 1], expected, atol=1e-10)
    for i in range(n_gates):
        np.testing.assert_allclose(
            hist[2 * i + 2], hist[2 * i + 1] * np.exp(1j * phases[i]), atol=1e-12
        )


def test_snap_parameter_layout():
    n_gates, n_snap = 2, 6
    phases, alphas = _snap_params(n_gates, n_snap=n_snap, seed=49)
    seq = sq.snap_sequence(n_gates, N_FOCK, n_snap=n_snap)
    flat = _snap_flat(phases, alphas)
    assert seq.n_params == flat.shape[0]
    assert seq.init(jax.random.PRNGKey(0)).shape == (seq.n_params,)
    assert not seq.has_qubit
    assert seq.rotation_angles(flat).shape == (0,)

    described = seq.describe(flat)
    np.testing.assert_allclose(described["snap_phases"], phases)
    np.testing.assert_allclose(described["alphas"], alphas)

    psi0 = jnp.asarray(_low_kets(50))
    psi, _, strengths, amps = seq.propagate(flat, seq.lift(psi0))
    np.testing.assert_allclose(strengths, np.abs(alphas), atol=1e-14)
    assert amps is None

    targets = jnp.asarray(_low_kets(51))
    expected = np.einsum("kd,kd->k", np.conj(targets), np.asarray(psi))
    np.testing.assert_allclose(seq.overlaps(psi, targets), expected, atol=1e-14)


@pytest.mark.parametrize("bad", [0, N_FOCK + 1])
def test_snap_rejects_out_of_range_phase_count(bad):
    with pytest.raises(ValueError, match="n_snap"):
        sq.snap_sequence(1, N_FOCK, n_snap=bad)
