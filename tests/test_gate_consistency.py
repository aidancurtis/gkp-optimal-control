"""Consistency of the ECD and SNAP gate sets across their implementations.

The same circuits are implemented several times: the JAX block-basis
propagators in :mod:`gate_optimization` (three displacement methods each), the
NumPy mirror and branch expansion in :mod:`ecd_symbolic`, and the conventions
written in the module docstrings. These tests pin them to each other and to
independent dense references built here with ``scipy.linalg.expm``.

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
from gkp_optimal_control import ecd_symbolic as es
from gkp_optimal_control import gate_optimization as go

N_FOCK = 48
LOW = 16
DISP_METHODS = ("expm", "quadrature", "fixed")
ALPHA = 0.6 - 0.4j

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


# ---------------------------------------------------------------------------
# Independent references and parameter helpers
# ---------------------------------------------------------------------------


def _ref_displacement(alpha: complex, n_fock: int = N_FOCK) -> np.ndarray:
    a = np.diag(np.sqrt(np.arange(1, n_fock)), 1).astype(complex)
    return expm(alpha * a.conj().T - np.conj(alpha) * a)


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


@pytest.mark.parametrize("method", ["expm", "quadrature"])
def test_displacement_is_unitary(method):
    d_op = np.asarray(go.make_displacement(N_FOCK, method)(ALPHA))
    np.testing.assert_allclose(d_op.conj().T @ d_op, np.eye(N_FOCK), atol=1e-12)


def test_fast_displacement_is_unitary():
    d_op = es.fast_displacement(N_FOCK)(ALPHA)
    np.testing.assert_allclose(d_op.conj().T @ d_op, np.eye(N_FOCK), atol=1e-12)


def test_displacement_builders_match_reference():
    ref = _ref_displacement(ALPHA)
    d_expm = np.asarray(go.make_displacement(N_FOCK, "expm")(ALPHA))
    d_quad = np.asarray(go.make_displacement(N_FOCK, "quadrature")(ALPHA))
    d_fast = es.fast_displacement(N_FOCK)(ALPHA)
    np.testing.assert_allclose(d_expm, ref, atol=1e-12)
    # Same truncated generator, different algorithm.
    np.testing.assert_allclose(d_fast, ref, atol=1e-8)
    # Different factorization: equal only away from the cutoff.
    np.testing.assert_allclose(d_quad[:LOW, :LOW], ref[:LOW, :LOW], atol=1e-12)


def test_fixed_apply_matches_quadrature_matrix():
    d_quad = np.asarray(go.make_displacement(N_FOCK, "quadrature")(ALPHA))
    apply = go.make_displacement_apply(N_FOCK)
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(3, N_FOCK)) + 1j * rng.normal(size=(3, N_FOCK))
    np.testing.assert_allclose(apply(jnp.asarray(psi), ALPHA), psi @ d_quad.T, atol=1e-12)
    np.testing.assert_allclose(
        apply(jnp.asarray(psi), ALPHA, adjoint=True), psi @ d_quad.conj(), atol=1e-12
    )


@pytest.mark.parametrize("method", ["expm", "quadrature"])
def test_displaced_vacuum_is_coherent_state(method):
    d_op = np.asarray(go.make_displacement(N_FOCK, method)(ALPHA))
    n = np.arange(LOW)
    log_fact = np.cumsum(np.log(np.maximum(n, 1)))
    coherent = np.exp(-abs(ALPHA) ** 2 / 2) * ALPHA**n / np.exp(0.5 * log_fact)
    np.testing.assert_allclose(d_op[:LOW, 0], coherent, atol=1e-12)


def test_displacement_product_matches_matrix_product():
    alphas = [0.3 + 0.2j, -0.25 + 0.4j, 0.1 - 0.35j]
    total, phase = es.displacement_product(alphas)
    product = np.eye(N_FOCK, dtype=complex)
    for alpha in alphas:  # D(alpha_n) ... D(alpha_1)
        product = _ref_displacement(alpha) @ product
    merged = np.exp(1j * phase) * _ref_displacement(total)
    np.testing.assert_allclose(product[:LOW, :LOW], merged[:LOW, :LOW], atol=1e-12)


# ---------------------------------------------------------------------------
# Qubit rotations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta, phi", [(0.0, 0.3), (np.pi / 2, 0.0), (1.1, -2.0), (np.pi, 0.7)])
def test_qubit_rotation_conventions_agree(theta, phi):
    ref = expm(-0.5j * theta * (np.cos(phi) * SIGMA_X + np.sin(phi) * SIGMA_Y))
    np.testing.assert_allclose(np.asarray(go.qubit_rotation(theta, phi)), ref, atol=1e-14)
    np.testing.assert_allclose(es.qubit_rotation(theta, phi), ref, atol=1e-14)


# ---------------------------------------------------------------------------
# ECD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", DISP_METHODS)
def test_ecd_gate_matches_definition(method):
    """ECD(beta) = D(beta/2)|e><g| + D(-beta/2)|g><e|, with identity rotations."""
    beta = 0.7 - 0.5j
    seq = go.ecd_sequence(1, N_FOCK, disp_method=method)
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
    echoed = _run(go.ecd_sequence(1, N_FOCK, disp_method=method, echoed=True), flat, psi0)
    bare = _run(go.ecd_sequence(1, N_FOCK, disp_method=method, echoed=False), flat, psi0)
    np.testing.assert_allclose(echoed, bare[:, ::-1, :], atol=1e-12)


@pytest.mark.parametrize("method", DISP_METHODS)
@pytest.mark.parametrize("n_gates", [1, 2, 4])
def test_ecd_sequence_matches_numpy_mirror(n_gates, method):
    thetas, phis, betas = _ecd_params(n_gates, seed=n_gates)
    seq = go.ecd_sequence(n_gates, N_FOCK, disp_method=method)
    psi0 = _low_kets(5)
    out = _run(seq, _ecd_flat(thetas, phis, betas), psi0)
    assert out.shape == (len(psi0), 2, N_FOCK)
    for k, ket in enumerate(psi0):
        mirror = es.ecd_propagate(thetas, phis, betas, N_FOCK, psi0=ket)
        np.testing.assert_allclose(out[k], mirror, atol=1e-10)


@pytest.mark.parametrize("method", DISP_METHODS)
def test_ecd_history_matches_mirror_and_conserves_norm(method):
    n_gates = 3
    thetas, phis, betas = _ecd_params(n_gates, seed=6)
    seq = go.ecd_sequence(n_gates, N_FOCK, disp_method=method)
    flat = _ecd_flat(thetas, phis, betas)
    psi0 = _low_kets(7)
    hist = np.asarray(seq.history(flat, seq.lift(jnp.asarray(psi0))))
    assert hist.shape == (2 * n_gates + 2, len(psi0), 2, N_FOCK)
    np.testing.assert_allclose(hist[-1], _run(seq, flat, psi0), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(hist, axis=(-2, -1)), 1.0, atol=1e-12)
    for k, ket in enumerate(psi0):
        _, mirror = es.ecd_propagate(thetas, phis, betas, N_FOCK, psi0=ket, return_history=True)
        np.testing.assert_allclose(hist[:, k], mirror, atol=1e-10)


@pytest.mark.parametrize("n_gates", [1, 3])
def test_ecd_full_unitary_is_unitary_and_matches_propagation(n_gates):
    thetas, phis, betas = _ecd_params(n_gates, seed=10 + n_gates)
    unitary = es.ecd_full_unitary(thetas, phis, betas, N_FOCK)
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(2 * N_FOCK), atol=1e-12)
    for ket in _low_kets(8):
        joint = np.kron(np.array([1, 0], dtype=complex), ket)  # kron(qubit, cavity)
        expected = es.ecd_propagate(thetas, phis, betas, N_FOCK, psi0=ket)
        np.testing.assert_allclose((unitary @ joint).reshape(2, N_FOCK), expected, atol=1e-10)


@pytest.mark.parametrize("n_gates", [1, 2, 3])
def test_branch_expansion_reconstructs_unitary(n_gates):
    thetas, phis, betas = _ecd_params(n_gates, seed=20 + n_gates)
    branches = es.branch_expansion(thetas, phis, betas)
    assert len(branches) == 2**n_gates

    unitary = es.ecd_full_unitary(thetas, phis, betas, N_FOCK).reshape(2, N_FOCK, 2, N_FOCK)
    rebuilt = es.reconstruct_unitary(branches, N_FOCK).reshape(2, N_FOCK, 2, N_FOCK)
    np.testing.assert_allclose(
        rebuilt[:, :LOW, :, :LOW], unitary[:, :LOW, :, :LOW], atol=1e-10
    )

    projected = es.reconstruct_projected(branches, N_FOCK)
    np.testing.assert_allclose(projected[:LOW, :LOW], unitary[0, :LOW, 0, :LOW], atol=1e-10)

    # <g|U|g>|0> in closed form vs. the JAX propagator from vacuum.
    seq = go.ecd_sequence(n_gates, N_FOCK)
    vac = np.zeros((1, N_FOCK), dtype=complex)
    vac[0, 0] = 1.0
    psi_g = _run(seq, _ecd_flat(thetas, phis, betas), vac)[0, 0]
    np.testing.assert_allclose(es.fock_coefficients(branches, max_n=8), psi_g[:8], atol=1e-10)


def test_ecd_parameter_layout():
    n_gates = 3
    thetas, phis, betas = _ecd_params(n_gates, seed=30)
    seq = go.ecd_sequence(n_gates, N_FOCK)
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
    seq = go.ecd_sequence(2, N_FOCK)
    psi0 = _low_kets(11)
    psi = seq.propagate(_ecd_flat(*_ecd_params(2, seed=31)), seq.lift(jnp.asarray(psi0)))[0]
    targets = jnp.asarray(_low_kets(12))
    expected = np.einsum("kd,kd->k", np.conj(targets), np.asarray(psi)[:, 0, :])
    np.testing.assert_allclose(seq.overlaps(psi, targets), expected, atol=1e-14)


def test_to_joint_ket_is_cavity_first():
    rng = np.random.default_rng(13)
    blocks = rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5))
    joint = go.to_joint_ket(blocks)
    assert joint.shape == (10,)
    for n_cav in range(5):
        for s_qubit in range(2):
            assert joint[2 * n_cav + s_qubit] == blocks[s_qubit, n_cav]


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
    seq = go.snap_sequence(n_gates, N_FOCK, disp_method=method)
    psi0 = _low_kets(40)
    out = _run(seq, _snap_flat(phases, alphas), psi0)
    assert out.shape == psi0.shape
    np.testing.assert_allclose(out, _snap_reference(phases, alphas, psi0), atol=1e-10)


@pytest.mark.parametrize("method", DISP_METHODS)
def test_snap_gate_is_diagonal_phase(method):
    """With zero displacements a SNAP multiplies each Fock amplitude by its phase."""
    phases, _ = _snap_params(1, seed=41)
    seq = go.snap_sequence(1, N_FOCK, disp_method=method)
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

    short = go.snap_sequence(n_gates, N_FOCK, n_snap=n_snap)
    full = go.snap_sequence(n_gates, N_FOCK)
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
    seq = go.snap_sequence(n_gates, N_FOCK, disp_method=method)
    psi0 = _low_kets(46)
    out = _run(seq, _snap_flat(np.zeros((n_gates, N_FOCK)), alphas), psi0)
    total, phase = es.displacement_product(alphas)
    expected = np.exp(1j * phase) * (psi0 @ _ref_displacement(total).T)
    np.testing.assert_allclose(out, expected, atol=1e-10)


@pytest.mark.parametrize("method", DISP_METHODS)
def test_snap_history_matches_gate_by_gate_reference(method):
    n_gates = 2
    phases, alphas = _snap_params(n_gates, seed=47)
    seq = go.snap_sequence(n_gates, N_FOCK, disp_method=method)
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
    seq = go.snap_sequence(n_gates, N_FOCK, n_snap=n_snap)
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
        go.snap_sequence(1, N_FOCK, n_snap=bad)
