"""Pulse compilation: configs, waveforms, ECD/SNAP compilers, and simulation.

Three layers are covered:

* unit-level pieces of :mod:`device` and :mod:`pulses` (unit parsing, envelopes, rotation
  waveforms, the semiclassical trajectory integrator, sequence bookkeeping),
  checked against closed forms;
* the ECD compiler (:func:`pulses.optimize_ecd_pulse`), checked against its
  own contract (target ``beta``, nulled residual displacement);
* end-to-end runs through :func:`pulse_simulation.simulate_sequence`, compared
  with the ideal circuits of :mod:`gate_optimization`.

Compiling an ECD pulse costs a few seconds, so the two pulses used by the
simulation tests are module-scoped fixtures and are reused for every gate of a
sequence by phase rotation (the same trick ``ecd_pulse_cache`` relies on).
"""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control import device
from gkp_optimal_control import operators as op
from gkp_optimal_control import pulse_simulation as ps
from gkp_optimal_control import pulses as pl
from gkp_optimal_control import sequences as sq

DEVICES = Path(__file__).resolve().parents[1] / "devices"
TWO_PI = 2 * np.pi
LINEAR = {"kappa": 0.0, "chi_prime": 0.0, "kerr": 0.0}

DEFAULT = device.SystemParams()
# A stiffer device than Table S1: with chi = 1 MHz the qubit phase theta' picked
# up by an ECD gate is ~0.3 rad instead of ~0.01, so sign conventions in the
# virtual-Z bookkeeping become visible in simulation.
FAST = DEFAULT.replace(chi=TWO_PI * 1.0, **LINEAR)

THETAS = np.array([1.9, 1.1, 2.3])
PHIS = np.array([0.4, -1.7, 2.2])
BETA_PHASES = np.array([0.5, 2.4])


def _displace(n_fock: int, alpha: complex) -> np.ndarray:
    a = np.diag(np.sqrt(np.arange(1, n_fock)), 1).astype(complex)
    return expm(alpha * a.conj().T - np.conj(alpha) * a)


def _vacuum(n_fock: int) -> np.ndarray:
    vac = np.zeros(n_fock, dtype=complex)
    vac[0] = 1.0
    return vac


def _ideal_ecd(thetas, phis, betas, n_fock: int) -> np.ndarray:
    """Final ``(2, n_fock)`` block state of the ideal ECD circuit from ``|g>|0>``."""
    seq = sq.ecd_sequence(len(betas), n_fock, disp_method="exact")
    betas = np.asarray(betas, dtype=complex)
    flat = np.concatenate(
        [np.stack([thetas, phis], -1).ravel(), np.stack([betas.real, betas.imag], -1).ravel()]
    )
    return np.asarray(seq.propagate(flat, seq.lift(_vacuum(n_fock)[None]))[0])[0]


def _assert_params_close(got: device.SystemParams, want: device.SystemParams):
    for name, value in want.to_dict().items():
        assert getattr(got, name) == pytest.approx(value, rel=1e-9), name


# ---------------------------------------------------------------------------
# Units and configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, kind, expected",
    [
        ("32.8 kHz", "frequency", TWO_PI * 32.8e-3),
        ("193 MHz", "frequency", TWO_PI * 193.0),
        ("1.5 Hz", "frequency", TWO_PI * 1.5e-6),
        ("2 rad/us", "frequency", 2.0),
        ("11 ns", "time", 0.011),
        ("436 µs", "time", 436.0),
        ("1 MHz", "rate", 1.0),
        ("32.8 khz", "frequency", TWO_PI * 32.8e-3),
        (0.25, "time", 0.25),
        (3, "frequency", 3.0),
    ],
)
def test_parse_quantity(value, kind, expected):
    assert device.parse_quantity(value, kind) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "value, kind, error",
    [
        ("5", "time", ValueError),  # no unit
        ("5 parsecs", "time", ValueError),
        ("fast", "time", ValueError),
        ("5 ns", "length", ValueError),
        (True, "time", TypeError),
        (None, "time", TypeError),
    ],
)
def test_parse_quantity_rejects_bad_input(value, kind, error):
    with pytest.raises(error):
        device.parse_quantity(value, kind)


def test_from_dict_applies_units_and_aliases():
    params = device.SystemParams.from_dict(
        {
            "chi": "40 kHz",
            "two_kerr": "1.0 Hz",  # halved into K_c
            "t1_cavity": "500 us",  # inverted into kappa
            "K_q": "200 MHz",
            "n_sigma_disp": 6,
            "subtract_pedestal": False,
        },
        dt="2 ns",
    )
    assert params.chi == pytest.approx(TWO_PI * 40e-3)
    assert params.kerr == pytest.approx(TWO_PI * 0.5e-6)
    assert params.kappa == pytest.approx(1 / 500.0)
    assert params.anharm == pytest.approx(TWO_PI * 200.0)
    assert params.n_sigma_disp == 6
    assert params.subtract_pedestal is False
    assert params.dt == pytest.approx(2e-3)


def test_from_dict_unknown_key_is_error_unless_lenient():
    with pytest.raises(ValueError, match="unknown parameter"):
        device.SystemParams.from_dict({"qubit_t2": "30 us"})
    with pytest.warns(UserWarning, match="qubit_t2"):
        params = device.SystemParams.from_dict({"qubit_t2": "30 us", "chi": "40 kHz"}, strict=False)
    assert params.chi == pytest.approx(TWO_PI * 40e-3)


def test_to_config_round_trips():
    params = DEFAULT.replace(chi=TWO_PI * 0.7, n_selective_periods=6, subtract_pedestal=False)
    _assert_params_close(device.SystemParams.from_dict(params.to_config()["system"]), params)


@pytest.mark.parametrize("ext", [".toml", ".json"])
def test_config_file_round_trips(tmp_path, ext):
    params = DEFAULT.replace(chi=TWO_PI * 0.7, dt=2e-3)
    path = tmp_path / f"device{ext}"
    params.to_config(str(path))
    loaded, extras = device.load_config(str(path))
    _assert_params_close(loaded, params)
    assert extras == {}
    _assert_params_close(
        device.SystemParams.from_config(str(path), dt="4 ns"), params.replace(dt=4e-3)
    )


def test_flat_config_splits_parameters_from_extras(tmp_path):
    path = tmp_path / "flat.json"
    path.write_text(json.dumps({"chi": "40 kHz", "ecd": {"alpha0": 5}, "note": "x"}))
    params, extras = device.load_config(str(path))
    assert params.chi == pytest.approx(TWO_PI * 40e-3)
    assert extras == {"ecd": {"alpha0": 5}, "note": "x"}


def test_unsupported_config_extension(tmp_path):
    path = tmp_path / "device.ini"
    path.write_text("")
    with pytest.raises(ValueError, match="unsupported config extension"):
        device.load_config(str(path))


def test_shipped_device_files_load():
    eickbusch, extras = device.load_config(str(DEVICES / "device_eickbusch.toml"))
    _assert_params_close(eickbusch, DEFAULT)  # both are Table S1
    assert "ecd" in extras

    snap, extras = device.load_config(str(DEVICES / "device_snap.toml"))
    assert snap.chi == pytest.approx(TWO_PI * 2.0)
    assert snap.t_snap_selective == pytest.approx(snap.n_selective_periods * TWO_PI / snap.chi)
    assert "snap" in extras


# ---------------------------------------------------------------------------
# Envelopes and rotation waveforms
# ---------------------------------------------------------------------------


def test_gaussian_envelope_shape_and_area():
    g, area = pl.gaussian_envelope(11e-3, 4, 1e-3)
    assert g.size == 44
    np.testing.assert_allclose(g, g[::-1], atol=1e-15)
    assert g.min() >= 0.0 and g.argmax() in (21, 22)
    assert area == pytest.approx(g.sum() * 1e-3)

    raw, _ = pl.gaussian_envelope(11e-3, 4, 1e-3, subtract_pedestal=False)
    np.testing.assert_allclose(raw - g, np.exp(-0.5 * 2.0**2), atol=1e-15)


def test_gaussian_envelope_returns_a_private_copy():
    g, _ = pl.gaussian_envelope(11e-3, 4, 1e-3)
    g[:] = 0.0
    again, _ = pl.gaussian_envelope(11e-3, 4, 1e-3)
    assert again.max() > 0.5


def test_gaussian_envelope_rejects_undersampled_pulse():
    with pytest.raises(ValueError, match="under-sampled"):
        pl.gaussian_envelope(1e-3, 1, 1e-3)


@pytest.mark.parametrize("theta, phi", [(np.pi, 0.0), (np.pi / 2, 0.9), (1.1, -2.0)])
def test_rotation_waveform_realizes_the_ideal_rotation(theta, phi):
    omega = pl.rotation_waveform(theta, phi, DEFAULT)
    assert omega.size == round(DEFAULT.t_qubit / DEFAULT.dt)
    assert omega.sum() * DEFAULT.dt == pytest.approx(0.5 * theta * np.exp(1j * phi))

    # Bare qubit under Omega^* q + Omega q^dag with q = |g><e|, basis (g, e).
    unitary = np.eye(2, dtype=complex)
    for om in omega:
        unitary = expm(-1j * DEFAULT.dt * np.array([[0, np.conj(om)], [om, 0]])) @ unitary
    np.testing.assert_allclose(unitary, np.asarray(op.qubit_rotation(theta, phi)), atol=1e-12)


# ---------------------------------------------------------------------------
# Semiclassical trajectories
# ---------------------------------------------------------------------------


def test_displacement_pulse_area_sets_the_displacement():
    """d(alpha)/dt = -i eps, so eps = i alpha g / area realizes D(alpha)."""
    g, area = pl.gaussian_envelope(DEFAULT.sigma_disp, DEFAULT.n_sigma_disp, DEFAULT.dt)
    alpha = 0.7 - 0.2j
    free = DEFAULT.replace(chi=0.0, **LINEAR)
    traj = pl.frame_trajectory((1j * alpha / area) * g, free, delta=0.0)
    assert traj.shape == (g.size + 1,)
    assert traj[0] == 0
    assert traj[-1] == pytest.approx(alpha, abs=1e-12)


def test_undriven_trajectories_match_closed_forms():
    n, alpha0, delta = 400, 1.0 + 0.5j, 0.4
    t_total = n * DEFAULT.dt
    eps = np.zeros(n)

    params = DEFAULT.replace(chi=0.9, kappa=0.3, chi_prime=0.0, kerr=0.0)
    both = pl._integrate_alpha(eps, np.stack([np.zeros(n), np.ones(n)], 1), params, delta, alpha0)
    assert both.shape == (n + 1, 2)
    decay = np.exp(-0.5 * params.kappa * t_total)
    assert both[-1, 0] == pytest.approx(alpha0 * decay * np.exp(-1j * delta * t_total), abs=1e-12)
    assert both[-1, 1] == pytest.approx(
        alpha0 * decay * np.exp(1j * (params.chi - delta) * t_total), abs=1e-12
    )

    kerr_only = DEFAULT.replace(chi=0.0, kappa=0.0, chi_prime=0.0, kerr=0.05)
    traj = pl._integrate_alpha(eps, np.zeros(n), kerr_only, 0.0, alpha0)
    phase = 2.0 * kerr_only.kerr * abs(alpha0) ** 2 * t_total
    assert traj[-1] == pytest.approx(alpha0 * np.exp(1j * phase), abs=1e-12)


def test_coarsened_integration_preserves_pulse_area():
    g, area = pl.gaussian_envelope(DEFAULT.sigma_disp, DEFAULT.n_sigma_disp, DEFAULT.dt)
    eps = (1j * 2.0 / area) * g
    q_g = np.zeros(g.size)
    fine, _ = pl.branch_trajectories(eps, q_g, DEFAULT, 0.5 * DEFAULT.chi)
    coarse, _ = pl.branch_trajectories(eps, q_g, DEFAULT, 0.5 * DEFAULT.chi, stride=2)
    assert coarse.size == g.size // 2 + 1
    assert coarse[-1] == pytest.approx(fine[-1], rel=1e-3)


# ---------------------------------------------------------------------------
# ECD pulse construction
# ---------------------------------------------------------------------------


def test_ecd_waveform_layout():
    params = DEFAULT
    g, _ = pl.gaussian_envelope(params.sigma_disp, params.n_sigma_disp, params.dt)
    n_g, n_w = g.size, 10
    n_q = round(params.t_qubit / params.dt)
    eps0, r2, r3, r4 = 3.0, 0.9, 1.1, 1.2
    eps, omega, q_g = pl._ecd_waveforms([eps0, r2, r3, r4], n_w * params.dt, params)

    assert eps.size == omega.size == q_g.size == 4 * n_g + 2 * n_w + n_q
    bounds = np.cumsum([0, n_g, n_w, n_g, n_q, n_g, n_w, n_g])
    pieces = [eps[a:b] for a, b in itertools.pairwise(bounds)]
    for piece, scale in zip(pieces[::2], [eps0, -eps0 * r2, -eps0 * r3, eps0 * r4], strict=True):
        np.testing.assert_allclose(piece, scale * g)
    for gap in pieces[1::2]:
        assert not gap.any()

    pi_window = slice(bounds[3], bounds[4])
    np.testing.assert_allclose(omega[pi_window], pl.rotation_waveform(np.pi, 0.0, params))
    assert not np.delete(omega, np.r_[pi_window]).any()

    # The modelled qubit flip sits at the centre of the pi pulse.
    i_flip = bounds[3] + n_q // 2
    assert not q_g[:i_flip].any() and q_g[i_flip:].all()


@pytest.fixture(scope="module")
def default_pulse() -> pl.ECDPulse:
    """Table S1 device, alpha0 = 30: the wait-time-limited regime."""
    return pl.optimize_ecd_pulse(1.0 + 0.5j, 30.0, DEFAULT)


@pytest.fixture(scope="module")
def fast_pulse() -> pl.ECDPulse:
    """chi = 1 MHz device: drive-limited, with a large qubit phase theta'."""
    return pl.optimize_ecd_pulse(2.0, 8.0, FAST)


def test_ecd_pulse_hits_target_and_nulls_residual(default_pulse):
    pulse, target = default_pulse, 1.0 + 0.5j
    assert abs(pulse.beta) == pytest.approx(abs(target), rel=1e-4)
    assert np.angle(pulse.beta) == pytest.approx(np.angle(target), abs=1e-12)
    assert abs(pulse.lam) < 1e-4

    assert pulse.eps.size == pulse.omega.size
    assert pulse.alpha_g.size == pulse.alpha_e.size == pulse.eps.size + 1
    assert pulse.beta == pytest.approx(pulse.alpha_g[-1] - pulse.alpha_e[-1])
    assert pulse.lam == pytest.approx(0.5 * (pulse.alpha_g[-1] + pulse.alpha_e[-1]))
    assert pulse.omega.sum() * DEFAULT.dt == pytest.approx(np.pi / 2)  # one pi pulse

    # Wait-time-limited: the excursion reaches the requested radius.
    assert pulse.t_wait > 0
    assert pulse.alpha0 == pytest.approx(30.0, rel=0.02)
    radius = 0.5 * np.abs(pulse.alpha_g + pulse.alpha_e).max()
    assert radius == pytest.approx(pulse.alpha0, rel=0.02)
    assert np.abs(pulse.eps).max() <= DEFAULT.eps_max


def test_ecd_pulse_lowers_radius_when_drive_limited(fast_pulse):
    assert abs(fast_pulse.beta) == pytest.approx(2.0, rel=1e-4)
    assert abs(fast_pulse.lam) < 1e-4
    assert fast_pulse.t_wait == 0.0
    assert fast_pulse.alpha0 < 8.0


def test_ecd_pulse_rejects_zero_target():
    with pytest.raises(ValueError, match="nonzero"):
        pl.optimize_ecd_pulse(0.0, 30.0, DEFAULT)


def test_phase_rotation_is_a_symmetry_of_the_trajectories(default_pulse):
    target = abs(default_pulse.beta) * np.exp(2.1j)
    rotated = pl._phase_rotate(default_pulse, target)
    assert rotated.beta == pytest.approx(target)
    assert rotated.theta_prime == default_pulse.theta_prime
    np.testing.assert_array_equal(rotated.omega, default_pulse.omega)

    # Re-integrating the rotated drive reproduces the rotated bookkeeping.
    _, _, q_g = pl._ecd_waveforms(default_pulse.ratios, default_pulse.t_wait, DEFAULT)
    a_g, a_e = pl.branch_trajectories(rotated.eps, q_g, DEFAULT, 0.5 * DEFAULT.chi)
    beta, lam, theta_prime = pl._geometric_phase(rotated.eps, a_g, a_e, DEFAULT)
    assert beta == pytest.approx(target, abs=1e-9)
    assert abs(lam) == pytest.approx(abs(default_pulse.lam), abs=1e-9)
    assert theta_prime == pytest.approx(default_pulse.theta_prime, abs=1e-9)


def test_ecd_pulse_cache_compiles_each_magnitude_once(monkeypatch, default_pulse):
    calls = []

    def fake_optimize(beta, alpha0, params, delta=None, **kwargs):
        calls.append(abs(beta))
        scale = abs(beta) / abs(default_pulse.beta)
        return pl._phase_rotate(
            pl.ECDPulse(**{**vars(default_pulse), "beta": default_pulse.beta * scale}), abs(beta)
        )

    monkeypatch.setattr(pl, "optimize_ecd_pulse", fake_optimize)
    get = pl.ecd_pulse_cache(DEFAULT, 30.0)
    for beta in (1.0, 1j, -0.6 + 0.8j, 2.0, -2.0):
        assert get(beta).beta == pytest.approx(beta)
    assert calls == [1.0, 2.0]


# ---------------------------------------------------------------------------
# ECD sequence assembly and virtual-Z bookkeeping
# ---------------------------------------------------------------------------


def _stub_pulses(monkeypatch, theta_primes, n_samples: int = 8):
    """Make compile_ecd_sequence use blank pulses with prescribed theta'."""
    remaining = iter(theta_primes)

    def factory(params, alpha0, delta=None, **kwargs):
        def get(beta):
            blank = np.zeros(n_samples, dtype=complex)
            return pl.ECDPulse(
                eps=blank, omega=blank, beta=complex(beta), lam=0j,
                theta_prime=next(remaining), alpha0=alpha0, t_wait=0.0,
                ratios=np.ones(4), cost=0.0,
            )  # fmt: skip

        return get

    monkeypatch.setattr(pl, "ecd_pulse_cache", factory)


def _reuse_pulse(monkeypatch, pulse: pl.ECDPulse):
    """Make compile_ecd_sequence phase-rotate one precompiled pulse per gate."""
    monkeypatch.setattr(
        pl, "ecd_pulse_cache", lambda *a, **k: lambda beta: pl._phase_rotate(pulse, beta)
    )


def test_ecd_sequence_layout_and_frame_updates(monkeypatch):
    theta_primes = [0.7, -1.3]
    betas = np.array([0.8 + 0.4j, -0.5 + 0.9j])
    _stub_pulses(monkeypatch, theta_primes)
    seq = pl.compile_ecd_sequence(betas, THETAS, PHIS, params=DEFAULT)

    assert [s.kind for s in seq.segments] == ["rotation", "ecd", "rotation", "ecd", "rotation"]
    assert seq.segments[0].start == 0 and seq.segments[-1].stop == seq.n_samples
    for left, right in itertools.pairwise(seq.segments):
        assert left.stop == right.start
    np.testing.assert_array_equal(seq.gate_indices, [s.stop for s in seq.segments])
    n_rot = round(DEFAULT.t_qubit / DEFAULT.dt)
    assert seq.n_samples == 3 * n_rot + 2 * 8
    assert seq.duration == pytest.approx(seq.n_samples * DEFAULT.dt)
    assert seq.delta == pytest.approx(0.5 * DEFAULT.chi)

    # Virtual Z: phi_i -> phi_i - F, then F -> -F - theta'_i.
    frame = 0.0
    rotations = [s for s in seq.segments if s.kind == "rotation"]
    for i, seg in enumerate(rotations):
        assert seg.meta["frame"] == pytest.approx(frame)
        assert seg.meta["phi"] == pytest.approx(PHIS[i] - frame)
        assert seg.meta["phi_ideal"] == pytest.approx(PHIS[i])
        np.testing.assert_allclose(
            seq.omega[seg.slice()], pl.rotation_waveform(THETAS[i], PHIS[i] - frame, DEFAULT)
        )
        assert not seq.eps[seg.slice()].any()
        if i < len(theta_primes):
            frame = -frame - theta_primes[i]

    assert seq.meta["gate_set"] == "ecd" and seq.meta["n_ecd"] == 2
    np.testing.assert_allclose(seq.meta["betas_realized"], betas)
    np.testing.assert_allclose(seq.meta["theta_primes"], theta_primes)


def _circuit_with_leftover_z(seq: pl.PulseSequence, betas, z_angles, n_fock: int) -> np.ndarray:
    """Apply the compiled rotations around gates ``ECD(beta_i) Z(z_i)``.

    ``Z(a) = diag(e^{ia/2}, e^{-ia/2})`` in the (g, e) basis. Returns the final
    ``(2, n_fock)`` block state from ``|g>|0>``.
    """
    rotations = [s.meta for s in seq.segments if s.kind == "rotation"]
    psi = np.stack([_vacuum(n_fock), np.zeros(n_fock, dtype=complex)])
    for meta, beta, angle in zip(rotations, betas, z_angles, strict=False):
        psi = np.asarray(op.qubit_rotation(meta["theta"], meta["phi"])) @ psi
        psi = np.diag([np.exp(0.5j * angle), np.exp(-0.5j * angle)]) @ psi
        d_op = _displace(n_fock, beta / 2)
        psi = np.stack([d_op.conj().T @ psi[1], d_op @ psi[0]])
    return np.asarray(op.qubit_rotation(rotations[-1]["theta"], rotations[-1]["phi"])) @ psi


@pytest.mark.parametrize("frame_sign", [1.0, -1.0])
def test_virtual_z_rule_cancels_a_leftover_z(monkeypatch, frame_sign):
    """``frame_sign = s`` exactly undoes gates realized as ``ECD(beta) Z(-s theta')``.

    Checked with large theta' so the cancellation is unmistakable, and against
    the opposite sign to show the rule is not vacuous.
    """
    n_fock = 40
    theta_primes = np.array([0.7, -1.3, 0.45])
    betas = np.array([0.8 + 0.4j, -0.5 + 0.9j, 0.3 - 0.6j])
    thetas, phis = np.append(THETAS, 0.8), np.append(PHIS, -0.6)
    _stub_pulses(monkeypatch, theta_primes)
    seq = pl.compile_ecd_sequence(betas, thetas, phis, params=DEFAULT, frame_sign=frame_sign)
    ideal = _ideal_ecd(thetas, phis, betas, n_fock)
    p_ground = np.linalg.norm(ideal[0]) ** 2

    matched = _circuit_with_leftover_z(seq, betas, -frame_sign * theta_primes, n_fock)
    assert abs(np.vdot(ideal[0], matched[0])) == pytest.approx(p_ground, abs=1e-10)
    assert np.linalg.norm(matched[0]) ** 2 == pytest.approx(p_ground, abs=1e-10)

    mismatched = _circuit_with_leftover_z(seq, betas, frame_sign * theta_primes, n_fock)
    assert abs(np.linalg.norm(mismatched[0]) ** 2 - p_ground) > 0.05


def test_ecd_sequence_trailing_displacement(monkeypatch):
    _stub_pulses(monkeypatch, [0.0, 0.0])
    betas = np.array([0.8 + 0.4j, -0.5 + 0.9j, 0.3 - 0.2j])
    seq = pl.compile_ecd_sequence(betas, THETAS, PHIS, params=DEFAULT, final_displacement=True)
    assert [s.kind for s in seq.segments][-2:] == ["rotation", "displacement"]
    assert seq.meta["n_ecd"] == 2
    tail = seq.segments[-1]
    # alpha = -i * integral(eps) = beta / 2
    assert -1j * seq.eps[tail.slice()].sum() * DEFAULT.dt == pytest.approx(betas[-1] / 2)
    assert not seq.omega[tail.slice()].any()


def test_ecd_sequence_rejects_mismatched_lengths(monkeypatch):
    _stub_pulses(monkeypatch, [0.0] * 4)
    with pytest.raises(ValueError, match="len\\(thetas\\)"):
        pl.compile_ecd_sequence([1.0, 1.0], [0.1] * 4, [0.1] * 4, params=DEFAULT)
    with pytest.raises(ValueError, match="equal length"):
        pl.compile_ecd_sequence([1.0, 1.0], [0.1] * 3, [0.1] * 2, params=DEFAULT)


# ---------------------------------------------------------------------------
# ECD: simulated pulses against the ideal circuit
# ---------------------------------------------------------------------------


def _simulated_ground_agreement(seq: pl.PulseSequence, ideal: np.ndarray, n_fock: int):
    """Normalized ground-block fidelity with ``ideal`` and the simulated P(g)."""
    result = ps.simulate_sequence(seq, n_fock=n_fock, save_every=10**9)
    np.testing.assert_allclose(np.linalg.norm(result.states, axis=-1), 1.0, atol=1e-9)
    assert result.truncation_error() < 1e-8
    ground = result.blocks()[-1, 0]
    # Compare in the co-moving frame: displace the target, not the state.
    target = _displace(n_fock, -result.alpha[-1]) @ ideal[0]
    fidelity = abs(np.vdot(target, ground)) ** 2 / (
        np.linalg.norm(target) ** 2 * np.linalg.norm(ground) ** 2
    )
    return fidelity, np.linalg.norm(ground) ** 2


def test_compiled_ecd_sequence_reproduces_ideal_circuit(monkeypatch, default_pulse):
    """Table S1 device, including chi', Kerr and the kappa force in compilation."""
    n_fock = 40
    betas = abs(default_pulse.beta) * np.exp(1j * BETA_PHASES)
    _reuse_pulse(monkeypatch, default_pulse)
    seq = pl.compile_ecd_sequence(betas, THETAS, PHIS, alpha0=30.0, params=DEFAULT)
    assert seq.check_drive_limits(warn=False)["eps_ok"]

    ideal = _ideal_ecd(THETAS, PHIS, betas, n_fock)
    fidelity, p_ground = _simulated_ground_agreement(seq, ideal, n_fock)
    assert fidelity > 0.999
    assert p_ground == pytest.approx(np.linalg.norm(ideal[0]) ** 2, abs=0.01)


def test_simulated_ecd_gate_imparts_plus_theta_prime(fast_pulse):
    """One pulse on (|g> + |e>)|0>/sqrt(2) realizes ``ECD(beta) Z(+theta')``.

    The |g>-input branch ends in |e> displaced by +beta/2, the |e>-input branch
    in |g> displaced by -beta/2, and their relative phase is theta'.
    """
    n_fock = 50
    seq = pl.PulseSequence(
        eps=fast_pulse.eps, omega=fast_pulse.omega, params=FAST, delta=0.5 * FAST.chi
    )
    psi0 = np.zeros((2, n_fock), dtype=complex)
    psi0[:, 0] = 1 / np.sqrt(2)
    result = ps.simulate_sequence(seq, psi0=psi0, n_fock=n_fock, save_every=10**9)
    lab = result.blocks()[-1] @ _displace(n_fock, result.alpha[-1]).T

    vac = _vacuum(n_fock)
    amp_e = np.vdot(_displace(n_fock, fast_pulse.beta / 2) @ vac, lab[1])
    amp_g = np.vdot(_displace(n_fock, -fast_pulse.beta / 2) @ vac, lab[0])
    assert abs(amp_e) == pytest.approx(1 / np.sqrt(2), abs=0.01)
    assert abs(amp_g) == pytest.approx(1 / np.sqrt(2), abs=0.01)
    assert abs(fast_pulse.theta_prime) > 0.2  # large enough to resolve the sign
    assert np.angle(amp_e / amp_g) == pytest.approx(fast_pulse.theta_prime, abs=0.01)


def test_virtual_z_sign_matches_simulation(monkeypatch, fast_pulse):
    """On a device where theta' ~ 0.3 rad, only the default frame sign tracks the ideal circuit."""
    n_fock = 40
    betas = abs(fast_pulse.beta) * np.exp(1j * BETA_PHASES)
    _reuse_pulse(monkeypatch, fast_pulse)
    ideal = _ideal_ecd(THETAS, PHIS, betas, n_fock)

    seq = pl.compile_ecd_sequence(betas, THETAS, PHIS, alpha0=8.0, params=FAST)
    assert seq.meta["frame_sign"] == -1.0
    fidelity, p_ground = _simulated_ground_agreement(seq, ideal, n_fock)
    assert fidelity > 0.98
    assert p_ground == pytest.approx(np.linalg.norm(ideal[0]) ** 2, abs=0.04)

    flipped = pl.compile_ecd_sequence(betas, THETAS, PHIS, alpha0=8.0, params=FAST, frame_sign=1.0)
    assert _simulated_ground_agreement(flipped, ideal, n_fock)[0] < 0.97


# ---------------------------------------------------------------------------
# SNAP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def snap_device() -> device.SystemParams:
    params, _ = device.load_config(str(DEVICES / "device_snap.toml"))
    return params.replace(dt=10e-3, **LINEAR)


def test_snap_waveform_structure(snap_device):
    phases = np.array([0.3, -1.2, 2.0, 0.7])
    eps, omega = pl.snap_waveform(phases, snap_device, n_drive=6)
    n_pulse = round(snap_device.t_snap_selective / snap_device.dt)
    assert eps.shape == omega.shape == (2 * n_pulse,)
    assert not eps.any()

    # The first pulse carries no phase information.
    _, other = pl.snap_waveform(np.zeros(4), snap_device, n_drive=6)
    np.testing.assert_allclose(omega[:n_pulse], other[:n_pulse])
    assert not np.allclose(omega[n_pulse:], other[n_pulse:])

    # Levels beyond the supplied phases are driven with theta_n = 0.
    _, padded = pl.snap_waveform(np.append(phases, [0.0, 0.0]), snap_device)
    np.testing.assert_allclose(omega, padded)
    _, truncated = pl.snap_waveform(np.append(phases, [1.0, 1.0]), snap_device, n_drive=4)
    np.testing.assert_allclose(truncated, pl.snap_waveform(phases, snap_device)[1])

    # The padded-identity correction is a pi shift of the second pulse.
    _, bare = pl.snap_waveform(phases, snap_device, n_drive=6, match_padded_identity=False)
    np.testing.assert_allclose(bare[:n_pulse], omega[:n_pulse])
    np.testing.assert_allclose(bare[n_pulse:], -omega[n_pulse:])


def test_snap_sequence_layout(snap_device):
    rng = np.random.default_rng(0)
    phases = rng.uniform(-np.pi, np.pi, (2, 6))
    alphas = np.array([0.4 - 0.2j, -0.3 + 0.5j, 0.1 + 0.6j])
    seq = pl.compile_snap_sequence(phases, alphas, params=snap_device, n_drive=8)

    kinds = ["displacement", "snap", "displacement", "snap", "displacement"]
    assert [s.kind for s in seq.segments] == kinds
    for left, right in itertools.pairwise(seq.segments):
        assert left.stop == right.start
    np.testing.assert_array_equal(seq.gate_indices, [s.stop for s in seq.segments])
    assert seq.delta == 0.0
    assert seq.meta["gate_set"] == "snap" and seq.meta["n_drive"] == 8

    displacements = [s for s in seq.segments if s.kind == "displacement"]
    for seg, alpha in zip(displacements, alphas, strict=True):
        assert seg.stop - seg.start >= 8  # widened to stay resolved at dt = 10 ns
        assert -1j * seq.eps[seg.slice()].sum() * seq.dt == pytest.approx(alpha)
        assert not seq.omega[seg.slice()].any()
    for i, seg in enumerate(s for s in seq.segments if s.kind == "snap"):
        expected = pl.snap_waveform(phases[i], snap_device, n_drive=8)[1]
        np.testing.assert_allclose(seq.omega[seg.slice()], expected)
        assert not seq.eps[seg.slice()].any()
    assert seq.meta["sigma_disp"] > snap_device.sigma_disp


def test_snap_sequence_dt_override_and_argument_checks(snap_device):
    phases, alphas = np.zeros((1, 4)), [0.1, 0.2]
    seq = pl.compile_snap_sequence(phases, alphas, params=snap_device, n_drive=4, dt=20e-3)
    assert seq.dt == pytest.approx(20e-3)
    with pytest.warns(UserWarning, match="n_drive defaulted"):
        pl.compile_snap_sequence(phases, alphas, params=snap_device)
    with pytest.raises(ValueError, match="len\\(alphas\\)"):
        pl.compile_snap_sequence(phases, [0.1, 0.2, 0.3], params=snap_device, n_drive=4)


def test_simulated_snap_gate_imparts_the_requested_phases(snap_device):
    n_fock, n_levels = 16, 4
    params = snap_device.replace(dt=5e-3, n_selective_periods=16)
    phases = np.array([0.9, -2.1, 2.6, -0.4])
    seq = pl.compile_snap_sequence(phases[None], [0.0, 0.0], params=params, n_drive=6)
    psi0 = np.zeros(n_fock, dtype=complex)
    psi0[:n_levels] = 0.5
    result = ps.simulate_sequence(seq, psi0=psi0, n_fock=n_fock, save_every=10**9)

    assert result.p_ground()[-1] > 0.995
    out = result.blocks()[-1, 0, :n_levels]
    # Absolute, not just relative: driven levels pick up +e^{i theta_n}.
    np.testing.assert_allclose(np.angle(out / psi0[:n_levels] * np.exp(-1j * phases)), 0, atol=0.01)


def test_compiled_snap_sequence_reproduces_ideal_circuit(snap_device):
    n_fock, n_gates, n_snap = 16, 2, 6
    rng = np.random.default_rng(1)
    phases = rng.uniform(-np.pi, np.pi, (n_gates, n_snap))
    alphas = 0.5 * (rng.normal(size=n_gates + 1) + 1j * rng.normal(size=n_gates + 1))

    ideal_seq = sq.snap_sequence(n_gates, n_fock, n_snap=n_snap)
    flat = np.concatenate([phases.ravel(), np.stack([alphas.real, alphas.imag], -1).ravel()])
    vac = jnp.asarray(_vacuum(n_fock))[None]
    ideal = np.asarray(ideal_seq.propagate(jnp.asarray(flat), vac)[0])

    seq = pl.compile_snap_sequence(phases, alphas, params=snap_device, n_drive=8)
    result = ps.simulate_sequence(seq, n_fock=n_fock, save_every=10**9)
    assert result.truncation_error() < 1e-6
    # The frame integrator interpolates the drive between samples, so at this
    # coarse dt the frame lands a few 1e-3 away from the sum of displacements.
    assert result.alpha[-1] == pytest.approx(alphas.sum(), abs=0.01)
    assert result.fidelity(ideal[0]) > 0.999
    assert result.p_ground()[-1] > 0.9


# ---------------------------------------------------------------------------
# Sequence container and dispatch
# ---------------------------------------------------------------------------


def test_pulse_sequence_save_load_round_trip(tmp_path, snap_device):
    seq = pl.compile_snap_sequence(np.zeros((1, 4)), [0.1, 0.2j], params=snap_device, n_drive=4)
    path = tmp_path / "seq.npz"
    seq.save(str(path))
    loaded = pl.PulseSequence.load(str(path))

    np.testing.assert_array_equal(loaded.eps, seq.eps)
    np.testing.assert_array_equal(loaded.omega, seq.omega)
    np.testing.assert_array_equal(loaded.gate_indices, seq.gate_indices)
    assert loaded.params == seq.params and loaded.delta == seq.delta
    assert [(s.kind, s.start, s.stop, s.label) for s in loaded.segments] == [
        (s.kind, s.start, s.stop, s.label) for s in seq.segments
    ]
    assert loaded.meta["gate_set"] == "snap"
    np.testing.assert_allclose(loaded.t, (np.arange(seq.n_samples) + 0.5) * seq.dt)


def test_drive_limit_check_warns_when_exceeded(snap_device):
    seq = pl.compile_snap_sequence(np.zeros((1, 4)), [1.0, 0.0], params=snap_device, n_drive=4)
    info = seq.check_drive_limits()
    assert info["eps_ok"] and info["omega_ok"]
    assert (info["eps_peak"], info["omega_peak"]) == seq.peak_amplitudes()
    weak_params = snap_device.replace(eps_max=1e-3, omega_max=1e-3)
    weak = pl.PulseSequence(seq.eps, seq.omega, weak_params, 0.0)
    with pytest.warns(UserWarning) as caught:
        info = weak.check_drive_limits()
    assert not info["eps_ok"] and not info["omega_ok"]
    assert len(caught) == 2
    assert weak.check_drive_limits(warn=False) == info


def test_compile_from_result_dispatches_on_gate_set(monkeypatch, snap_device):
    snap = SimpleNamespace(
        gate_set="snap", config={}, params={"snap_phases": np.zeros((1, 4)), "alphas": [0.1, 0.2]}
    )
    assert pl.compile_from_result(snap, snap_device, n_drive=4).meta["gate_set"] == "snap"

    _stub_pulses(monkeypatch, [0.0, 0.0])
    ecd_params = {"betas": np.array([1.0, 1j]), "thetas": THETAS, "phis": PHIS}
    ecd = SimpleNamespace(gate_set="ecd", config={"echoed": True}, params=ecd_params)
    assert pl.compile_from_result(ecd, DEFAULT).meta["gate_set"] == "ecd"

    bare = SimpleNamespace(gate_set="ecd", config={"echoed": False}, params=ecd_params)
    with pytest.raises(ValueError, match="echoed"):
        pl.compile_from_result(bare, DEFAULT)
    with pytest.raises(ValueError, match="unknown gate set"):
        pl.compile_from_result(SimpleNamespace(gate_set="csq", config={}, params={}), DEFAULT)
