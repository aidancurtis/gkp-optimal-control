"""The gate-sequence optimizer: its stages, options, caching and penalty terms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control import gate_optimization as go
from gkp_optimal_control.gate_optimization import (
    GateBounds,
    OptimizerConfig,
    TrajectoryPenalty,
    adam_run,
    cosine_schedule,
    optimize_gate_sequence,
    sequence_history,
)
from gkp_optimal_control.sequences import csq_sequence, ecd_sequence, snap_sequence
from gkp_optimal_control.states import squeezed_vacuum

N_FOCK = 16
QUICK = OptimizerConfig(n_seeds=4, n_adam_iters=200, polish_maxiter=100)


def _fock(n: int) -> np.ndarray:
    return np.eye(N_FOCK, dtype=complex)[n]


def _run(seq, target=1, optimizer=QUICK, **kwargs):
    """Optimize from vacuum; ``target`` is a Fock index or a ket."""
    ket = _fock(target) if isinstance(target, int) else target
    return optimize_gate_sequence(seq, _fock(0), ket, optimizer=optimizer, verbose=False, **kwargs)


# ---------------------------------------------------------------------------
# Adam building blocks
# ---------------------------------------------------------------------------


def test_cosine_schedule_warms_up_then_decays():
    lr_at = cosine_schedule(100, peak_lr=0.1, warmup_frac=0.1, final_frac=0.05)
    lr = np.array([float(lr_at(i)) for i in range(100)])
    np.testing.assert_allclose(lr[:10], 0.1 * np.arange(1, 11) / 10)
    assert lr[10] == pytest.approx(0.1) and lr.argmax() in (9, 10)
    assert np.all(np.diff(lr[10:]) < 0)
    assert lr[-1] == pytest.approx(0.005, rel=0.02)


def test_adam_run_minimizes_a_quadratic():
    target = jnp.array([1.0, -2.0, 0.5])
    loss_and_grad = jax.value_and_grad(lambda p: jnp.sum((p - target) ** 2))
    params, history = adam_run(
        loss_and_grad, jnp.zeros(3), 400, cosine_schedule(400, 0.1, 0.05, 0.01)
    )
    assert history.shape == (400,)
    assert history[0] == pytest.approx(5.25) and history[-1] < 1e-8
    np.testing.assert_allclose(params, target, atol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end optimization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq, target",
    [
        (ecd_sequence(4, N_FOCK), 1),
        (snap_sequence(2, N_FOCK, n_snap=6), 1),
        (csq_sequence(2, N_FOCK), np.asarray(squeezed_vacuum(N_FOCK, 0.3))),
    ],
    ids=["ecd", "snap", "csq"],
)
def test_reaches_the_target(seq, target):
    result = _run(seq, target, bounds=GateBounds(max_disp=3.0, max_squeeze=1.5))
    assert result.fidelity > 0.99
    assert result.gate_set == seq.name and result.n_gates == seq.n_gates
    assert result.flat_params.shape == (seq.n_params,)
    assert result.per_seed_fidelity.shape == (4,)
    assert result.adam_history.shape == (4, 200) and np.isfinite(result.adam_history).all()
    assert result.fidelity >= result.per_seed_fidelity.max() - 1e-12
    assert result.polish_info["fidelity_before"] == result.per_seed_fidelity[result.best_seed]
    assert result.trajectory[-1] == pytest.approx(result.fidelity, abs=1e-9)
    assert result.params["strength_magnitudes"].max() < 3.2

    # The reported state and fidelity are what the stored parameters produce.
    history = sequence_history(result, _fock(0))
    np.testing.assert_allclose(history[-1], result.final_states, atol=1e-12)
    ground = history[-1][0, 0] if seq.has_qubit else history[-1][0]
    ket = _fock(target) if isinstance(target, int) else target
    assert abs(np.vdot(ket, ground)) ** 2 == pytest.approx(result.fidelity, abs=1e-9)
    assert "fidelity" in result.summary()


def test_runs_are_reproducible_and_reuse_compiled_code():
    go._COMPILED.clear()
    first = _run(ecd_sequence(3, N_FOCK))
    assert len(go._COMPILED) == 1
    again = _run(ecd_sequence(3, N_FOCK))  # a new but equivalent sequence object
    assert len(go._COMPILED) == 1
    np.testing.assert_array_equal(again.flat_params, first.flat_params)
    assert again.fidelity == first.fidelity

    other_target = _run(ecd_sequence(3, N_FOCK), target=2)
    assert len(go._COMPILED) == 1  # states are arguments, not part of the compiled code
    assert not np.allclose(other_target.flat_params, first.flat_params)

    _run(ecd_sequence(3, N_FOCK, echoed=False))
    _run(ecd_sequence(3, N_FOCK), loss_type="log_infidelity")
    assert len(go._COMPILED) == 3


def test_seed_changes_the_start_and_params0_overrides_it():
    seq = ecd_sequence(3, N_FOCK)
    base = _run(seq)
    reseeded = _run(seq, optimizer=OptimizerConfig(n_seeds=4, n_adam_iters=200, seed=5))
    assert not np.allclose(reseeded.per_seed_fidelity, base.per_seed_fidelity)

    warm = _run(seq, params0=base.flat_params)  # one start, already converged
    assert warm.per_seed_fidelity.shape == (1,)
    assert warm.fidelity >= base.fidelity - 1e-6

    starts = np.stack([base.flat_params, np.zeros_like(base.flat_params)])
    two = _run(seq, params0=starts, optimizer=OptimizerConfig(n_adam_iters=20, polish=False))
    assert two.per_seed_fidelity.shape == (2,) and two.polish_info == {}
    with pytest.raises(ValueError, match="params0 has"):
        _run(seq, params0=np.zeros(3))


def test_target_fidelity_stops_adam_early():
    seq = snap_sequence(2, N_FOCK, n_snap=6)
    config = OptimizerConfig(
        n_seeds=4, n_adam_iters=600, polish=False, target_fidelity=0.9, prune_every=50
    )
    result = _run(seq, optimizer=config)
    assert result.fidelity >= 0.9
    ran = np.isfinite(result.adam_history).all(axis=0).sum()
    assert 0 < ran < 600 and ran % 50 == 0
    assert np.isnan(result.adam_history[:, ran:]).all()

    unreachable = _run(seq, optimizer=OptimizerConfig(n_seeds=2, n_adam_iters=120, polish=False,
                                                      target_fidelity=2.0))  # fmt: skip
    assert np.isfinite(unreachable.adam_history).all()


def test_pruning_halves_the_running_seeds():
    seq = ecd_sequence(3, N_FOCK)
    config = OptimizerConfig(n_seeds=8, n_adam_iters=200, polish=False, prune_every=50)
    result = _run(seq, optimizer=config)
    running = np.isfinite(result.adam_history).sum(axis=0)
    np.testing.assert_array_equal(
        running[[0, 49, 50, 99, 100, 149, 150, 199]], [8, 8, 4, 4, 2, 2, 1, 1]
    )
    assert np.isfinite(result.per_seed_fidelity).all()
    # The survivor was the best seed at every cut, and is the one reported.
    assert np.isfinite(result.adam_history[result.best_seed]).all()

    plain = _run(seq, optimizer=OptimizerConfig(n_seeds=8, n_adam_iters=200, polish=False))
    np.testing.assert_allclose(
        result.adam_history[:, :50], plain.adam_history[:, :50], rtol=1e-9, atol=1e-12
    )
    assert result.fidelity >= 0.9 * plain.fidelity

    gentle = _run(
        seq,
        optimizer=OptimizerConfig(
            n_seeds=8, n_adam_iters=100, polish=False, prune_every=50, keep_fraction=0.75
        ),
    )
    assert np.isfinite(gentle.adam_history).sum(axis=0)[-1] == 6


def test_multiple_pairs_and_coherent_reduction():
    seq = snap_sequence(3, N_FOCK, n_snap=6)
    pairs = dict(psi_init=[_fock(0), _fock(1)], psi_targ=[_fock(1), _fock(0)])
    result = optimize_gate_sequence(
        seq, **pairs, batch_reduction="coherent", optimizer=QUICK, verbose=False
    )
    assert result.final_states.shape == (2, N_FOCK)
    assert result.params["per_pair_fidelity"].shape == (2,)
    assert result.fidelity > 0.95
    # Coherent: the mean overlap is squared, so relative phases are fixed too.
    overlaps = np.einsum("kd,kd->k", np.conj(np.stack(pairs["psi_targ"])), result.final_states)
    assert abs(overlaps.mean()) ** 2 == pytest.approx(result.fidelity, abs=1e-9)


def test_argument_validation():
    with pytest.raises(ValueError, match="must pair up"):
        optimize_gate_sequence(ecd_sequence(2, N_FOCK), [_fock(0), _fock(1)], _fock(2))
    with pytest.raises(ValueError, match="coherent"):
        _run(ecd_sequence(2, N_FOCK, qubit_target="traced"), batch_reduction="coherent")
    with pytest.raises(ValueError, match="unknown loss_type"):
        _run(ecd_sequence(2, N_FOCK), loss_type="rmse")
    with pytest.raises(ValueError, match="geodesic penalty"):
        optimize_gate_sequence(
            ecd_sequence(2, N_FOCK), _fock(0), _fock(0),
            trajectory=TrajectoryPenalty.brachistochrone(), verbose=False,
        )  # fmt: skip
    with pytest.raises(ValueError, match="geodesic_mode"):
        TrajectoryPenalty(geodesic_mode="spiral")


def test_csq_warns_about_unreachable_parity():
    quick = OptimizerConfig(n_seeds=1, n_adam_iters=5, polish=False)
    with pytest.warns(RuntimeWarning, match="parity"):
        _run(csq_sequence(2, N_FOCK), target=1, optimizer=quick)


# ---------------------------------------------------------------------------
# Trajectory penalties
# ---------------------------------------------------------------------------


def test_penalties_are_reported_only_when_active():
    seq = ecd_sequence(3, N_FOCK)
    assert _run(seq).penalties == {}
    result = _run(seq, trajectory=TrajectoryPenalty.notes_defaults())
    assert set(result.penalties) == {
        "pen_mono",
        "pen_curve",
        "pen_control",
        "pen_rot",
        "pen_geodesic",
    }
    assert all(v >= 0 for v in result.penalties.values())
    assert result.geodesic["deviation"].shape == result.trajectory.shape
    np.testing.assert_allclose(result.penalties["pen_rot"], np.sum(result.params["thetas"] ** 2))
    # The reported loss is the endpoint loss plus the weighted penalties.
    cfg = TrajectoryPenalty.notes_defaults()
    weighted = (
        cfg.mono_weight * result.penalties["pen_mono"]
        + cfg.curve_weight * result.penalties["pen_curve"]
        + cfg.control_weight * result.penalties["pen_control"]
        + cfg.rot_weight * result.penalties["pen_rot"]
    )
    assert result.loss == pytest.approx(1 - result.fidelity + result.leakage + weighted, abs=1e-9)


def test_rotation_penalty_shrinks_the_rotation_angles():
    seq = ecd_sequence(3, N_FOCK)
    free = _run(seq)
    damped = _run(seq, trajectory=TrajectoryPenalty(rot_weight=0.2))
    assert np.sum(damped.params["thetas"] ** 2) < np.sum(free.params["thetas"] ** 2)


def test_path_terms_on_handmade_amplitudes():
    # (L, M, K, Q) amplitudes: probe 0 is the target; one pair, one block.
    fids = np.array([0.0, 0.5, 0.3, 0.9])
    amps = np.zeros((4, 3, 1, 1), dtype=complex)
    amps[:, 0, 0, 0] = np.sqrt(fids)
    np.testing.assert_allclose(go._path_fidelity(jnp.asarray(amps)), fids)

    terms = go._trajectory_terms(jnp.asarray(amps), [], jnp.zeros((0,)))
    assert float(terms["mono"]) == pytest.approx(0.2**2)  # the single drop, 0.5 -> 0.3
    second_diff = fids[2:] - 2 * fids[1:-1] + fids[:-2]
    assert float(terms["curve"]) == pytest.approx(np.sum(second_diff**2))
    assert float(terms["geodesic"]) == 0.0 and float(terms["control"]) == 0.0


def test_control_roughness_wraps_angles():
    values = jnp.asarray([[0.0], [0.5], [0.5]])
    assert float(go._control_roughness([(values, False)])) == pytest.approx(0.25)
    # A jump from -pi + 0.1 to pi - 0.1 is a step of 0.2 on the circle.
    angles = jnp.asarray([[-np.pi + 0.1], [np.pi - 0.1]])
    assert float(go._control_roughness([(angles, True)])) == pytest.approx(0.04)
    assert float(go._control_roughness([(angles, False)])) == pytest.approx((2 * np.pi - 0.2) ** 2)
    assert float(go._control_roughness([(values[:1], False)])) == 0.0


def test_geodesic_overlap_is_one_on_the_geodesic():
    rng = np.random.default_rng(0)
    psi_i = rng.normal(size=(1, N_FOCK)) + 1j * rng.normal(size=(1, N_FOCK))
    psi_t = rng.normal(size=(1, N_FOCK)) + 1j * rng.normal(size=(1, N_FOCK))
    psi_i /= np.linalg.norm(psi_i)
    psi_t /= np.linalg.norm(psi_t)
    perp, theta = go._geodesic_frame(psi_i, psi_t)
    assert abs(np.vdot(psi_i[0], perp[0])) < 1e-12
    assert np.linalg.norm(perp) == pytest.approx(1.0)
    assert theta[0] == pytest.approx(np.arccos(abs(np.vdot(psi_i[0], psi_t[0]))))

    # States at uniform steps along the arc from psi_i towards psi_t.
    steps = np.linspace(0.0, 1.0, 5)
    arc = np.stack(
        [np.cos(s * theta[0]) * psi_i[0] + np.sin(s * theta[0]) * perp[0] for s in steps]
    )
    probes = np.stack([psi_t, psi_i, perp])  # (M, K, d)
    amps = jnp.asarray(np.einsum("mkd,ld->lmk", np.conj(probes), arc)[..., None])
    theta_j = jnp.asarray(theta)

    f_schedule, a_schedule = go._geodesic_overlap(amps, theta_j, "schedule")
    np.testing.assert_allclose(f_schedule[:, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(a_schedule[:, 0], steps * theta[0], atol=1e-12)
    f_tube, a_tube = go._geodesic_overlap(amps, theta_j, "tube")
    np.testing.assert_allclose(f_tube[:, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(a_tube[:, 0], steps * theta[0], atol=1e-6)
    np.testing.assert_allclose(amps[-1, 0, 0, 0] * np.conj(amps[-1, 0, 0, 0]), 1.0, atol=1e-12)

    # Off the arc (run backwards) the schedule overlap drops but the tube one does not.
    f_back, _ = go._geodesic_overlap(amps[::-1], theta_j, "schedule")
    assert float(f_back[0, 0]) < 0.99
    np.testing.assert_allclose(
        go._geodesic_overlap(amps[::-1], theta_j, "tube")[0], 1.0, atol=1e-12
    )
