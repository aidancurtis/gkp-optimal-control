"""Saving and reloading gate-optimization results."""

import json

import numpy as np
import pytest

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control.gate_optimization import (
    GateBounds,
    GateOptResult,
    OptimizerConfig,
    TrajectoryPenalty,
    build_sequence,
    csq_sequence,
    ecd_sequence,
    optimize_gate_sequence,
    sequence_history,
    snap_sequence,
)
from gkp_optimal_control.gate_results import (
    gate_result_path,
    load_gate_result,
    save_gate_result,
)

N_FOCK = 16
OPTIMIZER = OptimizerConfig(n_seeds=2, n_adam_iters=40, polish_maxiter=20)


def _fock(n: int) -> np.ndarray:
    ket = np.zeros(N_FOCK, dtype=complex)
    ket[n] = 1.0
    return ket


@pytest.fixture(scope="module")
def results() -> dict[str, GateOptResult]:
    runs = {
        "ecd": (ecd_sequence(2, N_FOCK, echoed=False, n_leak=3), _fock(1), {}),
        "snap": (snap_sequence(2, N_FOCK, n_snap=5), _fock(1), {}),
        "csq": (
            csq_sequence(2, N_FOCK, qubit_target="traced"),
            _fock(2),
            {"trajectory": TrajectoryPenalty.notes_defaults()},
        ),
    }
    return {
        name: optimize_gate_sequence(
            seq,
            _fock(0),
            target,
            bounds=GateBounds(max_disp=3.0),
            optimizer=OPTIMIZER,
            verbose=False,
            **extra,
        )  # fmt: skip
        for name, (seq, target, extra) in runs.items()
    }


def test_result_records_what_is_needed_to_rebuild(results):
    assert results["ecd"].sequence.options == {
        "n_leak": 3, "qubit_target": "ground", "disp_method": "fixed", "echoed": False,
    }  # fmt: skip
    assert results["snap"].sequence.options == {"n_leak": 5, "disp_method": "fixed", "n_snap": 5}
    assert results["csq"].sequence.options == {"n_leak": 5, "qubit_target": "traced"}
    for result in results.values():
        np.testing.assert_allclose(result.initial_state, _fock(0)[None])
        np.testing.assert_allclose(result.repropagate()[-1], result.final_states, atol=1e-12)


@pytest.mark.parametrize("gate_set", ["ecd", "snap", "csq"])
def test_save_load_round_trip(tmp_path, results, gate_set):
    result = results[gate_set]
    path = save_gate_result(result, tmp_path / "run", note="hello", wall_time_s=1.5)
    assert path == tmp_path / "run.json"
    loaded = load_gate_result(tmp_path / "run")  # suffix optional on load too

    for name in ("gate_set", "n_gates", "n_fock", "fidelity", "loss", "leakage", "best_seed",
                 "qubit_purity", "penalties", "polish_info"):  # fmt: skip
        assert getattr(loaded, name) == getattr(result, name), name
    for name in ("flat_params", "final_states", "per_seed_fidelity", "adam_history",
                 "trajectory", "initial_state"):  # fmt: skip
        got, want = getattr(loaded, name), getattr(result, name)
        assert got.dtype == want.dtype and got.shape == want.shape, name
        np.testing.assert_array_equal(got, want, err_msg=name)
    assert loaded.params.keys() == result.params.keys()
    for key, value in result.params.items():
        np.testing.assert_array_equal(loaded.params[key], value, err_msg=key)
    for key, value in result.geodesic.items():
        np.testing.assert_array_equal(loaded.geodesic[key], value, err_msg=key)

    # Config dataclasses come back as dataclasses, equal to the originals.
    assert loaded.config == result.config
    assert isinstance(loaded.config["bounds"], GateBounds)
    assert isinstance(loaded.config["trajectory"], TrajectoryPenalty)

    assert loaded.metadata["note"] == "hello" and loaded.metadata["wall_time_s"] == 1.5
    assert loaded.ansatz == result.sequence.options
    assert loaded.sequence.options == result.sequence.options
    assert loaded.summary() == result.summary()


@pytest.mark.parametrize("gate_set", ["ecd", "snap", "csq"])
def test_loaded_result_repropagates_and_resaves(tmp_path, results, gate_set):
    result = results[gate_set]
    loaded = load_gate_result(save_gate_result(result, tmp_path / "a"))
    history = loaded.repropagate()
    np.testing.assert_allclose(history, sequence_history(result, _fock(0)), atol=1e-12)
    np.testing.assert_allclose(history[-1], loaded.final_states, atol=1e-12)

    again = load_gate_result(save_gate_result(loaded, tmp_path / "b", second="pass"))
    np.testing.assert_array_equal(again.flat_params, result.flat_params)
    assert again.ansatz == loaded.ansatz
    assert again.metadata["second"] == "pass"


def test_save_refuses_to_overwrite_and_accepts_explicit_initial_state(tmp_path, results):
    result = results["snap"]
    path = save_gate_result(result, tmp_path / "run.json")
    with pytest.raises(FileExistsError):
        save_gate_result(result, path)
    save_gate_result(result, path, overwrite=True, initial_state=[_fock(0), _fock(3)])
    assert load_gate_result(path).initial_state.shape == (2, N_FOCK)

    with pytest.raises(TypeError, match="GateOptResult"):
        save_gate_result({"fidelity": 1.0}, tmp_path / "bad")
    with pytest.raises(TypeError, match="cannot serialize"):
        save_gate_result(result, tmp_path / "bad", callback=print)
    with pytest.raises(ValueError, match="params"):
        save_gate_result(result, tmp_path / "bad", ansatz={"n_snap": 3})


def test_files_with_retired_options_still_load(tmp_path, results):
    """Older files carry builder options and config fields that no longer exist."""
    path = save_gate_result(results["ecd"], tmp_path / "old")
    record = json.loads(path.read_text())
    record["ansatz"]["disp_method"] = "quadrature"  # former name of "fixed"
    record["config"]["bounds"]["data"]["n_leak"] = 8
    record["config"]["schedule"] = {"__dataclass__": "somewhere.Gone", "data": {"x": 1}}
    del record["metadata"], record["geodesic"]
    path.write_text(json.dumps(record))

    loaded = load_gate_result(path)
    assert loaded.config["bounds"] == results["ecd"].config["bounds"]
    assert loaded.config["schedule"] == {"x": 1}
    assert loaded.metadata == {} and loaded.geodesic == {}
    assert loaded.sequence.options["disp_method"] == "fixed"
    np.testing.assert_allclose(loaded.repropagate()[-1], loaded.final_states, atol=1e-12)

    # With no method recorded, the one that reproduces the final state is chosen.
    del record["ansatz"]["disp_method"]
    path.write_text(json.dumps(record))
    assert load_gate_result(path).sequence.options["disp_method"] == "fixed"

    record["schema_version"] = 99
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema version"):
        load_gate_result(path)


def test_unrecorded_exact_displacement_is_inferred(tmp_path):
    """A result optimized with the exact displacement, saved without saying so."""
    seq = ecd_sequence(3, 12, disp_method="exact")  # small space: the edge matters
    rng = np.random.default_rng(0)
    flat = np.concatenate([rng.uniform(0, np.pi, 8), 1.5 * rng.normal(size=6)])
    ket = np.zeros(12, dtype=complex)
    ket[3] = 1.0
    final = np.asarray(seq.propagate(flat, seq.lift(ket[None]))[0])
    result = GateOptResult(
        gate_set="ecd", n_gates=3, n_fock=12, fidelity=0.5, loss=0.5, leakage=0.0,
        params=seq.describe(flat), flat_params=flat, final_states=final,
        per_seed_fidelity=np.array([0.5]), best_seed=0, adam_history=np.zeros((1, 1)),
        sequence=seq, initial_state=ket[None],
    )  # fmt: skip
    path = save_gate_result(result, tmp_path / "exact")
    record = json.loads(path.read_text())
    assert record["ansatz"]["disp_method"] == "exact"

    del record["ansatz"]["disp_method"]
    path.write_text(json.dumps(record))
    loaded = load_gate_result(path)
    assert loaded.sequence.options["disp_method"] == "exact"
    np.testing.assert_allclose(loaded.repropagate()[-1], final, atol=1e-12)

    record["ansatz"]["disp_method"] = "expm"  # its former name
    path.write_text(json.dumps(record))
    assert load_gate_result(path).sequence.options["disp_method"] == "exact"


def test_build_sequence_and_path_helpers(tmp_path):
    seq = build_sequence("snap", 3, N_FOCK, n_snap=4, echoed=True, qubit_target=None)
    assert seq.name == "snap" and seq.n_params == 3 * 4 + 2 * 4
    assert build_sequence("ecd", 2, N_FOCK, n_snap=4).n_params == 10
    with pytest.raises(ValueError, match="unknown gate set"):
        build_sequence("cnot", 1, N_FOCK)

    assert gate_result_path(tmp_path, "ecd", 12, 0.99984) == tmp_path / "ecd_g12_fid0.9998.json"
    assert gate_result_path("results", "csq", 4).name == "csq_g4.json"
