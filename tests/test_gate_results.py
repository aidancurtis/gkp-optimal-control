"""Saving, reloading and indexing gate-optimization results."""

import json
import time

import numpy as np
import pytest

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control.gate_optimization import (
    GateBounds,
    GateOptResult,
    OptimizerConfig,
    TrajectoryPenalty,
    optimize_gate_sequence,
    sequence_history,
)
from gkp_optimal_control.gate_results import (
    gate_result_path,
    list_gate_results,
    load_gate_result,
    save_gate_result,
)
from gkp_optimal_control.sequences import (
    GateSequence,
    build_sequence,
    csq_sequence,
    ecd_sequence,
    snap_sequence,
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
        "ecd": (ecd_sequence(2, N_FOCK, echoed=False, n_leak=3), 1, {}),
        "snap": (snap_sequence(2, N_FOCK, n_snap=5, disp_method="exact"), 1, {}),
        "csq": (
            csq_sequence(2, N_FOCK, qubit_target="traced"),
            2,
            {"trajectory": TrajectoryPenalty.notes_defaults()},
        ),
    }
    return {
        name: optimize_gate_sequence(
            seq,
            _fock(0),
            _fock(target),
            bounds=GateBounds(max_disp=3.0),
            optimizer=OPTIMIZER,
            init_label="vac",
            target_label=f"fock:{target}",
            verbose=False,
            **extra,
        )  # fmt: skip
        for name, (seq, target, extra) in runs.items()
    }


def test_result_describes_its_problem(results):
    assert results["ecd"].sequence.options == {
        "n_leak": 3, "qubit_target": "ground", "disp_method": "fixed", "echoed": False,
    }  # fmt: skip
    assert results["snap"].sequence.options == {"n_leak": 5, "disp_method": "exact", "n_snap": 5}
    assert results["csq"].sequence.options == {"n_leak": 5, "qubit_target": "traced"}
    for name, target in (("ecd", 1), ("snap", 1), ("csq", 2)):
        result = results[name]
        assert (result.init_label, result.target_label) == ("vac", f"fock:{target}")
        np.testing.assert_allclose(result.initial_state, _fock(0)[None])
        np.testing.assert_allclose(result.target_state, _fock(target)[None])
        np.testing.assert_allclose(result.repropagate()[-1], result.final_states, atol=1e-12)


@pytest.mark.parametrize("gate_set", ["ecd", "snap", "csq"])
def test_save_load_round_trip(tmp_path, results, gate_set):
    result = results[gate_set]
    path = save_gate_result(result, tmp_path / "run", note="hello", wall_time_s=1.5)
    assert path == tmp_path / "run.json" and (tmp_path / "run.npz").exists()
    loaded = load_gate_result(tmp_path / "run")  # suffix optional on load too

    for name in ("gate_set", "n_gates", "n_fock", "fidelity", "loss", "leakage", "best_seed",
                 "qubit_purity", "penalties", "polish_info", "init_label",
                 "target_label"):  # fmt: skip
        assert getattr(loaded, name) == getattr(result, name), name
    for name in ("flat_params", "final_states", "per_seed_fidelity", "adam_history",
                 "trajectory", "initial_state", "target_state"):  # fmt: skip
        got, want = getattr(loaded, name), getattr(result, name)
        assert got.dtype == want.dtype and got.shape == want.shape, name
        np.testing.assert_array_equal(got, want, err_msg=name)
    assert loaded.params.keys() == result.params.keys()
    for key, value in result.params.items():
        assert loaded.params[key].dtype == np.asarray(value).dtype, key
        np.testing.assert_array_equal(loaded.params[key], value, err_msg=key)
    assert loaded.geodesic.keys() == result.geodesic.keys()
    for key, value in result.geodesic.items():
        np.testing.assert_array_equal(loaded.geodesic[key], value, err_msg=key)

    # Config dataclasses come back as dataclasses, equal to the originals.
    assert loaded.config == result.config
    assert isinstance(loaded.config["bounds"], GateBounds)
    assert isinstance(loaded.config["trajectory"], TrajectoryPenalty)
    assert isinstance(loaded.config["optimizer"], OptimizerConfig)

    assert loaded.metadata == {"note": "hello", "wall_time_s": 1.5}
    assert loaded.sequence.options == result.sequence.options
    assert loaded.summary() == result.summary()


def test_json_is_small_and_readable(tmp_path, results):
    path = save_gate_result(results["ecd"], tmp_path / "run")
    record = json.loads(path.read_text())
    assert record["schema_version"] == 2 and record["arrays"] == "run.npz"
    assert record["problem"] == {
        "gate_set": "ecd", "n_gates": 2, "n_fock": N_FOCK, "n_params": 10,
        "options": results["ecd"].sequence.options, "init_label": "vac", "target_label": "fock:1",
    }  # fmt: skip
    assert record["outcome"]["fidelity"] == results["ecd"].fidelity
    assert record["config"]["optimizer"]["n_seeds"] == 2
    # Gate parameters are plain lists; complex ones split into real and imaginary parts.
    assert record["gate_parameters"]["thetas"] == results["ecd"].params["thetas"].tolist()
    assert set(record["gate_parameters"]["betas"]) == {"re", "im"}
    # No state vectors or histories in the text file.
    assert "final_states" not in record and "adam_history" not in record
    with np.load(tmp_path / "run.npz") as npz:
        assert {"initial_state", "target_state", "final_states", "adam_history"} <= set(npz.files)
        assert "geodesic.deviation" in npz.files


@pytest.mark.parametrize("gate_set", ["ecd", "snap", "csq"])
def test_loaded_result_repropagates_and_resaves(tmp_path, results, gate_set):
    result = results[gate_set]
    loaded = load_gate_result(save_gate_result(result, tmp_path / "a"))
    history = loaded.repropagate()
    np.testing.assert_allclose(history, sequence_history(result, _fock(0)), atol=1e-12)
    np.testing.assert_allclose(history[-1], loaded.final_states, atol=1e-12)

    again = load_gate_result(save_gate_result(loaded, tmp_path / "b", second="pass"))
    np.testing.assert_array_equal(again.flat_params, result.flat_params)
    assert again.sequence.options == loaded.sequence.options
    assert again.metadata == {"second": "pass"}


def test_paths_are_derived_from_the_problem(tmp_path, results):
    path = save_gate_result(results["ecd"], root=tmp_path)
    assert path.parent == tmp_path / "fock-1" / "ecd"
    stamp = time.strftime("%Y%m%d")
    assert path.name.startswith(f"n02_{stamp}-") and path.suffix == ".json"
    # A second run in the same second gets its own file.
    other = save_gate_result(results["ecd"], root=tmp_path)
    assert other != path and other.parent == path.parent
    assert len(list(path.parent.glob("*.json"))) == len(list(path.parent.glob("*.npz"))) == 2

    assert save_gate_result(results["csq"], root=tmp_path).parent == tmp_path / "fock-2" / "csq"

    assert gate_result_path("r", "ecd", 7, "gkp:0") == tmp_path.__class__("r/gkp-0/ecd/n07.json")
    assert gate_result_path("r", "snap", 12).as_posix() == "r/snap/n12.json"
    assert gate_result_path("r", "csq", 4, "cat even 2.0", "warm").as_posix() == (
        "r/cat-even-2.0/csq/warm.json"
    )


def test_stable_path_as_a_cache_key(tmp_path, results):
    result = results["snap"]
    path = gate_result_path(tmp_path, "snap", 2, result.target_label)
    assert not path.exists()
    assert save_gate_result(result, path) == path
    assert load_gate_result(path).fidelity == result.fidelity
    with pytest.raises(FileExistsError):
        save_gate_result(result, path)
    save_gate_result(result, path, overwrite=True, attempt=2)
    assert load_gate_result(path).metadata == {"attempt": 2}


def test_list_gate_results_indexes_a_tree(tmp_path, results):
    assert list_gate_results(tmp_path) == []
    paths = {name: save_gate_result(res, root=tmp_path, tag=name) for name, res in results.items()}
    (tmp_path / "notes.json").write_text('{"unrelated": true}')
    (tmp_path / "broken.json").write_text("{not json")

    rows = list_gate_results(tmp_path)
    assert len(rows) == 3
    assert {row["path"] for row in rows} == set(paths.values())
    assert [row["saved_at"] for row in rows] == sorted((r["saved_at"] for r in rows), reverse=True)
    by_set = {row["gate_set"]: row for row in rows}
    for name, result in results.items():
        row = by_set[name]
        assert row["fidelity"] == result.fidelity and row["leakage"] == result.leakage
        assert (row["n_gates"], row["n_fock"]) == (2, N_FOCK)
        assert row["n_params"] == result.flat_params.size
        assert row["target_label"] == result.target_label and row["init_label"] == "vac"
        assert row["options"] == result.sequence.options
        assert row["metadata"] == {"tag": name}
    assert [r["gate_set"] for r in rows if r["target_label"] == "fock:2"] == ["csq"]
    assert list_gate_results(tmp_path / "fock-1" / "snap")[0]["path"] == paths["snap"]


def test_save_rejects_what_it_cannot_reload(tmp_path, results):
    result = results["snap"]
    with pytest.raises(TypeError, match="GateOptResult"):
        save_gate_result({"fidelity": 1.0}, tmp_path / "bad")
    with pytest.raises(TypeError, match="cannot serialize"):
        save_gate_result(result, tmp_path / "bad", callback=print)
    assert not (tmp_path / "bad.json").exists() and not (tmp_path / "bad.npz").exists()

    # A hand-built sequence carries no builder options to rebuild it from.
    seq = result.sequence
    custom = GateSequence(**{**vars(seq), "options": {}})
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        save_gate_result(GateOptResult(**{**vars(result), "sequence": custom}), tmp_path / "bad")


def test_load_rejects_other_schemas(tmp_path, results):
    path = save_gate_result(results["ecd"], tmp_path / "run")
    record = json.loads(path.read_text())
    record["schema_version"] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unsupported gate-result schema"):
        load_gate_result(path)
    assert list_gate_results(tmp_path) == []


def test_build_sequence():
    seq = build_sequence("snap", 3, N_FOCK, n_snap=4, echoed=True, qubit_target=None)
    assert seq.name == "snap" and seq.n_params == 3 * 4 + 2 * 4
    assert build_sequence("ecd", 2, N_FOCK, n_snap=4).n_params == 10
    with pytest.raises(ValueError, match="unknown gate set"):
        build_sequence("cnot", 1, N_FOCK)
