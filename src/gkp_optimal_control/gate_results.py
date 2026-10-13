"""Save, reload and index gate-optimization results.

Each run is two files side by side:

``<name>.json``
    Small and readable: the problem (gate set, depth, truncation, builder
    options, state labels), the optimizer settings, scalar outcomes and the
    gate parameters.
``<name>.npz``
    The bulk arrays: initial, target and final states, per-seed fidelities,
    the Adam loss history and the fidelity path.

Runs are organized by problem under one root::

    results/gate_optimization/<target>/<gate_set>/n<depth>_<timestamp>.json

    res = optimize_gate_sequence(ecd_sequence(12, 80), vac, gkp_0, target_label="gkp:0")
    path = save_gate_result(res)            # derived from the problem
    res = load_gate_result(path)            # rebuilds res.sequence
    traj = res.repropagate()                # state after every gate, no optimization

    for row in list_gate_results():         # one dict per run, from the .json files only
        print(row["target_label"], row["gate_set"], row["n_gates"], row["fidelity"])
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .gate_optimization import GateBounds, GateOptResult, OptimizerConfig, TrajectoryPenalty
from .sequences import build_sequence

__all__ = [
    "DEFAULT_ROOT",
    "gate_result_path",
    "save_gate_result",
    "load_gate_result",
    "list_gate_results",
]

DEFAULT_ROOT = Path("results/gate_optimization")
_SCHEMA_VERSION = 2
_CONFIG_CLASSES = {
    "bounds": GateBounds,
    "trajectory": TrajectoryPenalty,
    "optimizer": OptimizerConfig,
}
#: Result fields stored in the ``.npz`` (``geodesic`` is flattened into it too).
_ARRAY_FIELDS = (
    "initial_state",
    "target_state",
    "final_states",
    "per_seed_fidelity",
    "adam_history",
    "trajectory",
)


def _slug(label: str) -> str:
    """A label as a directory name: ``"gkp:0"`` -> ``"gkp-0"``."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", label).strip("-") or "unlabelled"


def _plain(obj, where: str = "result"):
    """Recursively convert a value into JSON-serializable data."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _plain(asdict(obj), where)
    if isinstance(obj, dict):
        return {str(k): _plain(v, f"{where}[{k!r}]") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.generic) or hasattr(obj, "__array__"):
        arr = np.asarray(obj)
        if arr.dtype != object:
            if np.iscomplexobj(arr):
                return {"re": arr.real.tolist(), "im": arr.imag.tolist()}
            return arr.tolist()
    raise TypeError(f"cannot serialize {where} of type {type(obj).__name__}")


def _array(value) -> np.ndarray:
    """Inverse of :func:`_plain` for an array."""
    if isinstance(value, dict):
        return np.asarray(value["re"]) + 1j * np.asarray(value["im"])
    return np.asarray(value)


def gate_result_path(
    root, gate_set: str, n_gates: int, target_label: str | None = None, name: str | None = None
) -> Path:
    """Where a run for this problem lives: ``<root>/<target>/<gate_set>/<name>.json``.

    ``name`` defaults to ``n<depth>``, which makes the path a stable cache key
    for "optimize once, reload afterwards". The ``<target>`` level is omitted
    when no label is given.
    """
    folder = Path(root) / _slug(target_label) if target_label else Path(root)
    return folder / gate_set / f"{name or f'n{n_gates:02d}'}.json"


def save_gate_result(
    result: GateOptResult,
    path=None,
    *,
    root=DEFAULT_ROOT,
    overwrite: bool = False,
    **metadata,
) -> Path:
    """Write a result as ``<name>.json`` plus ``<name>.npz``.

    Parameters
    ----------
    result : GateOptResult
        Output of :func:`~gkp_optimal_control.gate_optimization.optimize_gate_sequence`
        or of :func:`load_gate_result`.
    path : str or Path, optional
        Destination ``.json`` file. By default the path is derived from the
        problem, as ``<root>/<target>/<gate_set>/n<depth>_<timestamp>.json``,
        so repeated runs never collide.
    root : str or Path
        Root for derived paths.
    overwrite : bool
        Allow replacing an existing run at ``path``.
    **metadata
        Extra JSON-serializable fields stored under ``metadata``
        (e.g. ``source_notebook="..."``).

    Returns
    -------
    Path
        The ``.json`` file written.
    """
    if not isinstance(result, GateOptResult):
        raise TypeError(f"result must be a GateOptResult, got {type(result).__name__}")
    seq = result.sequence
    if build_sequence(seq.name, seq.n_gates, seq.n_fock, **seq.options).n_params != seq.n_params:
        raise ValueError(
            f"this {seq.name} sequence cannot be rebuilt from its options {seq.options}; "
            "only sequences made by the sequences builders can be saved"
        )

    saved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if path is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = gate_result_path(
            root, seq.name, seq.n_gates, result.target_label, f"n{seq.n_gates:02d}_{stamp}"
        )
        k = 1
        while path.exists():  # two runs within the same second
            path = path.with_name(f"n{seq.n_gates:02d}_{stamp}-{k}.json")
            k += 1
    path = Path(path).with_suffix(".json")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    record = {
        "schema_version": _SCHEMA_VERSION,
        "saved_at": saved_at,
        "problem": {
            "gate_set": seq.name,
            "n_gates": seq.n_gates,
            "n_fock": seq.n_fock,
            "n_params": seq.n_params,
            "options": seq.options,
            "init_label": result.init_label,
            "target_label": result.target_label,
        },
        "outcome": {
            "fidelity": float(result.fidelity),
            "loss": float(result.loss),
            "leakage": float(result.leakage),
            "qubit_purity": result.qubit_purity,
            "best_seed": int(result.best_seed),
            "penalties": result.penalties,
            "polish_info": result.polish_info,
        },
        "config": result.config,
        "gate_parameters": dict(result.params),
        "flat_params": result.flat_params,
        "metadata": {**result.metadata, **metadata},
        "arrays": path.with_suffix(".npz").name,
    }
    text = json.dumps(_plain(record), indent=2) + "\n"

    arrays = {name: getattr(result, name) for name in _ARRAY_FIELDS}
    arrays.update({f"geodesic.{k}": v for k, v in result.geodesic.items()})
    arrays = {k: np.asarray(v) for k, v in arrays.items() if v is not None}

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.write_text(text)
    return path


def load_gate_result(path) -> GateOptResult:
    """Load a run written by :func:`save_gate_result`.

    The result carries a rebuilt ``sequence``, so ``result.repropagate()``,
    :func:`~gkp_optimal_control.gate_optimization.sequence_history` and warm
    starts via ``optimize_gate_sequence(result.sequence, ..., params0=result.flat_params)``
    work as on a live result.
    """
    path = Path(path).with_suffix(".json")
    record = json.loads(path.read_text())
    if record.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported gate-result schema {record.get('schema_version')!r} "
            f"(expected {_SCHEMA_VERSION})"
        )
    with np.load(path.with_name(record["arrays"])) as npz:
        arrays = {k: npz[k] for k in npz.files}
    problem, outcome = record["problem"], record["outcome"]
    seq = build_sequence(
        problem["gate_set"], problem["n_gates"], problem["n_fock"], **problem["options"]
    )
    config = dict(record["config"])
    for key, cls in _CONFIG_CLASSES.items():
        if key in config:
            config[key] = cls(**config[key])
    return GateOptResult(
        gate_set=problem["gate_set"],
        n_gates=problem["n_gates"],
        n_fock=problem["n_fock"],
        fidelity=outcome["fidelity"],
        loss=outcome["loss"],
        leakage=outcome["leakage"],
        params={k: _array(v) for k, v in record["gate_parameters"].items()},
        flat_params=_array(record["flat_params"]),
        final_states=arrays["final_states"],
        per_seed_fidelity=arrays["per_seed_fidelity"],
        best_seed=outcome["best_seed"],
        adam_history=arrays["adam_history"],
        sequence=seq,
        polish_info=outcome["polish_info"],
        config=config,
        trajectory=arrays.get("trajectory"),
        penalties=outcome["penalties"],
        qubit_purity=outcome["qubit_purity"],
        geodesic={k.split(".", 1)[1]: v for k, v in arrays.items() if k.startswith("geodesic.")},
        initial_state=arrays.get("initial_state"),
        target_state=arrays.get("target_state"),
        init_label=problem["init_label"],
        target_label=problem["target_label"],
        metadata=record["metadata"],
    )


def list_gate_results(root=DEFAULT_ROOT) -> list[dict]:
    """Summarize every run under ``root``, newest first.

    Reads only the ``.json`` files, so it stays fast for large sweeps. Each row
    has ``path``, ``saved_at``, ``target_label``, ``init_label``, ``gate_set``,
    ``n_gates``, ``n_fock``, ``n_params``, ``fidelity``, ``leakage``, ``loss``,
    ``qubit_purity``, ``options`` and ``metadata``; pass the list to
    ``pandas.DataFrame`` for filtering and plots. Files that are not gate
    results are skipped.
    """
    rows = []
    for path in Path(root).rglob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("schema_version") != _SCHEMA_VERSION:
            continue
        problem, outcome = record["problem"], record["outcome"]
        rows.append(
            {
                "path": path,
                "saved_at": record["saved_at"],
                **{k: problem[k] for k in ("target_label", "init_label", "gate_set")},
                **{k: problem[k] for k in ("n_gates", "n_fock", "n_params")},
                **{k: outcome[k] for k in ("fidelity", "leakage", "loss", "qubit_purity")},
                "options": problem["options"],
                "metadata": record["metadata"],
            }
        )
    return sorted(rows, key=lambda row: (row["saved_at"], str(row["path"])), reverse=True)
