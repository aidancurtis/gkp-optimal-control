"""Persist gate-optimization results to a JSON document.

Re-running the multi-start optimization behind a
:class:`~gkp_optimal_control.gate_optimization.GateOptResult` is expensive, so
this module gives it an on-disk form that can be re-loaded and re-propagated
without re-optimizing::

    res = optimize_gate_sequence(ecd_sequence(12, n_fock, n_leak=8), psi_init, gkp_0)
    path = save_gate_result(res, gate_result_path("results", "ecd", 12), note="first run")

    loaded = load_gate_result(path)   # rebuilds loaded.sequence from "ansatz"
    traj = loaded.repropagate()       # state after every gate, no optimization

On-disk format
--------------
One JSON object holding the result's fields (``gate_set``, ``n_gates``,
``n_fock``, ``fidelity``, ``flat_params``, ``gate_parameters``,
``initial_state``, ``final_state``, ``config``, ...) plus ``metadata`` and
``ansatz``, the builder options needed to rebuild the
:class:`~gkp_optimal_control.gate_optimization.GateSequence`.

Arrays are stored with their ``dtype`` and ``shape``, complex ones split into
``real``/``imag`` so they round-trip exactly. The config dataclasses
(:class:`GateBounds`, :class:`OptimizerConfig`, :class:`TrajectoryPenalty`) are
tagged with their type and rebuilt on load.

Older files keep loading: config fields the current classes no longer have are
dropped, displacement methods saved under their former names are mapped to the
current ones, and where a file predates recording the method it is inferred
from which one reproduces the stored final state.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from .gate_optimization import (
    GateBounds,
    GateOptResult,
    OptimizerConfig,
    TrajectoryPenalty,
    _as_ket_batch,
    build_sequence,
)

__all__ = ["save_gate_result", "load_gate_result", "gate_result_path"]

_SCHEMA_VERSION = 1
# Displacement methods under their former names.
_LEGACY_DISP = {"expm": "exact", "quadrature": "fixed"}
_CONFIG_CLASSES = {c.__name__: c for c in (GateBounds, OptimizerConfig, TrajectoryPenalty)}


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------


def _as_array(x) -> np.ndarray:
    """A state or array-like (including a ``Qarray`` or a list of them) as an ndarray."""
    if isinstance(x, (list, tuple)):
        return np.stack([_as_array(s) for s in x])
    arr = np.asarray(getattr(x, "data", x))
    if arr.dtype == object:
        raise TypeError(f"could not extract a numeric array from {type(x).__name__}")
    return arr


def _encode(obj, where: str = "result"):
    """Recursively convert a result field into JSON-serializable data."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        data = {f.name: _encode(getattr(obj, f.name), f"{where}.{f.name}") for f in fields(obj)}
        return {"__dataclass__": f"{cls.__module__}.{cls.__name__}", "data": data}
    if isinstance(obj, dict):
        return {str(k): _encode(v, f"{where}[{k!r}]") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if hasattr(obj, "__array__") or hasattr(obj, "data"):  # ndarray, jax.Array, Qarray
        try:
            arr = _as_array(obj)
        except TypeError as err:
            raise TypeError(f"cannot serialize {where}: {err}") from None
        head = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        if np.iscomplexobj(arr):
            return {
                "__array__": "complex",
                **head,
                "real": arr.real.tolist(),
                "imag": arr.imag.tolist(),
            }
        return {"__array__": "real", **head, "data": arr.tolist()}
    raise TypeError(f"cannot serialize {where} of type {type(obj).__name__}")


def _decode(obj):
    """Inverse of :func:`_encode`."""
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if obj.get("__array__") in ("real", "complex"):
        dtype, shape = np.dtype(obj["dtype"]), tuple(obj["shape"])
        if obj["__array__"] == "complex":
            arr = np.array(obj["real"], dtype=dtype) + 1j * np.array(obj["imag"], dtype=dtype)
        else:
            arr = np.array(obj["data"], dtype=dtype)
        return arr.reshape(shape)
    if "__dataclass__" in obj:
        data = _decode(obj["data"])
        cls = _CONFIG_CLASSES.get(obj["__dataclass__"].rpartition(".")[2])
        if cls is None:
            return data
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
    return {k: _decode(v) for k, v in obj.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _infer_disp_method(record: dict, ansatz: dict) -> str:
    """The displacement method that reproduces a file's stored final state.

    Files written before the method was recorded do not say which one they
    were optimized with, and the two differ at the truncation edge.
    """
    if record.get("initial_state") is None:
        return "fixed"
    errors = {}
    for method in ("fixed", "exact"):
        seq = build_sequence(
            record["gate_set"], record["n_gates"], record["n_fock"], **ansatz, disp_method=method
        )
        psi0 = seq.lift(_as_ket_batch(record["initial_state"], seq.n_fock))
        final = np.asarray(seq.propagate(record["flat_params"], psi0)[0])
        errors[method] = np.abs(final - record["final_state"]).max()
    return min(errors, key=errors.get)


def gate_result_path(
    directory,
    gate_set: str,
    n_gates: int,
    fidelity: float | None = None,
    *,
    ext: str = ".json",
) -> Path:
    """A conventional filename for a persisted gate result.

    ``gate_result_path("results", "ecd", 12, 0.9998)`` ->
    ``results/ecd_g12_fid0.9998.json``. Pass ``fidelity=None`` to omit it.
    """
    name = f"{gate_set}_g{n_gates}"
    if fidelity is not None:
        name += f"_fid{fidelity:.4f}"
    return Path(directory) / f"{name}{ext}"


def save_gate_result(
    result: GateOptResult,
    path,
    *,
    initial_state=None,
    ansatz: dict | None = None,
    overwrite: bool = False,
    **metadata,
) -> Path:
    """Serialize a gate-optimization result to a JSON file.

    Parameters
    ----------
    result : GateOptResult
        Output of :func:`~gkp_optimal_control.gate_optimization.optimize_gate_sequence`
        or of :func:`load_gate_result`.
    path : str or Path
        Destination; a missing ``.json`` suffix is added automatically.
    initial_state : optional
        Overrides ``result.initial_state``, the cavity ket(s) the optimization
        started from.
    ansatz : dict, optional
        Extra builder options, merged over those recorded on
        ``result.sequence``. Only needed for a hand-built sequence.
    overwrite : bool
        Allow overwriting an existing file.
    **metadata
        Extra fields stored under ``metadata`` (e.g. ``source_notebook="..."``).

    Returns
    -------
    Path
        The file written.
    """
    if not isinstance(result, GateOptResult):
        raise TypeError(f"result must be a GateOptResult, got {type(result).__name__}")
    p = Path(path).with_suffix(".json")
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} exists; pass overwrite=True to replace it")

    seq = result.sequence
    spec = {"n_leak": int(seq.n_leak), **seq.options, **result.ansatz, **(ansatz or {})}
    if seq.qubit_target is not None:
        spec.setdefault("qubit_target", seq.qubit_target)
    n_flat = int(np.asarray(result.flat_params).size)
    rebuilt = build_sequence(result.gate_set, result.n_gates, result.n_fock, **spec)
    if rebuilt.n_params != n_flat:
        raise ValueError(
            f"ansatz {spec} rebuilds a {result.gate_set} sequence with "
            f"{rebuilt.n_params} params, but the result has {n_flat}"
        )

    init = result.initial_state if initial_state is None else initial_state
    record = {
        "schema_version": _SCHEMA_VERSION,
        "metadata": {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "gkp_optimal_control.gate_results",
            **result.metadata,
            **metadata,
        },
        "gate_set": result.gate_set,
        "n_gates": result.n_gates,
        "n_fock": result.n_fock,
        "n_params": n_flat,
        "fidelity": float(result.fidelity),
        "ansatz": spec,
        "initial_state": None if init is None else _as_array(init),
        "final_state": result.final_states,
        "gate_parameters": dict(result.params),
        "flat_params": result.flat_params,
        "loss": float(result.loss),
        "leakage": float(result.leakage),
        "best_seed": int(result.best_seed),
        "per_seed_fidelity": result.per_seed_fidelity,
        "adam_history": result.adam_history,
        "polish_info": result.polish_info,
        "config": result.config,
        "trajectory": result.trajectory,
        "penalties": result.penalties,
        "qubit_purity": None if result.qubit_purity is None else float(result.qubit_purity),
        "geodesic": result.geodesic,
    }
    text = json.dumps(_encode(record), indent=2) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def load_gate_result(path) -> GateOptResult:
    """Load a gate-optimization result from JSON (see :func:`save_gate_result`).

    The returned result carries a rebuilt ``sequence``, so
    :func:`~gkp_optimal_control.gate_optimization.sequence_history`,
    ``result.repropagate()`` and warm starts via
    ``optimize_gate_sequence(result.sequence, ..., params0=result.flat_params)``
    work as on a live result.
    """
    record = _decode(json.loads(Path(path).with_suffix(".json").read_text()))
    schema = record.get("schema_version", _SCHEMA_VERSION)
    if schema != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported gate-result schema version {schema!r} (expected {_SCHEMA_VERSION})"
        )
    ansatz = dict(record.get("ansatz", {}))
    gate_set, n_gates, n_flat = record["gate_set"], record["n_gates"], record["flat_params"].size
    if gate_set == "snap" and n_gates > 0:  # files that predate the recorded phase count
        ansatz.setdefault("n_snap", (n_flat - 2 * (n_gates + 1)) // n_gates)
    if gate_set in ("ecd", "snap"):
        method = ansatz.get("disp_method") or _infer_disp_method(record, ansatz)
        ansatz["disp_method"] = _LEGACY_DISP.get(method, method)
    seq = build_sequence(gate_set, n_gates, record["n_fock"], **ansatz)
    if seq.n_params != n_flat:
        raise ValueError(
            f"{path}: ansatz {ansatz} rebuilds a {gate_set} sequence with {seq.n_params} "
            f"params, but the file has {n_flat}"
        )
    return GateOptResult(
        gate_set=record["gate_set"],
        n_gates=record["n_gates"],
        n_fock=record["n_fock"],
        fidelity=record["fidelity"],
        loss=record["loss"],
        leakage=record["leakage"],
        params=record["gate_parameters"],
        flat_params=record["flat_params"],
        final_states=record["final_state"],
        per_seed_fidelity=record["per_seed_fidelity"],
        best_seed=record["best_seed"],
        adam_history=record["adam_history"],
        sequence=seq,
        polish_info=record.get("polish_info", {}),
        config=record.get("config", {}),
        trajectory=record.get("trajectory"),
        penalties=record.get("penalties", {}),
        qubit_purity=record.get("qubit_purity"),
        geodesic=record.get("geodesic", {}),
        initial_state=record.get("initial_state"),
        ansatz=ansatz,
        metadata=record.get("metadata", {}),
    )
