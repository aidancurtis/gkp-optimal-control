"""Persist gate-optimization results to a stable JSON document.

:mod:`gate_optimization.optimize_gate_sequence` returns a
:class:`~gkp_optimal_control.gate_optimization.GateOptResult` -- a live object
full of NumPy arrays and frozen dataclasses. That object does not survive a
restart, but re-running the multi-start Adam optimization it came from is
expensive. This module gives it a stable on-disk form so a finished
optimization can be re-loaded and re-propagated **without re-optimizing**.

On-disk format
--------------
A single JSON document with these top-level categories (the ones a result is
most often inspected for, up front):

    {
      "schema_version": 1,
      "metadata": {"saved_at": "...", "source": "..."},
      "gate_set": "ecd",          # or "snap"
      "n_gates": 12,
      "n_fock": 80,
      "n_params": 37,
      "fidelity": 0.99983,
      "initial_state": <ndarray>,  # psi_init the circuit was optimized from
      "final_state": <ndarray>,    # psi_final from the best seed
      "gate_parameters": {          # labelled gate parameters (thetas, betas, ...)
      "flat_params": <ndarray>,    # the optimizable vector
      "loss": ...,
      "leakage": ...,
      "best_seed": 5,
      "per_seed_fidelity": <ndarray>,
      "adam_history": <ndarray>,
      "polish_info": {...},
      "config": {...}              # ansatz options + bounds/optimizer settings
    }

NumPy arrays are stored with their ``dtype`` and ``shape``; complex arrays
(e.g. ``betas``, ``alphas``, cavity kets) are split into ``real``/``imag``
parts so they round-trip exactly through JSON. Frozen dataclasses stored inside
``config`` (e.g. :class:`GateBounds`, :class:`OptimizerConfig`) are tagged
with their type and reconstructed on load, so a re-loaded :class:`GateResult`
is a drop-in for any code that expects a :class:`GateOptResult`.

Example
-------

::

    from gkp_optimal_control.gate_results import (
        save_gate_result, load_gate_result, gate_result_path,
    )
    from gkp_optimal_control.gate_optimization import (
        optimize_gate_sequence, sequence_history,
    )

    res = optimize_gate_sequence("ecd", 12, psi_init, gkp_0, ...)
    path = save_gate_result(
        res,
        gate_result_path("results", "ecd", 12, res.fidelity),
        initial_state=psi_init,
        source_notebook=__file__,
    )

    # Later (or in another notebook): skip the optimization entirely
    loaded = load_gate_result(path)
    traj = loaded.repropagate()            # cheap propagation from stored params
    assert np.allclose(traj[-1], loaded.final_state)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np

from gkp_optimal_control.gate_optimization import (
    GateBounds,
    GateOptResult,
    OptimizerConfig,
    build_sequence,
    sequence_history,
)

__all__ = [
    "GateResult",
    "save_gate_result",
    "load_gate_result",
    "gate_result_path",
]

_SCHEMA_VERSION = 1

# Frozen dataclasses that may appear inside a result (in `config`) and that
# need reconstructing on load so a loaded GateResult behaves like a live
# GateOptResult. Extend here if gate_optimization gains more.
_DATACLASS_REGISTRY: dict[str, type] = {
    f"{c.__module__}.{c.__name__}": c for c in (GateBounds, OptimizerConfig)
}


# ---------------------------------------------------------------------------
# Array <-> JSON conversion
# ---------------------------------------------------------------------------


def _to_array(x) -> np.ndarray | None:
    """Coerce a state/array-like (incl. a ``Qarray``) to a NumPy array."""
    if x is None:
        return None
    if hasattr(x, "array"):  # jaxquantum.Qarray exposes `.array`
        x = x.array
    return np.asarray(x)


def _encode_array(a: np.ndarray) -> dict:
    """Encode a (possibly complex) ndarray as a JSON-able dict."""
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return {
            "__array__": "complex",
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "real": np.ascontiguousarray(a.real).tolist(),
            "imag": np.ascontiguousarray(a.imag).tolist(),
        }
    return {
        "__array__": "real",
        "dtype": str(a.dtype),
        "shape": list(a.shape),
        "data": np.ascontiguousarray(a).tolist(),
    }


def _decode_array(o: dict) -> np.ndarray:
    dtype = np.dtype(o["dtype"])
    shape = tuple(o["shape"])
    if o["__array__"] == "complex":
        arr = np.array(o["real"], dtype=dtype) + 1j * np.array(o["imag"], dtype=dtype)
    else:
        arr = np.array(o["data"], dtype=dtype)
    return np.ascontiguousarray(arr).reshape(shape)


def _to_jsonable(obj):
    """Recursively convert a result field tree into JSON-serializable data."""
    if isinstance(obj, np.ndarray):
        return _encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        return {
            "__dataclass__": f"{cls.__module__}.{cls.__name__}",
            "data": {k: _to_jsonable(v) for k, v in asdict(obj).items()},
        }
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(obj):
    """Inverse of :func:`_to_jsonable`."""
    if isinstance(obj, dict):
        tag = obj.get("__dataclass__")
        if tag is not None:
            cls = _DATACLASS_REGISTRY.get(tag)
            if cls is None:
                raise ValueError(
                    f"cannot reconstruct dataclass {tag!r}; is the originating "
                    "class registered in _DATACLASS_REGISTRY?"
                )
            return cls(**_from_jsonable(obj["data"]))
        if obj.get("__array__") in ("real", "complex"):
            return _decode_array(obj)
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class GateResult(GateOptResult):
    """A :class:`GateOptResult` made durable on disk.

    Subclasses :class:`GateOptResult` so it is a drop-in for any code expecting
    one (e.g.
    :func:`~gkp_optimal_control.gate_optimization.sequence_history`), and
    additionally carries the ``initial_state`` the optimizer was started from,
    plus bookkeeping fields, so a re-loaded result can be re-propagated without
    re-optimizing.

    Attributes
    ----------
    initial_state : ndarray or Qarray, optional
        The cavity ket the circuit was optimized from. Needed to re-derive the
        state trajectory via :meth:`repropagate`; not stored by
        :class:`GateOptResult` itself.
    schema_version : int
        On-disk schema version (forward-compatible).
    saved_at : str
        UTC timestamp written by :func:`save_gate_result`.
    source : str
        Where the record came from (e.g. a notebook path).
    """

    initial_state: np.ndarray | None = None
    schema_version: int = _SCHEMA_VERSION
    saved_at: str = ""
    source: str = ""

    def sequence(self):
        """The :class:`GateSequence` this result was optimized for.

        Rebuilt from the stored ansatz configuration, so it needs no live
        optimizer state.
        """
        cfg = self.config
        bounds = cfg.get("bounds")
        n_leak = bounds.n_leak if bounds is not None else GateBounds().n_leak
        return build_sequence(
            self.gate_set,
            self.n_gates,
            self.n_fock,
            disp_method=cfg.get("disp_method", "expm"),
            echoed=cfg.get("echoed", True),
            qubit_target=cfg.get("qubit_target", "ground"),
            n_snap=cfg.get("n_snap"),
            n_leak=n_leak,
        )

    def repropagate(self, psi_init=None) -> np.ndarray:
        """Re-derive the state trajectory from stored parameters (no optimization).

        Returns the same array as
        :func:`~gkp_optimal_control.gate_optimization.sequence_history`: the
        state after every gate, shape ``(2 * n_gates + 2, K, ...)``. The final
        frame matches ``self.final_state`` (up to float round-off), so loading a
        result and re-propagating reproduces the optimized circuit's output
        without paying for the optimization again.
        """
        psi = self.initial_state if psi_init is None else psi_init
        if psi is None:
            raise ValueError("this result has no initial_state; pass psi_init explicitly")
        return sequence_history(self, psi)


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
    overwrite: bool = False,
    **metadata,
) -> Path:
    """Serialize a gate-optimization result to a JSON file.

    Parameters
    ----------
    result : GateOptResult
        Output of :func:`~gkp_optimal_control.gate_optimization.optimize_gate_sequence`
        (a :class:`GateResult` works too, and reuses its ``initial_state``).
    path : str or Path
        Destination; a missing ``.json`` suffix is added automatically.
    initial_state :
        The cavity ket the optimization started from. ``GateOptResult`` does not
        store this, so pass it explicitly (``psi_init``) to make the saved
        record fully re-propagatable. If ``result`` is already a
        :class:`GateResult` with an ``initial_state``, that is used.
    overwrite : bool
        Allow overwriting an existing file.
    **metadata
        Extra fields stored verbatim under ``metadata`` (e.g.
        ``source_notebook=__file__``).

    Returns
    -------
    Path
        The file written.
    """
    if not isinstance(result, GateOptResult):
        raise TypeError(f"result must be a GateOptResult, got {type(result).__name__}")
    p = Path(path)
    if p.suffix != ".json":
        p = p.with_suffix(".json")
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} exists; pass overwrite=True to replace it")

    init = _to_array(initial_state)
    if init is None and isinstance(result, GateResult):
        init = _to_array(result.initial_state)

    record = {
        "schema_version": _SCHEMA_VERSION,
        "metadata": {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "gkp_optimal_control.gate_results",
            **metadata,
        },
        "gate_set": result.gate_set,
        "n_gates": result.n_gates,
        "n_fock": result.n_fock,
        "n_params": int(np.asarray(result.flat_params).size),
        "fidelity": float(result.fidelity),
        "initial_state": _to_jsonable(init),
        "final_state": _to_jsonable(result.final_states),
        "gate_parameters": {k: _to_jsonable(v) for k, v in result.params.items()},
        "flat_params": _to_jsonable(result.flat_params),
        "loss": float(result.loss),
        "leakage": float(result.leakage),
        "best_seed": int(result.best_seed),
        "per_seed_fidelity": _to_jsonable(result.per_seed_fidelity),
        "adam_history": _to_jsonable(result.adam_history),
        "polish_info": _to_jsonable(result.polish_info),
        "config": _to_jsonable(result.config),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2) + "\n")
    return p


def load_gate_result(path) -> GateResult:
    """Load a gate-optimization result from JSON (see :func:`save_gate_result`).

    Returns a :class:`GateResult` that is a drop-in for the original
    :class:`~gkp_optimal_control.gate_optimization.GateOptResult`, with
    ``initial_state`` populated so it can be re-propagated directly.
    """
    p = Path(path)
    if p.suffix != ".json":
        p = p.with_suffix(".json")
    record = json.loads(p.read_text())
    schema = record.get("schema_version")
    if schema not in (None, _SCHEMA_VERSION):
        raise ValueError(
            f"unsupported GateResult schema version {schema!r} (expected {_SCHEMA_VERSION})"
        )
    metadata = record.get("metadata", {})

    return GateResult(
        gate_set=record["gate_set"],
        n_gates=int(record["n_gates"]),
        n_fock=int(record["n_fock"]),
        fidelity=float(record["fidelity"]),
        loss=float(record["loss"]),
        leakage=float(record["leakage"]),
        params={k: _from_jsonable(v) for k, v in record["gate_parameters"].items()},
        flat_params=_from_jsonable(record["flat_params"]),
        final_states=_from_jsonable(record["final_state"]),
        per_seed_fidelity=_from_jsonable(record["per_seed_fidelity"]),
        best_seed=int(record["best_seed"]),
        adam_history=_from_jsonable(record["adam_history"]),
        polish_info=_from_jsonable(record.get("polish_info", {})),
        config=_from_jsonable(record.get("config", {})),
        initial_state=(
            _from_jsonable(record["initial_state"])
            if record.get("initial_state") is not None
            else None
        ),
        schema_version=int(record.get("schema_version", _SCHEMA_VERSION)),
        saved_at=metadata.get("saved_at", ""),
        source=metadata.get("source", ""),
    )
