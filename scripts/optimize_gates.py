"""Optimize a gate sequence that prepares one cavity state from another.

Command-line front end to
:func:`gkp_optimal_control.gate_optimization.optimize_gate_sequence` for a
single state transfer. Builds the two states and the circuit ansatz from the
arguments, runs the multi-start optimization, prints the summary and saves the
result as JSON with :func:`gkp_optimal_control.gate_results.save_gate_result`.

Examples
--------
Vacuum to GKP |0> with 9 ECD gates::

    uv run python scripts/optimize_gates.py --gate-set ecd --n-gates 9 \\
        --n-fock 60 --target gkp:0 --n-seeds 32 --max-disp 3.0

Fock |1> with two SNAP gates, optimizing only the lowest 8 phases::

    uv run python scripts/optimize_gates.py --gate-set snap --n-gates 2 \\
        --n-fock 20 --target fock:1 --n-snap 8

State specs
-----------
``vac``                  vacuum
``fock:N``               Fock state |N>
``coherent:ALPHA``       coherent state, e.g. ``coherent:1.5+0.5j``
``gkp:0`` / ``gkp:1``    square-lattice finite-energy GKP logical state
                         (shape set by ``--gkp-delta`` and ``--gkp-cutoff``)
``cat:even:ALPHA``       even / odd cat state, e.g. ``cat:odd:2.0``
``sqvac:R[:PHI]``        squeezed vacuum with squeezing r and optional angle
``file:PATH``            ket loaded from a ``.npy`` file of length ``n_fock``

Plot
----
After the optimization a grid of Wigner functions opens, one panel for the
input state and one after every gate that changes the cavity state (every ECD
or conditional squeezer; every displacement and SNAP), each labelled with its
fidelity to the target, with that fidelity plotted along the sequence
underneath. For the qubit gate sets the qubit is traced out. ``--save-plot``
also writes the figure as a PNG and ``--no-plot`` skips the window, e.g. when
running in a loop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import gkp_optimal_control  # noqa: F401  (enables 64-bit JAX)
from gkp_optimal_control.gate_optimization import (
    GateBounds,
    OptimizerConfig,
    optimize_gate_sequence,
    sequence_history,
)
from gkp_optimal_control.gate_results import DEFAULT_ROOT, gate_result_path, save_gate_result
from gkp_optimal_control.plotting import plot_wigner_grid, set_plot_style
from gkp_optimal_control.sequences import build_sequence
from gkp_optimal_control.states import (
    cat_states,
    coherent_state_vectors,
    fock_state,
    gkp_states,
    squeezed_vacuum,
)

GKP_ALPHA = np.sqrt(np.pi / 2)
GKP_BETA = 1j * np.sqrt(np.pi / 2)

# Ansatz options and the gate sets they apply to.
_ANSATZ_FLAGS = {
    "disp_method": ("ecd", "snap"),
    "n_snap": ("snap",),
    "qubit_target": ("ecd", "csq"),
    "echoed": ("ecd",),
}
_BOUND_FLAGS = {"max_disp": ("ecd", "snap"), "max_squeeze": ("csq",)}


def parse_state(spec: str, n_fock: int, gkp_delta: float, gkp_cutoff: int) -> np.ndarray:
    """Turn a state spec (see the module docstring) into a ``(n_fock,)`` ket."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    args = [a for a in rest.split(":") if a] if kind != "file" else [rest]

    def expect(*counts):
        if len(args) not in counts:
            raise ValueError(f"state spec {spec!r} has the wrong number of fields")

    if kind in ("vac", "vacuum"):
        expect(0)
        state = fock_state(n_fock, 0)
    elif kind == "fock":
        expect(1)
        state = fock_state(n_fock, int(args[0]))
    elif kind == "coherent":
        expect(1)
        state = coherent_state_vectors(n_fock, jnp.asarray([complex(args[0])]))[0]
    elif kind == "gkp":
        expect(1)
        if args[0] not in ("0", "1"):
            raise ValueError(f"gkp logical index must be 0 or 1, got {args[0]!r}")
        state = gkp_states(n_fock, GKP_ALPHA, GKP_BETA, gkp_delta, gkp_cutoff)[int(args[0])]
    elif kind == "cat":
        expect(2)
        if args[0] not in ("even", "odd"):
            raise ValueError(f"cat parity must be 'even' or 'odd', got {args[0]!r}")
        state = cat_states(n_fock, complex(args[1]))[args[0] == "odd"]
    elif kind == "sqvac":
        expect(1, 2)
        state = squeezed_vacuum(n_fock, float(args[0]), *(float(a) for a in args[1:]))
    elif kind == "file":
        state = np.load(Path(args[0]).expanduser())
    else:
        raise ValueError(
            f"unknown state spec {spec!r}; expected vac, fock:N, coherent:ALPHA, gkp:0|1, "
            "cat:even|odd:ALPHA, sqvac:R[:PHI] or file:PATH"
        )

    ket = np.asarray(state, dtype=complex).reshape(-1)
    if ket.size != n_fock:
        raise ValueError(f"state {spec!r} has dimension {ket.size}, expected n_fock={n_fock}")
    norm = np.linalg.norm(ket)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"state {spec!r} is not normalizable")
    return ket / norm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    problem = parser.add_argument_group("problem")
    problem.add_argument("--gate-set", required=True, choices=("ecd", "snap", "csq"))
    problem.add_argument("--n-gates", required=True, type=int, help="circuit depth")
    problem.add_argument("--n-fock", type=int, default=40, help="Fock truncation (default: 40)")

    states = parser.add_argument_group("states")
    states.add_argument("--init", default="vac", help="initial state spec (default: vac)")
    states.add_argument("--target", required=True, help="target state spec")
    states.add_argument("--gkp-delta", type=float, default=0.3, help="GKP envelope (default: 0.3)")
    states.add_argument("--gkp-cutoff", type=int, default=30, help="GKP lattice sum (default: 30)")

    opt = parser.add_argument_group("optimizer")
    opt.add_argument("--n-seeds", type=int, default=8, help="random restarts (default: 8)")
    opt.add_argument("--n-iters", type=int, default=1500, help="Adam steps/seed (default: 1500)")
    opt.add_argument("--lr", type=float, default=0.03, help="peak learning rate (default: 0.03)")
    opt.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0)")
    opt.add_argument(
        "--loss",
        default="infidelity",
        choices=("infidelity", "log_infidelity", "neg_fidelity"),
        help="loss type; log_infidelity helps beyond F = 0.99 (default: infidelity)",
    )
    opt.add_argument("--no-polish", action="store_true", help="skip the L-BFGS-B polish")
    opt.add_argument("--polish-maxiter", type=int, default=500, help="polish steps (default: 500)")
    opt.add_argument(
        "--target-fidelity", type=float, help="stop the Adam stage once a seed reaches this"
    )
    opt.add_argument(
        "--prune-every",
        type=int,
        metavar="N",
        help="every N Adam steps keep only the better half of the seeds",
    )

    bounds = parser.add_argument_group("bounds (soft penalties)")
    bounds.add_argument("--max-disp", type=float, help="cap on |beta| (ecd) or |alpha| (snap)")
    bounds.add_argument("--max-squeeze", type=float, help="cap on |r| (csq)")
    bounds.add_argument("--leakage-weight", type=float, default=1.0, help="(default: 1.0)")

    ansatz = parser.add_argument_group("ansatz (library defaults unless given)")
    ansatz.add_argument("--disp-method", choices=("fixed", "exact"), help="ecd, snap")
    ansatz.add_argument("--n-snap", type=int, help="Fock phases optimized per SNAP gate")
    ansatz.add_argument("--qubit-target", choices=("ground", "traced"), help="ecd, csq")
    ansatz.add_argument(
        "--no-echo", dest="echoed", action="store_const", const=False, help="bare CD (ecd)"
    )
    ansatz.add_argument("--n-leak", type=int, help="top Fock levels counted as leakage")

    out = parser.add_argument_group("output")
    out.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"results root; runs go to <root>/<target>/<gate-set>/ (default: {DEFAULT_ROOT})",
    )
    out.add_argument("--name", help="file name; default: n<depth>_<timestamp>")
    out.add_argument("--overwrite", action="store_true", help="replace an existing --name")
    out.add_argument("--no-save", action="store_true", help="print the summary only")
    out.add_argument("--quiet", action="store_true", help="suppress optimizer progress")

    plot = parser.add_argument_group("plot")
    plot.add_argument(
        "--save-plot",
        nargs="?",
        const=True,
        metavar="PNG",
        help="save the Wigner row as a PNG; without a path it goes next to the JSON",
    )
    plot.add_argument("--no-plot", action="store_true", help="do not open the plot window")
    plot.add_argument(
        "--wigner-bound", type=float, default=5.0, help="phase-space half-width (default: 5)"
    )
    plot.add_argument(
        "--plot-cols", type=int, metavar="N", help="panels per row (default: automatic, at most 6)"
    )
    return parser


def gate_snapshots(result, psi_init) -> tuple[list[str], list[np.ndarray]]:
    """Cavity states after every gate that acts on the cavity, with panel labels.

    SNAP circuits give kets after each displacement and SNAP. The qubit gate
    sets give cavity density matrices with the qubit traced out, taken after
    each layer: the qubit rotations in between leave the cavity state unchanged.
    """
    history = sequence_history(result, psi_init)[:, 0]  # single state pair
    n_gates = result.n_gates
    if result.gate_set == "snap":
        labels = ["input", "D 1"]
        for i in range(1, n_gates + 1):
            labels += [f"SNAP {i}", f"D {i + 1}"]
        return labels, list(history)

    if len(history) != 2 * n_gates + 2:
        raise ValueError(f"unexpected history length {len(history)} for {n_gates} gates")
    name = {"ecd": "ECD", "csq": "CSq"}[result.gate_set]
    blocks = history[[0, *range(3, 2 * n_gates + 2, 2)]]  # input, then after each layer
    states = [blk.T @ blk.conj() for blk in blocks]
    return ["input", *(f"{name} {i}" for i in range(1, n_gates + 1))], states


def plot_gate_wigners(result, psi_init, psi_targ, bound: float, n_cols: int | None = None):
    """Grid of Wigner functions along the optimized circuit; returns the figure.

    One panel per gate, each annotated with its fidelity to the target, and a
    strip underneath showing that fidelity along the sequence.
    """
    labels, states = gate_snapshots(result, psi_init)
    # <t|psi> for kets, <t|rho|t> for the traced-out density matrices.
    fidelities = [
        float(
            abs(np.vdot(psi_targ, s)) ** 2
            if s.ndim == 1
            else np.real(psi_targ.conj() @ s @ psi_targ)
        )
        for s in map(np.asarray, states)
    ]
    set_plot_style()
    return plot_wigner_grid(
        states,
        psi_targ,
        bound,
        bound,
        labels=labels,
        annotations=[f"$F = {f:.3f}$" for f in fidelities],
        curve=fidelities,
        curve_label="Fidelity to target",
        n_cols=n_cols,
        title=f"{result.gate_set.upper()}, {result.n_gates} gates, $F = {result.fidelity:.4f}$",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for flag, gate_sets in {**_ANSATZ_FLAGS, **_BOUND_FLAGS}.items():
        if getattr(args, flag) is not None and args.gate_set not in gate_sets:
            option = "--no-echo" if flag == "echoed" else f"--{flag.replace('_', '-')}"
            parser.error(f"{option} only applies to --gate-set {'/'.join(gate_sets)}")

    # With --name the destination is known up front; otherwise it is timestamped.
    path = None
    if args.name:
        path = gate_result_path(args.out_dir, args.gate_set, args.n_gates, args.target, args.name)
        if path.exists() and not args.no_save and not args.overwrite:
            # fail before spending the optimization
            parser.error(f"{path} exists; pass --overwrite or choose another --name")

    try:
        psi_init = parse_state(args.init, args.n_fock, args.gkp_delta, args.gkp_cutoff)
        psi_targ = parse_state(args.target, args.n_fock, args.gkp_delta, args.gkp_cutoff)
        ansatz = {k: getattr(args, k) for k in (*_ANSATZ_FLAGS, "n_leak")}
        ansatz = {k: v for k, v in ansatz.items() if v is not None}
        seq = build_sequence(args.gate_set, args.n_gates, args.n_fock, **ansatz)
    except (ValueError, OSError) as err:
        parser.error(str(err))

    result = optimize_gate_sequence(
        seq,
        psi_init,
        psi_targ,
        loss_type=args.loss,
        bounds=GateBounds(
            max_disp=args.max_disp,
            max_squeeze=args.max_squeeze,
            leakage_weight=args.leakage_weight,
        ),
        optimizer=OptimizerConfig(
            n_seeds=args.n_seeds,
            n_adam_iters=args.n_iters,
            peak_lr=args.lr,
            polish=not args.no_polish,
            polish_maxiter=args.polish_maxiter,
            seed=args.seed,
            target_fidelity=args.target_fidelity,
            prune_every=args.prune_every,
        ),
        init_label=args.init,
        target_label=args.target,
        verbose=not args.quiet,
    )
    print(result.summary())

    if not args.no_save:
        path = save_gate_result(
            result,
            path,
            root=args.out_dir,
            overwrite=args.overwrite,
            gkp_delta=args.gkp_delta,
            gkp_cutoff=args.gkp_cutoff,
            command=" ".join(["optimize_gates.py", *(sys.argv[1:] if argv is None else argv)]),
        )
        print(f"saved           : {path}  (+ .npz)")
    elif path is None:  # where it would have gone, for a plot saved next to it
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = gate_result_path(
            args.out_dir, args.gate_set, args.n_gates, args.target, f"n{args.n_gates:02d}_{stamp}"
        )

    if args.no_plot and not args.save_plot:
        return 0
    if args.no_plot:
        plt.switch_backend("Agg")
    fig = plot_gate_wigners(result, psi_init, psi_targ, args.wigner_bound, args.plot_cols)
    if args.save_plot:
        png = path.with_suffix(".png") if args.save_plot is True else Path(args.save_plot)
        png = png.expanduser().with_suffix(".png")
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=200)
        print(f"plot            : {png}")
    if not args.no_plot:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
