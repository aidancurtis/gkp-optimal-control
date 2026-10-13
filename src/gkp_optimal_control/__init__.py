"""Optimal control of bosonic codes: speed limits, gate sequences and pulses.

The most-used entry points are re-exported here. Pulse-level tools live in the
submodules :mod:`~gkp_optimal_control.grape`, :mod:`~gkp_optimal_control.pulses`,
:mod:`~gkp_optimal_control.pulse_simulation` and :mod:`~gkp_optimal_control.device`.
"""

import jax

jax.config.update("jax_enable_x64", True)

from gkp_optimal_control.animation import animate_wigner  # noqa: E402
from gkp_optimal_control.brachistochrone import quantum_brachistochrone_hamiltonian  # noqa: E402
from gkp_optimal_control.gate_optimization import (  # noqa: E402
    GateBounds,
    GateOptResult,
    OptimizerConfig,
    TrajectoryPenalty,
    optimize_gate_sequence,
    sequence_history,
)
from gkp_optimal_control.gate_results import (  # noqa: E402
    gate_result_path,
    list_gate_results,
    load_gate_result,
    save_gate_result,
)
from gkp_optimal_control.plotting import (  # noqa: E402
    plot_photon_number,
    plot_wigner,
    plot_wigner_grid,
    plot_wigner_snapshots,
    set_plot_style,
)
from gkp_optimal_control.sequences import (  # noqa: E402
    csq_sequence,
    ecd_sequence,
    snap_sequence,
)
from gkp_optimal_control.states import (  # noqa: E402
    cat_states,
    fock_state,
    gkp_states,
    squeezed_vacuum,
)
from gkp_optimal_control.utils import (  # noqa: E402
    compute_wigner,
    to_ket,
    to_ket_batch,
    to_qarray,
    wigner_trajectory,
)

__all__ = [
    "GateBounds",
    "GateOptResult",
    "OptimizerConfig",
    "TrajectoryPenalty",
    "animate_wigner",
    "cat_states",
    "compute_wigner",
    "csq_sequence",
    "ecd_sequence",
    "fock_state",
    "gate_result_path",
    "gkp_states",
    "list_gate_results",
    "load_gate_result",
    "optimize_gate_sequence",
    "plot_photon_number",
    "plot_wigner",
    "plot_wigner_grid",
    "plot_wigner_snapshots",
    "quantum_brachistochrone_hamiltonian",
    "save_gate_result",
    "sequence_history",
    "set_plot_style",
    "snap_sequence",
    "squeezed_vacuum",
    "to_ket",
    "to_ket_batch",
    "to_qarray",
    "wigner_trajectory",
]

__version__ = "0.1.0"
