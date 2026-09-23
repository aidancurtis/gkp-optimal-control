"""Backward-compat shim: re-exports ``gkp_optimal_control.grape_adam_padded``."""

from gkp_optimal_control.grape import (  # noqa: F401
    _adam_loop_padded_one,
    _params_to_pulse_padded,
    cost_pure_padded,
    run_grape_adam_batched_padded,
)

__all__ = [
    "run_grape_adam_batched_padded",
]
