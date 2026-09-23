"""Backward-compat shim: re-exports ``gkp_optimal_control.grape_adam``."""

from gkp_optimal_control.grape import (  # noqa: F401
    _adam_loop_one,
    default_lr_schedule,
    run_grape_adam_batched,
)

__all__ = [
    "run_grape_adam_batched",
    "default_lr_schedule",
]
