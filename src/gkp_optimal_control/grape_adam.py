"""Adam-based batched GRAPE.

Sibling to ``grape_batched.py`` (which uses optimistix BFGS). This module
swaps in ``optax.adam`` driven by a single ``lax.scan`` over a fixed number
of iterations, giving:

  * Much faster JAX compile times (no nested while-loops for line search or
    convergence checks — just a flat scan over identical Adam steps).
  * Better batch parallelism on GPUs: every problem in the batch executes
    the same fixed-size step sequence, so there is no warp divergence from
    variable-depth inner loops.

Tradeoff: Adam typically needs 3–10× more iterations than BFGS to reach the
same fidelity, and learning-rate tuning matters. Useful defaults are
provided but should be tuned on a pilot problem before a full sweep.

Reuses the building blocks from ``grape_batched``: ``cost_pure``,
``_params_to_pulse``, ``_forward_evolve``, ``StaticCfg``, etc. Only the
optimizer changes.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax

from gkp_optimal_control.grape import FourierBand, Penalties, System, TimeGrid
from gkp_optimal_control.grape_batched import (
    StaticCfg,
    _forward_evolve,
    _params_to_pulse,
    cost_pure,
)

# ---------------------------------------------------------------------------
# Single-problem Adam loop (the unit that gets vmapped)
# ---------------------------------------------------------------------------


def _adam_loop_one(
    params0,
    freq_indices,
    dt,
    h_drift,
    h_controls,
    psi_init,
    psi_targ,
    amp_w,
    deriv_w,
    bdry_w,
    eps_max,
    static: StaticCfg,
    n_iters: int,
    lr_schedule,
):
    """Run a fixed number of Adam steps on one problem.

    ``lr_schedule`` is an ``optax`` schedule (callable: step -> lr). All array
    args are unbatched here; vmap adds the batch axis.

    Returns final params, full loss history, and final pulse + fidelity.
    """

    def loss_fn(p):
        return cost_pure(
            p,
            freq_indices,
            dt,
            h_drift,
            h_controls,
            psi_init,
            psi_targ,
            amp_w,
            deriv_w,
            bdry_w,
            eps_max,
            static,
        )

    grad_fn = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(learning_rate=lr_schedule)

    init_state = optimizer.init(params0)

    def step(carry, _):
        params, opt_state = carry
        loss, grad = grad_fn(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params_final, _), loss_history = lax.scan(
        step,
        (params0, init_state),
        xs=None,
        length=n_iters,
    )

    # Final evaluation for diagnostics.
    pulse = _params_to_pulse(params_final, freq_indices, static.n_steps, static.n_controls)
    psi_f = _forward_evolve(pulse, dt, psi_init, h_drift, h_controls)
    fid = jnp.abs(jnp.vdot(psi_targ, psi_f)) ** 2

    return {
        "params": params_final,
        "pulse": pulse,
        "psi_final": psi_f,
        "F": fid,
        "loss_history": loss_history,  # (n_iters,)
        "final_loss": loss_history[-1],
    }


# ---------------------------------------------------------------------------
# Batched driver
# ---------------------------------------------------------------------------


def run_grape_adam_batched(
    system: System,
    time_grid: TimeGrid,
    bands: list[FourierBand],
    penalties: Penalties,
    *,
    n_iters: int = 2000,
    learning_rate: float = 0.02,
    lr_schedule: Any = None,
    params0_batch: np.ndarray | None = None,
    n_seeds_per_band: int = 1,
    seed: int = 0,
    init_scale: float = 0.05,
    verbose: bool = True,
):
    """Solve a batch of GRAPE problems with Adam via vmap + lax.scan.

    Parameters
    ----------
    n_iters
        Number of Adam steps. Fixed (no convergence check). Typical values
        for state-prep at n_fock~80, n_steps~500: 2000–5000.
    learning_rate
        Constant learning rate. Ignored if ``lr_schedule`` is provided.
        Quantum-control problems typically want lr in [0.005, 0.05] — much
        larger than deep-learning defaults of 1e-3.
    lr_schedule
        Optional ``optax`` schedule (callable: step -> lr). A
        cosine-with-warmup is often better than a constant lr. If ``None``,
        falls back to a constant lr.
    params0_batch, n_seeds_per_band, seed, init_scale
        Same semantics as ``grape_batched.run_grape_batched``.

    Returns
    -------
    dict shaped ``(n_bands, n_seeds, ...)`` with keys ``params``, ``pulse``,
    ``psi_final``, ``F``, ``loss_history``, plus ``best_*`` collapsing seeds.
    """
    n_bands = len(bands)
    n_seeds = n_seeds_per_band
    batch = n_bands * n_seeds

    # All bands must produce the same n_allowed (same as in the BFGS version).
    n_allowed_each = [b.n_allowed(time_grid) for b in bands]
    n_allowed = n_allowed_each[0]
    if any(n != n_allowed for n in n_allowed_each):
        raise ValueError(
            f"All bands in a single batched call must share n_allowed (got "
            f"{n_allowed_each}). Group bands by n_allowed and run separately."
        )

    param_shape = (system.n_controls, n_allowed, 2)

    # Per-band integer indices into the spectrum.
    indices_per_band = jnp.stack([jnp.where(b.mask(time_grid), size=n_allowed)[0] for b in bands])
    indices_batch = jnp.repeat(indices_per_band, n_seeds, axis=0)

    # Initial params.
    if params0_batch is None:
        rng = np.random.default_rng(seed)
        params0_init: np.ndarray = init_scale * rng.standard_normal((batch, *param_shape))
    else:
        params0_init = params0_batch
    params0_jax: jax.Array = jnp.asarray(params0_init)
    if params0_jax.shape != (batch, *param_shape):
        raise ValueError(
            f"params0_batch has shape {params0_jax.shape}, expected {(batch, *param_shape)}."
        )

    static = StaticCfg(
        n_steps=time_grid.n_steps,
        n_controls=system.n_controls,
        boundary_n_zero=penalties.boundary_n_zero,
    )

    # Build the learning-rate schedule.
    if lr_schedule is None:
        lr_schedule = optax.constant_schedule(learning_rate)

    # vmap _adam_loop_one over (params0, freq_indices).
    loop_batched = jax.vmap(
        _adam_loop_one,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
    )
    loop_jit = jit(loop_batched, static_argnums=(11, 12, 13))

    if verbose:
        print(f"Batched Adam GRAPE: {n_bands} bands × {n_seeds} seeds = {batch} problems")
        print(f"  n_steps={time_grid.n_steps}, param_shape={param_shape}, dim={system.dim}")
        print(f"  Adam: n_iters={n_iters}, lr={learning_rate}")
        print("  Compiling + solving...")
    t0 = time.time()

    out = loop_jit(
        params0_jax,
        indices_batch,
        time_grid.dt,
        system.H_drift,
        system.H_controls,
        system.psi_init,
        system.psi_targ,
        penalties.amp,
        penalties.deriv,
        penalties.boundary,
        penalties.eps_max,
        static,
        n_iters,
        lr_schedule,
    )
    jax.block_until_ready(out["F"])
    elapsed = time.time() - t0

    if verbose:
        F_all = np.asarray(out["F"]).reshape(n_bands, n_seeds)
        print(
            f"  Done in {elapsed:.1f}s ({elapsed / batch:.2f}s per problem "
            f"amortized; {elapsed / (batch * n_iters) * 1000:.2f}ms per "
            f"iter per problem)"
        )
        print(
            f"  Fidelities — best/median/worst: {F_all.max():.4f} / "
            f"{np.median(F_all):.4f} / {F_all.min():.4f}"
        )

    # Reshape (batch, ...) -> (n_bands, n_seeds, ...) and pick best.
    def reshape(x):
        x = np.asarray(x)
        return x.reshape(n_bands, n_seeds, *x.shape[1:])

    result: dict[str, Any] = {k: reshape(v) for k, v in out.items()}

    best_idx = result["F"].argmax(axis=1)
    rows = np.arange(n_bands)
    result["best_idx"] = best_idx
    result["best_F"] = result["F"][rows, best_idx]
    result["best_pulse"] = result["pulse"][rows, best_idx]
    result["best_params"] = result["params"][rows, best_idx]
    result["elapsed_seconds"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Convenience: a good default schedule for quantum control
# ---------------------------------------------------------------------------


def default_lr_schedule(n_iters: int, peak_lr: float = 0.03, warmup_frac: float = 0.05) -> Any:
    """Warmup-then-cosine-decay learning rate schedule.

    Often outperforms a constant lr for quantum control. Linearly warms up
    from 0 to ``peak_lr`` over the first ``warmup_frac`` of iterations,
    then cosine-decays to ``peak_lr / 100`` over the remainder.

    Use as:
        sched = default_lr_schedule(n_iters=3000)
        result = run_grape_adam_batched(..., n_iters=3000, lr_schedule=sched)
    """
    n_warmup = int(n_iters * warmup_frac)
    n_decay = n_iters - n_warmup
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, peak_lr, n_warmup),
            optax.cosine_decay_schedule(peak_lr, n_decay, alpha=0.01),
        ],
        boundaries=[n_warmup],
    )
