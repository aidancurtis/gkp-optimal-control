"""Adam-based batched GRAPE with cross-band padding.

Extends ``grape_batched_adam`` to support bands with *different* ``n_allowed``
in a single batched call. All bands share a padded ``n_allowed_max`` parameter
shape, with a per-band slot mask zeroing the contribution of padded slots.

This eliminates per-band-group recompiles in parameter sweeps where each
(T, f_max) cell would otherwise produce a different ``n_allowed`` and thus
a different ``param_shape``. With padding, the entire sweep compiles once.

The cost: bands with fewer real modes carry some optimizer state for slots
that produce zero output. This is a small waste of compute and memory,
typically <<1% of the total work.

Public API:
  * ``run_grape_adam_batched_padded`` — drop-in for the unpadded version,
    accepts bands with mixed ``n_allowed``.
  * ``default_lr_schedule`` — re-exported from ``grape_batched_adam``.
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
from gkp_optimal_control.grape_batched import StaticCfg, _forward_evolve

# ---------------------------------------------------------------------------
# Padded params -> pulse
# ---------------------------------------------------------------------------


def _params_to_pulse_padded(params, freq_indices, slot_mask, n_steps, n_controls):
    """Padded Fourier-coefficient params -> time-domain pulse.

    params:       (n_controls, n_allowed_max, 2)
    freq_indices: (n_allowed_max,) int  — indices into the length-``n_steps``
                  spectrum. For a band with n_real < n_allowed_max real modes,
                  positions [0:n_real] hold real indices and positions
                  [n_real:n_allowed_max] hold ``0`` (or any duplicate of an
                  earlier index — the slot_mask zeros their contribution).
    slot_mask:    (n_allowed_max,) float  — 1.0 for real slots, 0.0 for
                  padded slots.
    returns:      (n_controls, n_steps) real

    Implementation note: we use ``.at[].add`` rather than ``.at[].set`` so
    that duplicate indices (which arise from padded slots all pointing to
    index 0) are safe — both writes add to the same cell, and since the
    padded write contributes 0 after masking, only the real write matters.
    """
    spectrum = jnp.zeros((n_controls, n_steps), dtype=jnp.complex64)
    coeffs = (params[..., 0] + 1j * params[..., 1]) * slot_mask  # (n_controls, n_allowed_max)
    # scatter-add (safe under duplicate indices because padded coeffs are 0)
    spectrum = spectrum.at[:, freq_indices].add(coeffs)
    return jnp.fft.ifft(spectrum, axis=-1).real * n_steps


# ---------------------------------------------------------------------------
# Padded cost function (mirrors cost_pure from grape_batched.py)
# ---------------------------------------------------------------------------


def _amp_pen(pulse, eps_max):
    excess = jnp.maximum(jnp.abs(pulse) - eps_max, 0.0)
    return jnp.sum(excess**2)


def _deriv_pen(pulse):
    return jnp.sum(jnp.diff(pulse, axis=-1) ** 2)


def _bdry_pen(pulse, n_zero):
    return jnp.sum(pulse[:, :n_zero] ** 2) + jnp.sum(pulse[:, -n_zero:] ** 2)


def cost_pure_padded(
    params,
    freq_indices,
    slot_mask,
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
):
    """Loss for one padded GRAPE problem. Pure, vmappable."""
    pulse = _params_to_pulse_padded(
        params, freq_indices, slot_mask, static.n_steps, static.n_controls
    )
    psi_f = _forward_evolve(pulse, dt, psi_init, h_drift, h_controls)
    fid = jnp.abs(jnp.vdot(psi_targ, psi_f)) ** 2

    loss = -fid
    loss = loss + amp_w * _amp_pen(pulse, eps_max)
    loss = loss + deriv_w * _deriv_pen(pulse)
    loss = loss + bdry_w * _bdry_pen(pulse, static.boundary_n_zero)
    return loss


# ---------------------------------------------------------------------------
# Adam loop on one padded problem
# ---------------------------------------------------------------------------


def _adam_loop_padded_one(
    params0,
    freq_indices,
    slot_mask,
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
    def loss_fn(p):
        return cost_pure_padded(
            p,
            freq_indices,
            slot_mask,
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

    pulse = _params_to_pulse_padded(
        params_final, freq_indices, slot_mask, static.n_steps, static.n_controls
    )
    psi_f = _forward_evolve(pulse, dt, psi_init, h_drift, h_controls)
    fid = jnp.abs(jnp.vdot(psi_targ, psi_f)) ** 2

    return {
        "params": params_final,
        "pulse": pulse,
        "psi_final": psi_f,
        "F": fid,
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
    }


# ---------------------------------------------------------------------------
# Padded batched driver
# ---------------------------------------------------------------------------


def run_grape_adam_batched_padded(
    system: System,
    time_grid: TimeGrid,
    bands: list[FourierBand],
    penalties: Penalties,
    *,
    n_iters: int = 2000,
    learning_rate: float = 0.02,
    lr_schedule: Any = None,
    n_allowed_max: int | None = None,
    params0_batch: np.ndarray | None = None,
    n_seeds_per_band: int = 1,
    seed: int = 0,
    init_scale: float = 0.05,
    verbose: bool = True,
):
    """Solve a batch of GRAPE problems with Adam, padding bands to a common shape.

    Unlike ``run_grape_adam_batched``, this accepts bands with different
    ``n_allowed`` values and pads each to ``n_allowed_max``. Padded slots
    contribute zero to the pulse (via ``slot_mask``) and are essentially
    free of gradient signal.

    Parameters
    ----------
    n_allowed_max
        Padded width. If ``None``, defaults to ``max(b.n_allowed(time_grid)
        for b in bands)``. Setting this manually to a fixed value across all
        sweep calls is what enables single-compile sweeps — pick the maximum
        ``n_allowed`` that will appear anywhere in your full sweep.
    n_iters, learning_rate, lr_schedule, params0_batch, n_seeds_per_band,
    seed, init_scale, verbose
        Same semantics as ``run_grape_adam_batched``.

    Returns
    -------
    dict shaped ``(n_bands, n_seeds, ...)`` with keys ``params``, ``pulse``,
    ``psi_final``, ``F``, ``loss_history``, plus ``best_*`` collapsing seeds.
    """
    n_bands = len(bands)
    n_seeds = n_seeds_per_band
    batch = n_bands * n_seeds

    # Per-band real n_allowed.
    n_real_each = [b.n_allowed(time_grid) for b in bands]
    if n_allowed_max is None:
        n_allowed_max = max(n_real_each)
    if any(n > n_allowed_max for n in n_real_each):
        raise ValueError(
            f"Band has n_allowed={max(n_real_each)} > n_allowed_max="
            f"{n_allowed_max}. Increase n_allowed_max."
        )

    param_shape = (system.n_controls, n_allowed_max, 2)

    # Build per-band padded indices and slot masks.
    indices_list = []
    masks_list = []
    for b, n_real in zip(bands, n_real_each):
        real_indices = np.asarray(jnp.where(b.mask(time_grid), size=n_real)[0])
        # Pad with index 0 (safe under .at[].add since padded contribution is 0)
        padded = np.zeros(n_allowed_max, dtype=np.int32)
        padded[:n_real] = real_indices
        indices_list.append(padded)

        mask = np.zeros(n_allowed_max, dtype=np.float32)
        mask[:n_real] = 1.0
        masks_list.append(mask)

    indices_per_band = jnp.asarray(np.stack(indices_list))  # (n_bands, n_allowed_max)
    masks_per_band = jnp.asarray(np.stack(masks_list))  # (n_bands, n_allowed_max)
    indices_batch = jnp.repeat(indices_per_band, n_seeds, axis=0)
    masks_batch = jnp.repeat(masks_per_band, n_seeds, axis=0)

    # Initial params.
    if params0_batch is None:
        rng = np.random.default_rng(seed)
        params0_init = init_scale * rng.standard_normal((batch, *param_shape))
        # Zero out the padded slots in the initial guess. Harmless but cleaner;
        # avoids Adam's momentum tracking nonzero values for masked-out slots.
        for bi in range(n_bands):
            n_real = n_real_each[bi]
            for si in range(n_seeds):
                bidx = bi * n_seeds + si
                params0_init[bidx, :, n_real:, :] = 0.0
    else:
        params0_init = params0_batch
    params0_jax = jnp.asarray(params0_init)
    if params0_jax.shape != (batch, *param_shape):
        raise ValueError(
            f"params0_batch has shape {params0_jax.shape}, expected {(batch, *param_shape)}."
        )

    static = StaticCfg(
        n_steps=time_grid.n_steps,
        n_controls=system.n_controls,
        boundary_n_zero=penalties.boundary_n_zero,
    )

    if lr_schedule is None:
        lr_schedule = optax.constant_schedule(learning_rate)

    # vmap over (params0, freq_indices, slot_mask). All scalars and shared
    # arrays broadcast.
    loop_batched = jax.vmap(
        _adam_loop_padded_one,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
    )
    loop_jit = jit(loop_batched, static_argnums=(12, 13, 14))

    if verbose:
        print(f"Padded batched Adam GRAPE: {n_bands} bands × {n_seeds} seeds = {batch} problems")
        print(f"  n_allowed_max={n_allowed_max} (per-band real: {n_real_each})")
        print(
            f"  n_steps={time_grid.n_steps}, dim={system.dim}, "
            f"n_iters={n_iters}, lr={learning_rate}"
        )
        print("  Compiling + solving...")
    t0 = time.time()

    out = loop_jit(
        params0_jax,
        indices_batch,
        masks_batch,
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
    result["n_real_each"] = np.array(n_real_each)
    result["n_allowed_max"] = n_allowed_max
    return result


# ---------------------------------------------------------------------------
# Helper: compute n_allowed_max across an entire sweep grid
# ---------------------------------------------------------------------------


def compute_global_n_allowed_max(f_max_values, T_values, n_steps: int) -> int:
    """Find the largest n_allowed across an entire (T, f_max) grid.

    Use this to pick a single ``n_allowed_max`` for all calls in a sweep so
    the compile cache is hit on every cell after the first.
    """
    n_max = 0
    for T in T_values:
        tg = TimeGrid(T=float(T), n_steps=n_steps)
        for fm in f_max_values:
            b = FourierBand(f_max=float(fm))
            n_max = max(n_max, b.n_allowed(tg))
    return n_max
