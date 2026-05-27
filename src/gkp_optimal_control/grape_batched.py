from __future__ import annotations

import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import jit, lax
from jax.scipy.linalg import expm

# Reuse the problem-definition dataclasses from the serial module.
from gkp_optimal_control.grape import FourierBand, Penalties, System, TimeGrid

# ---------------------------------------------------------------------------
# Pure cost function — no closures, everything passed explicitly
# ---------------------------------------------------------------------------


class StaticCfg(NamedTuple):
    """Shapes and integer settings that must be compile-time constants.

    These are the things that, if changed, would force a recompile. Keeping
    them in a hashable NamedTuple makes them safe to pass as ``static_argnums``.
    """

    n_steps: int
    n_controls: int
    boundary_n_zero: int


def _params_to_pulse(params, freq_indices, n_steps, n_controls):
    """Fourier-coefficient params -> time-domain pulse.

    params:       (n_controls, n_allowed, 2)
    freq_indices: (n_allowed,) int  — indices of allowed frequencies in the
                  length-``n_steps`` FFT spectrum. Integer (not boolean)
                  because boolean ``.at[]`` requires concrete masks, which
                  blocks vmap over the mask.
    returns:      (n_controls, n_steps) real
    """
    spectrum = jnp.zeros((n_controls, n_steps), dtype=jnp.complex128)
    coeffs = params[..., 0] + 1j * params[..., 1]
    spectrum = spectrum.at[:, freq_indices].set(coeffs)
    return jnp.fft.ifft(spectrum, axis=-1).real * n_steps


def _forward_evolve(pulse, dt, psi_0, h_drift, h_controls):
    """Piecewise-constant evolution via ``expm``. Same as the serial version."""

    def step(psi, eps_k):
        h_sys = h_drift + jnp.einsum("c,cij->ij", eps_k, h_controls)
        u_sys = expm(-1j * dt * h_sys)
        return u_sys @ psi, None

    psi_f, _ = lax.scan(step, psi_0, pulse.T)
    return psi_f


def _amp_pen(pulse, eps_max):
    excess = jnp.maximum(jnp.abs(pulse) - eps_max, 0.0)
    return jnp.sum(excess**2)


def _deriv_pen(pulse):
    return jnp.sum(jnp.diff(pulse, axis=-1) ** 2)


def _bdry_pen(pulse, n_zero):
    return jnp.sum(pulse[:, :n_zero] ** 2) + jnp.sum(pulse[:, -n_zero:] ** 2)


def cost_pure(
    params,
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
):
    """Loss for one GRAPE problem. Pure function — vmappable on all leading args."""
    pulse = _params_to_pulse(params, freq_indices, static.n_steps, static.n_controls)
    psi_f = _forward_evolve(pulse, dt, psi_init, h_drift, h_controls)
    fid = jnp.abs(jnp.vdot(psi_targ, psi_f)) ** 2

    loss = -fid
    loss = loss + amp_w * _amp_pen(pulse, eps_max)
    loss = loss + deriv_w * _deriv_pen(pulse)
    loss = loss + bdry_w * _bdry_pen(pulse, static.boundary_n_zero)
    return loss


# ---------------------------------------------------------------------------
# Single-problem optimizer (the unit that gets vmapped)
# ---------------------------------------------------------------------------


def _solve_one(
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
    static,
    maxiter,
    rtol,
    atol,
):
    """Run optimistix BFGS on one problem. All array args are unbatched here."""

    def fn(p, _):
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

    solver = optx.BFGS(rtol=rtol, atol=atol)
    sol = optx.minimise(
        fn,
        solver,
        params0,
        max_steps=maxiter,
        throw=False,  # don't raise on non-convergence; we'll report it
    )

    # Re-evaluate at the solution so the returned diagnostics are exact.
    pulse = _params_to_pulse(sol.value, freq_indices, static.n_steps, static.n_controls)
    psi_f = _forward_evolve(pulse, dt, psi_init, h_drift, h_controls)
    fid = jnp.abs(jnp.vdot(psi_targ, psi_f)) ** 2
    final_loss = (
        -fid
        + amp_w * _amp_pen(pulse, eps_max)
        + deriv_w * _deriv_pen(pulse)
        + bdry_w * _bdry_pen(pulse, static.boundary_n_zero)
    )

    return {
        "params": sol.value,
        "pulse": pulse,
        "psi_final": psi_f,
        "F": fid,
        "loss": final_loss,
        "n_steps_taken": sol.stats.get("num_steps", jnp.array(-1)),
        "converged": sol.result == optx.RESULTS.successful,
    }


# ---------------------------------------------------------------------------
# Batched driver
# ---------------------------------------------------------------------------


def run_grape_batched(
    system: System,
    time_grid: TimeGrid,
    bands: list[FourierBand],
    penalties: Penalties,
    *,
    params0_batch: np.ndarray | None = None,
    n_seeds_per_band: int = 1,
    seed: int = 0,
    init_scale: float = 0.05,
    maxiter: int = 800,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    verbose: bool = True,
):
    """Solve a batch of GRAPE problems concurrently via vmap + optimistix.

    Each element of ``bands`` defines one problem (one set of allowed
    Fourier modes). If ``n_seeds_per_band > 1``, every band gets that many
    random restarts and the best result per band is reported (the rest are
    also returned).

    All problems share the same ``system``, ``time_grid``, ``penalties``, and
    array shapes. They differ only in the frequency mask (and the initial
    parameters).

    Parameters
    ----------
    bands :
        List of ``FourierBand`` instances. Each must produce a mask of length
        ``time_grid.n_steps`` with the same number of ``True`` entries (so
        that ``param_shape`` is shared across the batch). If the bands you
        want to sweep don't satisfy that, see the docstring of this module.
    params0_batch :
        Optional batched initial params of shape ``(batch, *param_shape)``
        where ``batch = len(bands) * n_seeds_per_band``. If ``None``, random
        params are drawn.
    n_seeds_per_band :
        Multistart count. Useful for escaping local minima.

    Returns
    -------
    dict with keys (all shaped ``(len(bands), n_seeds_per_band, ...)``
    except ``best_*`` which collapse the seeds axis):
        ``params``, ``pulse``, ``psi_final``, ``F``, ``loss``, ``converged``,
        ``best_idx``, ``best_F``, ``best_pulse``, ``best_params``.
    """
    n_bands = len(bands)
    n_seeds = n_seeds_per_band
    batch = n_bands * n_seeds

    # Sanity-check that all bands produce the same number of allowed freqs.
    n_allowed_each = [b.n_allowed(time_grid) for b in bands]
    n_allowed = n_allowed_each[0]
    if any(n != n_allowed for n in n_allowed_each):
        raise ValueError(
            f"All bands in a single batched call must have the same number of "
            f"allowed frequencies (got {n_allowed_each}). Pad your masks to a "
            f"common length, or split the sweep into constant-n_allowed groups."
        )

    param_shape = (system.n_controls, n_allowed, 2)

    # Build per-band integer indices into the length-n_steps spectrum.
    # We use integer indices (not boolean masks) because jnp's boolean
    # `.at[]` requires the mask to be concrete at trace time, which is
    # incompatible with vmap over the per-band frequency selection.
    indices_per_band = jnp.stack(
        [jnp.where(b.mask(time_grid), size=n_allowed)[0] for b in bands]
    )  # (n_bands, n_allowed) int
    indices_batch = jnp.repeat(indices_per_band, n_seeds, axis=0)  # (batch, n_allowed)

    # Initial params: either user-supplied or random. Use a fresh name so the
    # type is unambiguous to static checkers (the input `params0_batch` is
    # `np.ndarray | None`; the value we actually feed to vmap is a jax.Array).
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

    # vmap _solve_one over (params0, freq_indices). Everything else is shared.
    solve_batched = jax.vmap(
        _solve_one,
        in_axes=(
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    solve_jit = jit(solve_batched, static_argnums=(11, 12, 13, 14))

    if verbose:
        print(f"Batched GRAPE: {n_bands} bands × {n_seeds} seeds = {batch} problems")
        print(f"  n_steps={time_grid.n_steps}, param_shape={param_shape}, dim={system.dim}")
        print("  Compiling + solving...")
    t0 = time.time()

    out = solve_jit(
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
        maxiter,
        rtol,
        atol,
    )
    # Force compute (optimistix returns lazily until we touch the arrays).
    jax.block_until_ready(out["F"])
    elapsed = time.time() - t0

    if verbose:
        f_all = np.asarray(out["F"]).reshape(n_bands, n_seeds)
        print(f"  Done in {elapsed:.1f}s ({elapsed / batch:.2f}s per problem amortized)")
        print(
            f"  Fidelities — best/median/worst across batch: "
            f"{f_all.max():.4f} / {np.median(f_all):.4f} / {f_all.min():.4f}"
        )
        conv = np.asarray(out["converged"]).reshape(n_bands, n_seeds)
        print(f"  Converged: {conv.sum()}/{batch}")

    # Reshape (batch, ...) -> (n_bands, n_seeds, ...) and pick best seed per band.
    def reshape(x):
        x = np.asarray(x)
        return x.reshape(n_bands, n_seeds, *x.shape[1:])

    # Annotated as dict[str, Any] because we mix ndarrays with a scalar
    # ("elapsed_seconds") below; without this, pyright infers dict[str, ndarray]
    # from the comprehension and rejects the scalar assignment.
    result: dict[str, Any] = {k: reshape(v) for k, v in out.items()}

    # Best-per-band: pick the seed with highest F.
    best_idx = result["F"].argmax(axis=1)  # (n_bands,)
    rows = np.arange(n_bands)
    result["best_idx"] = best_idx
    result["best_F"] = result["F"][rows, best_idx]
    result["best_pulse"] = result["pulse"][rows, best_idx]
    result["best_params"] = result["params"][rows, best_idx]
    result["elapsed_seconds"] = elapsed

    return result


# ---------------------------------------------------------------------------
# Convenience: a small wrapper for a (T fixed, f_max varying) Pareto slice
# ---------------------------------------------------------------------------


def bandwidth_sweep(
    system: System,
    time_grid: TimeGrid,
    f_max_values: list[float],
    penalties: Penalties,
    *,
    n_seeds: int = 1,
    seed: int = 0,
    init_scale: float = 0.05,
    maxiter: int = 800,
    verbose: bool = True,
):
    """Sweep ``f_max`` at fixed ``T`` and ``n_steps``.

    Pads each band's frequency mask to the largest ``n_allowed`` in the sweep
    by appending unused (zeroed) Fourier slots — this is what lets a single
    vmapped call cover bands of different bandwidths.
    """
    bands = [FourierBand(f_max=fm) for fm in f_max_values]
    n_allowed_each = [b.n_allowed(time_grid) for b in bands]
    # _max_allowed = max(n_allowed_each)

    # Build "padded" masks: for a band with fewer allowed freqs than the max,
    # we keep its real mask but expose extra slots that the optimizer will
    # learn to zero (they correspond to higher frequencies that contribute
    # to neither the loss nor the pulse for that band — they're masked out).
    #
    # Concretely: param_shape uses max_allowed, but in cost_pure we index
    # into spectrum via the band's *own* mask, so the extra param slots are
    # ignored. We just need them to exist so the array shapes match.
    #
    # The easy way to do this is: for each band, build a mask of length
    # n_steps that selects exactly max_allowed True positions — the band's
    # real allowed freqs plus enough higher-freq slots to pad. Those padded
    # slots will be driven to zero by the amplitude penalty (set eps on
    # those slots to 0), OR we can just live with them being free parameters
    # that the optimizer will find don't matter for fidelity. Cleanest is
    # to pad with the *same* mask everywhere up to max_allowed, but bands
    # don't allow that directly.
    #
    # Simplest correct approach: only sweep bands that all have the same
    # n_allowed (e.g., choose f_max values on the FFT frequency grid so
    # they each admit a clean number of modes), OR run separate batched
    # calls grouped by n_allowed.

    # Group by n_allowed and run one batched call per group.
    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(n_allowed_each):
        groups[n].append(i)

    if verbose and len(groups) > 1:
        print(
            f"f_max values span {len(groups)} distinct n_allowed values; "
            f"running one batched call per group."
        )

    # Result containers indexed by the original band order. Explicit typing
    # because we initialize with `None` placeholders but always overwrite them
    # with ndarrays inside the loop below — pyright can't follow that across
    # iterations and would otherwise infer `list[None]`.
    all_f = np.zeros((len(bands), n_seeds))
    all_pulses: list[np.ndarray] = [None] * len(bands)  # type: ignore[list-item]
    all_params: list[np.ndarray] = [None] * len(bands)  # type: ignore[list-item]
    all_converged = np.zeros((len(bands), n_seeds), dtype=bool)
    total_elapsed = 0.0

    for n_alw, idxs in groups.items():
        sub_bands = [bands[i] for i in idxs]
        if verbose:
            print(f"\nGroup n_allowed={n_alw}: {len(sub_bands)} bands × {n_seeds} seeds")
        sub_result = run_grape_batched(
            system,
            time_grid,
            sub_bands,
            penalties,
            n_seeds_per_band=n_seeds,
            seed=seed,
            init_scale=init_scale,
            maxiter=maxiter,
            verbose=verbose,
        )
        for local_i, global_i in enumerate(idxs):
            all_f[global_i] = sub_result["F"][local_i]
            all_pulses[global_i] = sub_result["pulse"][local_i]
            all_params[global_i] = sub_result["params"][local_i]
            all_converged[global_i] = sub_result["converged"][local_i]
        total_elapsed += sub_result["elapsed_seconds"]

    best_idx = all_f.argmax(axis=1)
    rows = np.arange(len(bands))
    return {
        "f_max": np.array(f_max_values),
        "F": all_f,
        "pulses": all_pulses,
        "params": all_params,
        "converged": all_converged,
        "best_F": all_f[rows, best_idx],
        "best_pulse": np.array([all_pulses[i][best_idx[i]] for i in range(len(bands))]),
        "elapsed_seconds": total_elapsed,
    }
