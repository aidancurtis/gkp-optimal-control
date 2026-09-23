"""GRAPE (gradient ascent pulse engineering) for quantum optimal control.

Consolidated implementation. This module unifies what previously spanned four
files — ``grape.py``, ``grape_batched.py``, ``grape_adam.py``, and
``grape_adam_padded.py`` — de-duplicating the shared penalty suite into a
single ``amplitude_penalty`` / ``derivative_penalty`` / ``boundary_penalty``
set. The former submodule names are no longer importable; import everything
from here.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, overload

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
from jax import jit, lax, value_and_grad
from jax.scipy.linalg import expm
from scipy.optimize import minimize


@dataclass
class System:
    """Quantum system: drift + control Hamiltonians and target states."""

    H_drift: jnp.ndarray  # (dim, dim)
    H_controls: jnp.ndarray  # (n_controls, dim, dim)
    psi_init: jnp.ndarray  # (dim,) — single state-transfer for now
    psi_targ: jnp.ndarray  # (dim,)

    @property
    def n_controls(self) -> int:
        return self.H_controls.shape[0]

    @property
    def dim(self) -> int:
        return self.H_drift.shape[0]


@dataclass
class TimeGrid:
    """Time discretization."""

    T: float
    n_steps: int

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


@dataclass
class FourierBand:
    """Fourier-space pulse parametrization with hard frequency cutoff."""

    f_max: float  # in same units as 1/dt (e.g., MHz if dt in μs)
    f_min: float = 0.0  # set > 0 to also cut DC

    def mask(self, time_grid: TimeGrid) -> jnp.ndarray:
        freqs = jnp.fft.fftfreq(time_grid.n_steps, d=time_grid.dt)
        return (jnp.abs(freqs) >= self.f_min) & (jnp.abs(freqs) <= self.f_max)

    def n_allowed(self, time_grid: TimeGrid) -> int:
        return int(jnp.sum(self.mask(time_grid)))

    def param_shape(self, system: System, time_grid: TimeGrid) -> tuple:
        return (system.n_controls, self.n_allowed(time_grid), 2)


@dataclass
class Penalties:
    """Lagrange-multiplier weights for each penalty term."""

    amp: float = 0.0
    deriv: float = 0.0
    boundary: float = 0.0
    eps_max: float = jnp.inf
    boundary_n_zero: int = 3


@overload
def forward_evolve(
    pulse: jnp.ndarray,
    dt: float,
    psi_0: jnp.ndarray,
    h_drift: jnp.ndarray,
    h_controls: jnp.ndarray,
    *,
    return_history: Literal[False] = False,
) -> jnp.ndarray: ...


@overload
def forward_evolve(
    pulse: jnp.ndarray,
    dt: float,
    psi_0: jnp.ndarray,
    h_drift: jnp.ndarray,
    h_controls: jnp.ndarray,
    *,
    return_history: Literal[True],
) -> tuple[jnp.ndarray, jnp.ndarray]: ...


def forward_evolve(
    pulse: jnp.ndarray,
    dt: float,
    psi_0: jnp.ndarray,
    h_drift: jnp.ndarray,
    h_controls: jnp.ndarray,
    *,
    return_history: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    r"""Evolve ``psi_0`` under a piecewise-constant pulse via ``expm``.

    Parameters
    ----------
    pulse : jnp.ndarray
        Pulse array of shape ``(n_controls, n_steps)``. The transpose
        ``pulse.T`` gives ``(n_steps, n_controls)`` which is scanned over.
    dt : float
        Time per slice.
    psi_0 : jnp.ndarray
        Initial ket of shape ``(dim,)``.
    h_drift, h_controls : jnp.ndarray
        Drift Hamiltonian ``(dim, dim)`` and control stack
        ``(n_controls, dim, dim)``.
    return_history : bool, default False
        If True, also return the trajectory of intermediate states
        of shape ``(n_steps, dim)``.

    Returns
    -------
    psi_f : jnp.ndarray
        Final state of shape ``(dim,)``.
    history : jnp.ndarray, optional
        Trajectory of intermediate states, only when ``return_history=True``.
    """
    if return_history:

        def step_with_history(psi, eps_k):
            h_sys = h_drift + jnp.einsum("c,cij->ij", eps_k, h_controls)
            new_psi = expm(-1j * dt * h_sys) @ psi
            return new_psi, new_psi

        psi_f, history = lax.scan(step_with_history, psi_0, pulse.T)
        return psi_f, history

    def step(psi, eps_k):
        h_sys = h_drift + jnp.einsum("c,cij->ij", eps_k, h_controls)
        new_psi = expm(-1j * dt * h_sys) @ psi
        return new_psi, None

    psi_f, _ = lax.scan(step, psi_0, pulse.T)
    return psi_f


def amplitude_penalty(pulse: jnp.ndarray, eps_max: float) -> jnp.ndarray:
    excess = jnp.maximum(jnp.abs(pulse) - eps_max, 0.0)
    return jnp.sum(excess**2)


def derivative_penalty(pulse: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.diff(pulse, axis=-1) ** 2)


def boundary_penalty(pulse: jnp.ndarray, n_zero: int = 3) -> jnp.ndarray:
    return jnp.sum(pulse[:, :n_zero] ** 2) + jnp.sum(pulse[:, -n_zero:] ** 2)


def make_params_to_pulse(
    freq_mask: jnp.ndarray, n_steps: int, n_controls: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a closure converting Fourier-space params to a time-domain pulse."""

    def params_to_pulse(params):
        spectrum = jnp.zeros((n_controls, n_steps), dtype=jnp.complex128)
        coeffs = params[..., 0] + 1j * params[..., 1]
        spectrum = spectrum.at[:, freq_mask].set(coeffs)
        return jnp.fft.ifft(spectrum, axis=-1).real * n_steps

    return params_to_pulse


def make_cost(
    system: System, time_grid: TimeGrid, band: FourierBand, penalties: Penalties
) -> tuple[Callable, Callable, Callable]:
    """Build a cost function and supporting closures for a GRAPE problem.

    Returns
    -------
    cost : callable(params, dt) -> scalar loss
    params_to_pulse : callable(params) -> time-domain pulse
    diagnostics : callable(params) -> dict with F, penalty values, pulse
    """
    freq_mask = band.mask(time_grid)
    params_to_pulse = make_params_to_pulse(freq_mask, time_grid.n_steps, system.n_controls)

    def cost(params, dt):
        pulse = params_to_pulse(params)
        psi_f = forward_evolve(pulse, dt, system.psi_init, system.H_drift, system.H_controls)
        fid = jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2

        loss = -fid
        loss += penalties.amp * amplitude_penalty(pulse, penalties.eps_max)
        loss += penalties.deriv * derivative_penalty(pulse)
        loss += penalties.boundary * boundary_penalty(pulse, penalties.boundary_n_zero)
        return loss

    def diagnostics(params):
        pulse = params_to_pulse(params)
        psi_f = forward_evolve(
            pulse,
            dt=time_grid.dt,
            psi_0=system.psi_init,
            h_drift=system.H_drift,
            h_controls=system.H_controls,
        )
        return {
            "F": float(jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2),
            "amp_penalty": float(amplitude_penalty(pulse, penalties.eps_max)),
            "deriv_penalty": float(derivative_penalty(pulse)),
            "boundary_penalty": float(boundary_penalty(pulse, penalties.boundary_n_zero)),
            "pulse": np.asarray(pulse),
            "psi_final": np.asarray(psi_f),
        }

    return cost, params_to_pulse, diagnostics


def run_grape(
    system: System,
    time_grid: TimeGrid,
    band: FourierBand,
    penalties: Penalties,
    params0: np.ndarray | None = None,
    seed: int = 0,
    init_scale: float = 0.05,
    maxiter: int = 800,
    verbose: bool = True,
    progress_every: int = 10,
):
    """Run a GRAPE optimization end to end.

    Parameters
    ----------
    progress_every : int
        Print progress every N iterations during optimization.
        Set to 0 to disable iteration-level progress (only show final summary).
    """
    cost, params_to_pulse, diagnostics = make_cost(system, time_grid, band, penalties)
    cost_and_grad = jax.jit(value_and_grad(cost))

    param_shape = band.param_shape(system, time_grid)

    # Mutable state for the callback to track progress.
    progress = {
        "iter": 0,
        "last_val": None,
        "last_grad_norm": None,
        "history": [],
        "start_time": time.time(),
    }

    def scipy_objective(flat_params, dt):
        params = flat_params.reshape(param_shape)
        val, grad = cost_and_grad(params, dt)
        progress["last_val"] = float(val)
        progress["last_grad_norm"] = float(jnp.linalg.norm(grad))
        return float(val), np.asarray(grad).ravel()

    if verbose and progress_every > 0:
        print(f"{'iter':>5}  {'loss':>12}  {'F':>8}  {'|grad|':>10}  {'elapsed':>8}")
        print("-" * 55)

    def callback(xk):
        progress["iter"] += 1
        if progress_every > 0 and progress["iter"] % progress_every == 0:
            params = xk.reshape(param_shape)
            diag = diagnostics(jnp.array(params))
            elapsed = time.time() - progress["start_time"]
            progress["history"].append(
                (
                    progress["iter"],
                    progress["last_val"],
                    diag["F"],
                    elapsed,
                )
            )
            if verbose:
                print(
                    f"{progress['iter']:5d}  "
                    f"{progress['last_val']:+12.6f}  "
                    f"{diag['F']:8.5f}  "
                    f"{progress['last_grad_norm']:10.3e}  "
                    f"{elapsed:7.1f}s"
                )

    if params0 is None:
        rng = np.random.default_rng(seed)
        params0 = init_scale * rng.standard_normal(param_shape)

    result = minimize(
        scipy_objective,
        params0.ravel(),
        args=(time_grid.dt,),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10},
    )

    diag = diagnostics(jnp.array(result.x.reshape(param_shape)))
    diag["history"] = progress["history"]

    if verbose:
        print("-" * 55)
        print(f"Final fidelity: {diag['F']:.6f}")
        print(f"Iterations: {result.nit}")
        print(f"Optimizer message: {result.message}")
        print(
            f"Penalty values: amp={diag['amp_penalty']:.3e}, "
            f"deriv={diag['deriv_penalty']:.3e}, "
            f"boundary={diag['boundary_penalty']:.3e}"
        )
        print(f"Peak amplitude: {np.abs(diag['pulse']).max():.3f}")
        print(f"Total time: {time.time() - progress['start_time']:.1f}s")

    return result, diag, params_to_pulse


def save_pulse(path: str, pulse: np.ndarray, metadata: dict | None = None) -> None:
    """Save a pulse and its metadata."""
    np.savez(path, pulse=np.asarray(pulse), **(metadata or {}))


def load_pulse(path: str) -> tuple[np.ndarray, dict]:
    """Load a pulse and metadata."""
    data = np.load(path, allow_pickle=True)
    pulse = data["pulse"]
    metadata = {k: data[k] for k in data.files if k != "pulse"}
    return pulse, metadata


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
    loss = loss + amp_w * amplitude_penalty(pulse, eps_max)
    loss = loss + deriv_w * derivative_penalty(pulse)
    loss = loss + bdry_w * boundary_penalty(pulse, static.boundary_n_zero)
    return loss


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
        + amp_w * amplitude_penalty(pulse, eps_max)
        + deriv_w * derivative_penalty(pulse)
        + bdry_w * boundary_penalty(pulse, static.boundary_n_zero)
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
        Same semantics as ``run_grape_batched``.

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
        f_all = np.asarray(out["F"]).reshape(n_bands, n_seeds)
        print(
            f"  Done in {elapsed:.1f}s ({elapsed / batch:.2f}s per problem "
            f"amortized; {elapsed / (batch * n_iters) * 1000:.2f}ms per "
            f"iter per problem)"
        )
        print(
            f"  Fidelities — best/median/worst: {f_all.max():.4f} / "
            f"{np.median(f_all):.4f} / {f_all.min():.4f}"
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
    loss = loss + amp_w * amplitude_penalty(pulse, eps_max)
    loss = loss + deriv_w * derivative_penalty(pulse)
    loss = loss + bdry_w * boundary_penalty(pulse, static.boundary_n_zero)
    return loss


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
    for b, n_real in zip(bands, n_real_each, strict=True):
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
        f_all = np.asarray(out["F"]).reshape(n_bands, n_seeds)
        print(
            f"  Done in {elapsed:.1f}s ({elapsed / batch:.2f}s per problem "
            f"amortized; {elapsed / (batch * n_iters) * 1000:.2f}ms per "
            f"iter per problem)"
        )
        print(
            f"  Fidelities — best/median/worst: {f_all.max():.4f} / "
            f"{np.median(f_all):.4f} / {f_all.min():.4f}"
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


def forward_evolve_history(pulse, dt, psi_0, h_drift, h_controls):
    """Evolve ``psi_0`` under a piecewise-constant pulse and return ``(psi_f, history)``.

    Wrapper around :func:`forward_evolve` with ``return_history=True``,
    matching the call convention used by the GKP gate notebooks.
    """
    return forward_evolve(pulse, dt, psi_0, h_drift, h_controls, return_history=True)
