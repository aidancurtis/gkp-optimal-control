"""GRAPE (gradient ascent pulse engineering) for quantum optimal control.

Pulses are parameterized by their Fourier coefficients inside a hard frequency
band and propagated as piecewise-constant controls. Two front ends share one
cost function:

* :func:`run_grape` -- a single problem, SciPy L-BFGS-B, with progress output;
* :func:`run_grape_batched` -- many bands and random restarts in one vmapped
  optimistix BFGS solve, with :func:`bandwidth_sweep` on top of it.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import lax, value_and_grad
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

    def indices(self, time_grid: TimeGrid) -> jnp.ndarray:
        """Positions of the allowed modes in the length-``n_steps`` spectrum."""
        return jnp.where(self.mask(time_grid), size=self.n_allowed(time_grid))[0]

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
        Pulse array of shape ``(n_controls, n_steps)``.
    dt : float
        Time per slice.
    psi_0 : jnp.ndarray
        Initial ket of shape ``(dim,)``.
    h_drift, h_controls : jnp.ndarray
        Drift Hamiltonian ``(dim, dim)`` and control stack
        ``(n_controls, dim, dim)``.
    return_history : bool, default False
        If True, return ``(psi_f, history)`` where ``history`` holds the state
        after every slice, shape ``(n_steps, dim)``.
    """

    def step(psi, eps_k):
        h_sys = h_drift + jnp.einsum("c,cij->ij", eps_k, h_controls)
        psi = expm(-1j * dt * h_sys) @ psi
        return psi, (psi if return_history else None)

    psi_f, history = lax.scan(step, psi_0, pulse.T)
    return (psi_f, history) if return_history else psi_f


def _params_to_pulse(params, freq_indices, n_steps: int):
    """Fourier coefficients ``(n_controls, n_allowed, 2)`` -> real pulse ``(n_controls, n_steps)``.

    ``freq_indices`` are integer positions rather than a boolean mask so the
    band can be vmapped over.
    """
    spectrum = jnp.zeros((params.shape[0], n_steps), dtype=jnp.complex128)
    spectrum = spectrum.at[:, freq_indices].set(params[..., 0] + 1j * params[..., 1])
    return jnp.fft.ifft(spectrum, axis=-1).real * n_steps


def _evaluate(params, freq_indices, system: System, time_grid: TimeGrid, penalties: Penalties):
    """Pulse, final state, fidelity and the three penalty values for one parameter set."""
    pulse = _params_to_pulse(params, freq_indices, time_grid.n_steps)
    psi_f = forward_evolve(pulse, time_grid.dt, system.psi_init, system.H_drift, system.H_controls)
    n_zero = penalties.boundary_n_zero
    return {
        "pulse": pulse,
        "psi_final": psi_f,
        "F": jnp.abs(jnp.vdot(system.psi_targ, psi_f)) ** 2,
        "amp_penalty": jnp.sum(jnp.maximum(jnp.abs(pulse) - penalties.eps_max, 0.0) ** 2),
        "deriv_penalty": jnp.sum(jnp.diff(pulse, axis=-1) ** 2),
        "boundary_penalty": jnp.sum(pulse[:, :n_zero] ** 2) + jnp.sum(pulse[:, -n_zero:] ** 2),
    }


def _loss(out: dict, penalties: Penalties):
    return (
        -out["F"]
        + penalties.amp * out["amp_penalty"]
        + penalties.deriv * out["deriv_penalty"]
        + penalties.boundary * out["boundary_penalty"]
    )


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
) -> tuple[Any, dict, Callable]:
    """Optimize one pulse with L-BFGS-B.

    Parameters
    ----------
    params0 : ndarray, optional
        Initial Fourier coefficients, shape ``band.param_shape(system, time_grid)``.
        Random with scale ``init_scale`` if omitted.
    progress_every : int
        Print progress every N iterations. Set to 0 to show only the final summary.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
    diag : dict
        ``F``, the three penalty values, ``pulse``, ``psi_final`` and the
        progress ``history`` as ``(iter, loss, F, elapsed)`` tuples.
    params_to_pulse : callable
        Maps a parameter array to its time-domain pulse.
    """
    param_shape = band.param_shape(system, time_grid)
    freq_indices = band.indices(time_grid)

    evaluate = jax.jit(lambda p: _evaluate(p, freq_indices, system, time_grid, penalties))
    loss_and_grad = jax.jit(value_and_grad(lambda p: _loss(evaluate(p), penalties)))

    def diagnostics(flat):
        out = evaluate(jnp.asarray(flat).reshape(param_shape))
        return {k: np.asarray(v) if v.ndim else float(v) for k, v in out.items()}

    last = {"iter": 0, "val": None, "grad_norm": None}

    def objective(flat):
        val, grad = loss_and_grad(jnp.asarray(flat).reshape(param_shape))
        last["val"], last["grad_norm"] = float(val), float(jnp.linalg.norm(grad))
        return last["val"], np.asarray(grad).ravel()

    history = []
    start = time.time()
    if verbose and progress_every > 0:
        print(f"{'iter':>5}  {'loss':>12}  {'F':>8}  {'|grad|':>10}  {'elapsed':>8}")
        print("-" * 55)

    def callback(xk):
        last["iter"] += 1
        if progress_every > 0 and last["iter"] % progress_every == 0:
            fid, elapsed = diagnostics(xk)["F"], time.time() - start
            history.append((last["iter"], last["val"], fid, elapsed))
            if verbose:
                print(
                    f"{last['iter']:5d}  {last['val']:+12.6f}  {fid:8.5f}  "
                    f"{last['grad_norm']:10.3e}  {elapsed:7.1f}s"
                )

    if params0 is None:
        params0 = init_scale * np.random.default_rng(seed).standard_normal(param_shape)

    result = minimize(
        objective,
        np.asarray(params0).ravel(),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10},
    )

    diag = diagnostics(result.x)
    diag["history"] = history

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
        print(f"Total time: {time.time() - start:.1f}s")

    return result, diag, lambda p: _params_to_pulse(p, freq_indices, time_grid.n_steps)


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
) -> dict[str, Any]:
    """Solve a batch of GRAPE problems concurrently via vmap + optimistix BFGS.

    Each element of ``bands`` defines one problem; with ``n_seeds_per_band > 1``
    every band gets that many random restarts. All problems share ``system``,
    ``time_grid`` and ``penalties``, and every band must allow the same number
    of Fourier modes so the parameter arrays stack (use
    :func:`bandwidth_sweep` when they do not).

    Parameters
    ----------
    params0_batch : ndarray, optional
        Initial parameters of shape ``(len(bands) * n_seeds_per_band, *param_shape)``.
        Random with scale ``init_scale`` if omitted.

    Returns
    -------
    dict
        ``params``, ``pulse``, ``psi_final``, ``F``, ``loss``, ``converged``
        and ``n_steps_taken``, each shaped ``(len(bands), n_seeds_per_band, ...)``;
        ``best_idx``, ``best_F``, ``best_pulse`` and ``best_params`` with the
        seed axis collapsed; and ``elapsed_seconds``.
    """
    n_bands, n_seeds = len(bands), n_seeds_per_band
    batch = n_bands * n_seeds

    n_allowed_each = [b.n_allowed(time_grid) for b in bands]
    n_allowed = n_allowed_each[0]
    if any(n != n_allowed for n in n_allowed_each):
        raise ValueError(
            f"All bands in a single batched call must have the same number of "
            f"allowed frequencies (got {n_allowed_each}); use bandwidth_sweep, "
            f"which groups them."
        )
    param_shape = (system.n_controls, n_allowed, 2)

    indices = jnp.repeat(jnp.stack([b.indices(time_grid) for b in bands]), n_seeds, axis=0)
    if params0_batch is None:
        params0_batch = init_scale * np.random.default_rng(seed).standard_normal(
            (batch, *param_shape)
        )
    params0 = jnp.asarray(params0_batch)
    if params0.shape != (batch, *param_shape):
        raise ValueError(
            f"params0_batch has shape {params0.shape}, expected {(batch, *param_shape)}."
        )

    def solve_one(p0, freq_indices):
        def evaluate(p):
            return _evaluate(p, freq_indices, system, time_grid, penalties)

        sol = optx.minimise(
            lambda p, _: _loss(evaluate(p), penalties),
            optx.BFGS(rtol=rtol, atol=atol),
            p0,
            max_steps=maxiter,
            throw=False,  # report non-convergence instead of raising
        )
        out = evaluate(sol.value)
        return {
            "params": sol.value,
            "pulse": out["pulse"],
            "psi_final": out["psi_final"],
            "F": out["F"],
            "loss": _loss(out, penalties),
            "n_steps_taken": sol.stats.get("num_steps", jnp.array(-1)),
            "converged": sol.result == optx.RESULTS.successful,
        }

    if verbose:
        print(f"Batched GRAPE: {n_bands} bands × {n_seeds} seeds = {batch} problems")
        print(f"  n_steps={time_grid.n_steps}, param_shape={param_shape}, dim={system.dim}")
        print("  Compiling + solving...")
    t0 = time.time()
    out = jax.jit(jax.vmap(solve_one))(params0, indices)
    jax.block_until_ready(out["F"])
    elapsed = time.time() - t0

    result: dict[str, Any] = {
        k: np.asarray(v).reshape(n_bands, n_seeds, *v.shape[1:]) for k, v in out.items()
    }
    if verbose:
        fids = result["F"]
        print(f"  Done in {elapsed:.1f}s ({elapsed / batch:.2f}s per problem amortized)")
        print(
            f"  Fidelities — best/median/worst across batch: "
            f"{fids.max():.4f} / {np.median(fids):.4f} / {fids.min():.4f}"
        )
        print(f"  Converged: {result['converged'].sum()}/{batch}")

    best_idx = result["F"].argmax(axis=1)
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
) -> dict[str, Any]:
    """Sweep ``f_max`` at fixed ``T`` and ``n_steps``.

    Bands that allow the same number of Fourier modes are solved together in
    one :func:`run_grape_batched` call; the sweep runs one call per group.

    Returns
    -------
    dict
        ``f_max``, ``F`` and ``converged`` of shape ``(n_bands, n_seeds)``,
        ``pulses`` and ``params`` as per-band lists, ``best_F``,
        ``best_pulse`` and ``elapsed_seconds``.
    """
    bands = [FourierBand(f_max=fm) for fm in f_max_values]
    groups: dict[int, list[int]] = defaultdict(list)
    for i, band in enumerate(bands):
        groups[band.n_allowed(time_grid)].append(i)

    if verbose and len(groups) > 1:
        print(
            f"f_max values span {len(groups)} distinct n_allowed values; "
            f"running one batched call per group."
        )

    all_f = np.zeros((len(bands), n_seeds))
    all_converged = np.zeros((len(bands), n_seeds), dtype=bool)
    all_pulses: list = [None] * len(bands)
    all_params: list = [None] * len(bands)
    total_elapsed = 0.0

    for n_allowed, idxs in groups.items():
        if verbose:
            print(f"\nGroup n_allowed={n_allowed}: {len(idxs)} bands × {n_seeds} seeds")
        sub = run_grape_batched(
            system,
            time_grid,
            [bands[i] for i in idxs],
            penalties,
            n_seeds_per_band=n_seeds,
            seed=seed,
            init_scale=init_scale,
            maxiter=maxiter,
            verbose=verbose,
        )
        for local_i, global_i in enumerate(idxs):
            all_f[global_i] = sub["F"][local_i]
            all_converged[global_i] = sub["converged"][local_i]
            all_pulses[global_i] = sub["pulse"][local_i]
            all_params[global_i] = sub["params"][local_i]
        total_elapsed += sub["elapsed_seconds"]

    best_idx = all_f.argmax(axis=1)
    return {
        "f_max": np.array(f_max_values),
        "F": all_f,
        "pulses": all_pulses,
        "params": all_params,
        "converged": all_converged,
        "best_F": all_f[np.arange(len(bands)), best_idx],
        "best_pulse": np.array([all_pulses[i][best_idx[i]] for i in range(len(bands))]),
        "elapsed_seconds": total_elapsed,
    }


def save_pulse(path: str, pulse: np.ndarray, metadata: dict | None = None) -> None:
    """Save a pulse and its metadata."""
    np.savez(path, pulse=np.asarray(pulse), **(metadata or {}))


def load_pulse(path: str) -> tuple[np.ndarray, dict]:
    """Load a pulse and metadata."""
    data = np.load(path, allow_pickle=True)
    return data["pulse"], {k: data[k] for k in data.files if k != "pulse"}
