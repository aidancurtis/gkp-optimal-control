import time
from _collections_abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, value_and_grad
from jax.scipy.linalg import expm
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Forward evolution: propagator products
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Penalty functions
# ---------------------------------------------------------------------------


def amplitude_penalty(pulse: jnp.ndarray, eps_max: float) -> jnp.ndarray:
    excess = jnp.maximum(jnp.abs(pulse) - eps_max, 0.0)
    return jnp.sum(excess**2)


def derivative_penalty(pulse: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.diff(pulse, axis=-1) ** 2)


def boundary_penalty(pulse: jnp.ndarray, n_zero: int = 3) -> jnp.ndarray:
    return jnp.sum(pulse[:, :n_zero] ** 2) + jnp.sum(pulse[:, -n_zero:] ** 2)


# ---------------------------------------------------------------------------
# Pulse parametrization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cost factory
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_pulse(path: str, pulse: np.ndarray, metadata: dict | None = None) -> None:
    """Save a pulse and its metadata."""
    np.savez(path, pulse=np.asarray(pulse), **(metadata or {}))


def load_pulse(path: str) -> tuple[np.ndarray, dict]:
    """Load a pulse and metadata."""
    data = np.load(path, allow_pickle=True)
    pulse = data["pulse"]
    metadata = {k: data[k] for k in data.files if k != "pulse"}
    return pulse, metadata
