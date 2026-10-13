# GKP Optimal Control

Time-optimal preparation and state transfer for finite-energy GKP logical states.

This repository implements the [quantum brachistochrone][carlini] construction
for Gottesman–Kitaev–Preskill (GKP) bosonic codes: given an initial and target
state, it returns the constant Hamiltonian that drives the transition in the
minimum time allowed by the Mandelstam–Tamm bound,

$$
T_{\mathrm{min}} = \frac{\Omega_B}{\|H\|}, \qquad
\theta_B = \arccos |\langle \psi_i | \psi_f \rangle|.
$$

These optimal Hamiltonians serve as **benchmarks** — lower bounds on gate
time at fixed energy budget — against which realistic control schemes
(bounded drive, finite bandwidth, physically available operators) can be
compared.

[carlini]: https://doi.org/10.1103/PhysRevLett.96.060503

## What's inside

- **`gkp_optimal_control/`** — a Python package built on
  [JAX](https://jax.readthedocs.io) and
  [jaxquantum](https://github.com/EQuS/jaxquantum):
  - `states`, `operators`, `hamiltonians` — finite-energy GKP, cat, Fock and
    squeezed states; truncated-space operators, displacements and squeezers;
    model Hamiltonians.
  - `brachistochrone` — the time-optimal Hamiltonian between two pure states.
  - `sequences`, `gate_optimization`, `gate_results` — ECD, SNAP and
    conditional-squeezing gate sequences, their multi-start optimization
    (with optional seed pruning and a fidelity target), and saving,
    reloading and indexing the results.
  - `grape` — band-limited GRAPE pulse optimization.
  - `device`, `pulses`, `pulse_simulation` — unit-aware device configs,
    compilation of ECD and SNAP gate sequences into control pulses, and
    time-domain simulation of those pulses.
  - `plotting`, `animation`, `utils` — Wigner, photon-number and trajectory
    plots with a shared publication style, and MP4 / GIF animations.

- **`scripts/`** — command-line entry points. `optimize_gates.py` runs one
  gate-sequence optimization and plots a grid of Wigner functions, one panel
  per gate, with the fidelity along the sequence underneath:

  ```bash
  uv run python scripts/optimize_gates.py --gate-set ecd --n-gates 9 \
      --n-fock 60 --target gkp:0 --n-seeds 32 --save-plot
  ```

- **`notebooks/`** — runnable explorations, grouped into `foundations/`,
  `brachistochrone/`, `gate_optimization/` and `optimal_control/`.

- **`devices/`** — device parameter files for the pulse compiler.

- **`tests/`** — run with `uv run pytest tests`.

## Conventions

- **States are plain arrays.** Every state constructor returns a normalized
  1-D complex JAX array of shape `(n_fock,)`, and library functions return
  arrays too. Every function that takes a state also accepts a NumPy array, a
  column vector or a `jaxquantum.Qarray`. To call jaxquantum directly, wrap
  with `to_qarray(state)`; to get back, `to_ket(state)`.
- **Saved runs** are a small `.json` (problem, settings, gate parameters) plus
  an `.npz` (states and histories), organized as
  `results/gate_optimization/<target>/<gate_set>/n<depth>_<timestamp>.json`.
  `list_gate_results()` returns one summary row per run.

## Quickstart

```bash
git clone https://github.com/<you>/gkp-optimal-control.git
cd gkp_optimal_control
uv sync
uv run jupyter lab
```

LaTeX (e.g. TeX Live, MiKTeX) is required for figure rendering, and
`ffmpeg` for MP4 animations.

## References

- A. Carlini, A. Hosoya, T. Koike, Y. Okudaira.
  *Time-optimal quantum evolution.*
  [Phys. Rev. Lett. **96**, 060503 (2006)][carlini].
- D. Gottesman, A. Kitaev, J. Preskill.
  *Encoding a qubit in an oscillator.*
  [Phys. Rev. A **64**, 012310 (2001)](https://doi.org/10.1103/PhysRevA.64.012310).
