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

- **`gkp_optimal_control/`** — a small Python package built on
  [QuTiP](https://qutip.org) providing:
  - cardinal GKP logical-state constructors with a tunable finite-energy
    envelope,
  - the quantum-brachistochrone Hamiltonian between arbitrary pure states,
  - Wigner, photon-number, and control-pulse plotting helpers with a shared
    publication style,
  - Wigner-trajectory animation (MP4 / GIF).

- **`notebooks/`** — runnable explorations:
  - `foundations/` — Wigner portraits of cardinal GKP states, the Kerr
    oscillator spectrum, and other building blocks.
  - `brachistochrone/` — time-optimal state preparation from vacuum,
    time-optimal logical single-qubit gates, and Mandelstam–Tamm minimum
    times as a function of GKP squeezing.

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
