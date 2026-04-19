import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


def set_plot_style():
    """Apply serif/Times matplotlib defaults suitable for publication figures."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_wigner_function(
    state, x_bound, y_bound, ax=None, grid_points=200, add_colorbar=True
):
    """Plot the Wigner function of a bosonic state as a filled contour map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    xvec = np.linspace(-x_bound, x_bound, grid_points)
    yvec = np.linspace(-y_bound, y_bound, grid_points)
    wigner = qt.wigner(state, xvec, yvec)

    if not isinstance(wigner, np.ndarray):
        raise ValueError("Could not compute Wigner function for the given state.")

    wmax = float(np.abs(wigner).max())
    if wmax == 0.0:
        norm = mpl_colors.TwoSlopeNorm(vmin=-1e-12, vcenter=0.0, vmax=1e-12)
    else:
        norm = mpl_colors.TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)

    x, y = np.meshgrid(xvec, yvec)
    cf = ax.contourf(x, y, np.real(wigner), 100, cmap="RdBu_r", norm=norm)

    if add_colorbar:
        ax.figure.colorbar(cf, ax=ax)

    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_aspect("equal")
    return ax


def plot_photon_distribution(state, ax=None, y_lim=None, max_n=None):
    """Plot the photon-number probability distribution P(n) of a state."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    rho = qt.ket2dm(state) if state.type == "ket" else state

    photon_probs = np.real(rho.diag())
    if max_n is not None:
        photon_probs = photon_probs[:max_n]
    ns = np.arange(len(photon_probs))

    ax.bar(
        ns,
        photon_probs,
        width=1.0,
        align="center",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Photon number n")
    ax.set_ylabel("$P(n)$")
    ax.set_title("Photon Number Distribution")
    ax.set_xlim(-0.5, len(ns) - 0.5)

    if y_lim is not None:
        ax.set_ylim(0, y_lim)

    return ax


def plot_pulse(result, title=None, labels=None, ax=None, show_initial=False):
    """Plot control pulses from a QuTiP-QTRL optimisation result."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n_ctrls = result.final_amps.shape[1]

    if labels is None:
        labels = [f"Control {j}" for j in range(n_ctrls)]
    elif len(labels) != n_ctrls:
        raise ValueError(
            f"Expected {n_ctrls} labels to match the number of controls, got {len(labels)}."
        )

    for j in range(n_ctrls):
        final = np.hstack((result.final_amps[:, j], result.final_amps[-1, j]))
        ax.step(result.time, final, where="post", label=labels[j])

        if show_initial:
            initial = np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j]))
            ax.step(
                result.time,
                initial,
                where="post",
                linestyle="--",
                alpha=0.5,
                label=f"{labels[j]} (initial)",
            )

    if title:
        ax.set_title(title)

    ax.set_xlabel("Time")
    ax.set_ylabel("Control amplitude")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# State generation
# ---------------------------------------------------------------------------


def generate_cat_state(N, alpha):
    """Return normalised even and odd Schrödinger-cat states |α⟩ ± |−α⟩."""
    even_cat = qt.coherent(N, alpha) + qt.coherent(N, -alpha)
    odd_cat = qt.coherent(N, alpha) - qt.coherent(N, -alpha)
    return even_cat.unit(), odd_cat.unit()


def generate_gkp_state(N, alpha, beta, delta, cutoff):
    """Return the two finite-energy GKP logical states |0_L⟩ and |1_L⟩."""
    gkps = [qt.zero_ket(N), qt.zero_ket(N)]
    vac = qt.basis(N, 0)
    envelope = 0.5 * (1.0 - np.exp(-2.0 * delta**2))

    for k in range(-cutoff, cutoff + 1):
        for j in range(-cutoff, cutoff + 1):
            for i in range(2):
                displacement = (2 * k + i) * alpha + j * beta
                d_op = qt.displace(N, displacement)
                peak = np.exp(-1j * np.pi * (k + i / 2) * j) * d_op * vac
                weight = np.exp(-envelope * np.abs(displacement) ** 2)
                gkps[i] += weight * peak

    gkp_0, gkp_1 = gkps
    return gkp_0.unit(), gkp_1.unit()


# ---------------------------------------------------------------------------
# Wigner distributions from solver trajectories
# ---------------------------------------------------------------------------


def generate_wigner_distributions(result, x_bound, y_bound, grid_points=100):
    """Compute one Wigner distribution per state in a QuTiP solver result."""
    xvec = np.linspace(-x_bound, x_bound, grid_points)
    yvec = np.linspace(-y_bound, y_bound, grid_points)
    return [qt.wigner(state, xvec, yvec) for state in result.states]


# ---------------------------------------------------------------------------
# Hamiltonian assembly
# ---------------------------------------------------------------------------


def generate_hamiltonian(drift, controls, pulses, pulse_time):
    """Assemble a QuTiP list-format time-dependent Hamiltonian and matching tlist."""
    controls = list(controls)
    if pulses.ndim != 2:
        raise ValueError(
            f"pulses must be 2D with shape (n_timesteps, n_controls); got {pulses.shape}."
        )
    if pulses.shape[1] != len(controls):
        raise ValueError(
            f"pulses has {pulses.shape[1]} control channels but {len(controls)} "
            f"control Hamiltonians were provided."
        )

    h_tot = [drift]
    for i, hc in enumerate(controls):
        h_tot.append([hc, pulses[:, i]])

    tlist = np.linspace(0.0, pulse_time, pulses.shape[0])
    return h_tot, tlist
