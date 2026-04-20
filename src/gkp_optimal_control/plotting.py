import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.axes import Axes


def set_plot_style() -> None:
    r"""Apply serif/Times matplotlib defaults suitable for publication figures."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12


def plot_wigner(
    state: qt.Qobj,
    x_bound: float,
    y_bound: float,
    ax: Axes | None = None,
    title: str | None = None,
    grid_points: int = 200,
    add_colorbar: bool = True,
) -> Axes:
    r"""Plot the Wigner function of a bosonic state as a filled contour map.

    Parameters
    ----------
    state : qutip.Qobj
        Ket or density matrix of the bosonic state.
    x_bound : float
        Half-width of the :math:`q`-axis; the grid spans
        :math:`[-x_{\text{bound}}, x_{\text{bound}}]`.
    y_bound : float
        Half-width of the :math:`p`-axis; the grid spans
        :math:`[-y_{\text{bound}}, y_{\text{bound}}]`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    title : str, optional
        Axes title. If ``None``, no title is set.
    grid_points : int, default 200
        Number of samples along each phase-space axis.
    add_colorbar : bool, default True
        If ``True``, attach a colorbar to the parent figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If the Wigner function could not be computed for ``state``.
    """
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

    if title:
        ax.set_title(title)

    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_aspect("equal")

    return ax


def plot_photon_number(
    state: qt.Qobj,
    ax: Axes | None = None,
    title: str | None = None,
    y_lim: float | None = None,
    max_n: int | None = None,
) -> Axes:
    r"""Plot the photon-number probability distribution :math:`P(n)` of a state.

    Parameters
    ----------
    state : qutip.Qobj
        Ket or density matrix of the bosonic state.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    title : str, optional
        Axes title. If ``None``, no title is set.
    y_lim : float, optional
        Upper limit for the :math:`P(n)` axis. If ``None``, matplotlib autoscales.
    max_n : int, optional
        Truncate the distribution to photon numbers :math:`n < \text{max\_n}`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
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
    ax.set_xlim(-0.5, len(ns) - 0.5)

    if y_lim is not None:
        ax.set_ylim(0, y_lim)

    if title:
        ax.set_title(title)

    return ax


def plot_controls(
    result,
    title: str | None = None,
    labels: list[str] | None = None,
    ax: Axes | None = None,
    show_initial: bool = False,
) -> Axes:
    r"""Plot control pulses from a QuTiP-QTRL optimisation result.

    Parameters
    ----------
    result : qutip_qtrl.optimresult.OptimResult
        Optimisation result exposing ``time``, ``final_amps``, and
        ``initial_amps``.
    title : str, optional
        Axes title. If ``None``, no title is set.
    labels : list of str, optional
        One label per control channel. If ``None``, channels are labelled
        ``"Control 0"``, ``"Control 1"``, etc.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    show_initial : bool, default False
        If ``True``, overlay the initial (pre-optimisation) pulses as dashed
        lines.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If ``labels`` is provided and its length does not match the number of
        control channels in ``result``.
    """
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
