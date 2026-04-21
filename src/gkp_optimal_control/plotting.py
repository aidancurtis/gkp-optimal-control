import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.axes import Axes

from .utils import compute_wigner


def set_plot_style() -> None:
    r"""Apply serif/Times matplotlib defaults suitable for publication figures."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{braket}\usepackage{amsmath}"


def plot_wigner(
    state: qt.Qobj | None = None,
    x_bound: float | None = None,
    y_bound: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    grid_points: int = 200,
    add_colorbar: bool = True,
    wigner: np.ndarray | None = None,
    xvec: np.ndarray | None = None,
    yvec: np.ndarray | None = None,
) -> Axes:
    r"""Plot the Wigner function of a bosonic state as a filled contour map.

    This function has two calling conventions:

    * Pass ``state`` together with ``x_bound`` and ``y_bound`` to compute the
      Wigner function on a fresh grid (via :func:`utils.compute_wigner`).
    * Pass a precomputed ``wigner`` array together with its ``xvec`` and
      ``yvec``. No Wigner computation is performed, so the plot renders on
      exactly the supplied grid. This is the path to use when previewing a
      single frame of an animation: pass that frame and the ``xvec``/``yvec``
      from :func:`utils.wigner_trajectory`.

    Parameters
    ----------
    state : qutip.Qobj, optional
        Ket or density matrix of the bosonic state. Required when ``wigner``
        is not supplied.
    x_bound, y_bound : float, optional
        Half-widths of the :math:`q`- and :math:`p`-axes. Required when
        ``wigner`` is not supplied.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    title : str, optional
        Axes title. If ``None``, no title is set.
    grid_points : int, default 200
        Number of samples along each phase-space axis when computing the
        Wigner function from ``state``. Ignored when ``wigner`` is supplied.
    add_colorbar : bool, default True
        If ``True``, attach a colorbar to the parent figure.
    wigner : numpy.ndarray, optional
        Precomputed 2D Wigner distribution of shape ``(len(yvec), len(xvec))``.
        When supplied, ``xvec`` and ``yvec`` must also be supplied and the
        Wigner function is not recomputed.
    xvec, yvec : numpy.ndarray, optional
        1D grid samples matching ``wigner``. Required when ``wigner`` is
        supplied.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If neither a ``state`` with bounds nor a precomputed ``wigner`` with
        its grid is supplied, or if the Wigner function could not be
        computed for ``state``.
    """
    if wigner is not None:
        if xvec is None or yvec is None:
            raise ValueError("When passing a precomputed wigner, xvec and yvec are required.")
    elif state is not None:
        if x_bound is None or y_bound is None:
            raise ValueError("When passing a state, x_bound and y_bound are required.")
        xvec, yvec, wigner = compute_wigner(state, x_bound, y_bound, grid_points)
    else:
        raise ValueError("Must supply either a state with bounds or a precomputed wigner.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

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
