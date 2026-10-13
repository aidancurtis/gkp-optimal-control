"""Plotting utilities for bosonic states (pure JAX, no qutip)."""

import jax.numpy as jnp
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import StateLike, compute_wigner, to_qarray, wigner_trajectory


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


def _is_ket(arr: jnp.ndarray) -> bool:
    """Return True if ``arr`` is a 1D ket or column-vector ket.

    Takes a raw array, not a ``Qarray``: attribute access on a ``Qarray``
    goes through its ``__getattr__`` fallback, so ``.ndim`` there resolves to
    ``jnp.ndim`` (a function, never equal to 1 or 2) rather than an integer.
    Coerce with :func:`utils.to_qarray` and read ``.data`` before calling.
    """
    if arr.ndim == 1:
        return True
    if arr.ndim == 2 and arr.shape[1] == 1:
        return True
    return False


def _photon_probs(state: StateLike) -> np.ndarray:
    """Return the photon-number probability distribution ``P(n)`` as a numpy array.

    Accepts a ``Qarray``, a 1D ket ``(n_fock,)``, a column-vector ket
    ``(n_fock, 1)``, or a density matrix ``(n_fock, n_fock)``. For kets,
    ``P(n) = |c_n|^2``. For density matrices, ``P(n) = rho_{nn}``.
    """
    arr = jnp.asarray(to_qarray(state).data)
    if _is_ket(arr):
        probs = jnp.abs(arr.reshape(-1)) ** 2
    else:
        probs = jnp.real(jnp.diag(arr))
    return np.asarray(probs)


def plot_wigner(
    state: StateLike | None = None,
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
    state : jaxquantum.Qarray or array_like, optional
        Ket (1D or column-vector) or density matrix of the bosonic state.
        Coerced by :func:`utils.to_qarray`, so either representation works.
        Required when ``wigner`` is not supplied.
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
        its grid is supplied.
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

    # Ensure numpy arrays for matplotlib (compute_wigner returns numpy, but a
    # caller-supplied wigner may be a jax array).
    xvec = np.asarray(xvec)
    yvec = np.asarray(yvec)
    wigner = np.asarray(wigner)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    wmax = float(np.abs(wigner).max())
    if wmax == 0.0:
        norm = mpl_colors.TwoSlopeNorm(vmin=-1e-12, vcenter=0.0, vmax=1e-12)
    else:
        norm = mpl_colors.TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)

    x, y = np.meshgrid(xvec, yvec)
    cf = ax.pcolormesh(
        x,
        y,
        np.real(wigner),
        cmap="RdBu_r",
        norm=norm,
        shading="gouraud",  # smooth interpolation; use "auto" for pixelated
        rasterized=True,  # important if saving to PDF
    )

    if add_colorbar:
        # Created here, not unconditionally: append_axes always allocates an
        # Axes, so hoisting it above this branch leaves an empty white box on
        # every add_colorbar=False call and perturbs tight_layout.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax.figure.colorbar(cf, ax=ax, cax=cax)
        cbar.set_label(r"$W(q,p)$")

    if title:
        ax.set_title(title)

    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_aspect("equal")

    return ax


def plot_photon_number(
    state: StateLike,
    ax: Axes | None = None,
    title: str | None = None,
    y_lim: float | None = None,
    max_n: int | None = None,
) -> Axes:
    r"""Plot the photon-number probability distribution :math:`P(n)` of a state.

    Parameters
    ----------
    state : jaxquantum.Qarray or array_like
        Ket (1D or column-vector) or density matrix of the bosonic state.
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

    photon_probs = _photon_probs(state)
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


def _wigner_norm(reference: StateLike, x_bound: float, y_bound: float, grid_points: int):
    """Symmetric colour norm set by the peak Wigner value of ``reference``."""
    _, _, wigner = compute_wigner(reference, x_bound, y_bound, grid_points)
    wmax = float(np.abs(wigner).max())
    return mpl_colors.TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)


def plot_wigner_snapshots(
    states: list,
    psi_f: StateLike,
    x_bound: float,
    y_bound: float,
    grid_points: int = 200,
    n_snapshots: int = 5,
    title: str | None = None,
    add_colorbar: bool = True,
    labels: list[str] | None = None,
) -> Axes:
    """Plot a row of Wigner-function snapshots along a state trajectory.

    Samples ``n_snapshots`` states at evenly spaced fractions of the trajectory
    (from the first state at ``t/T* = 0`` to the last at ``t/T* = 1``), computes
    the Wigner quasiprobability distribution for each, and renders them side by
    side as ``pcolormesh``. The color scale is symmetric about zero and normalized
    to the peak Wigner value of the final state ``psi_f``.

    Parameters
    ----------
    states : list
        Ordered sequence of quantum states representing the time evolution
        (e.g. the output of a solver). Snapshot indices are selected by rounding
        evenly spaced fractions up to the nearest index, so the first and last
        entries are always included.
    psi_f : StateLike
        Final (or reference) state used to fix the color-scale bounds. Its peak
        absolute Wigner value sets the symmetric ``vmin``/``vmax`` applied to
        every panel, keeping colors comparable across snapshots.
    n_snapshots : int
        Number of snapshot panels to draw, by default 5.
    x_bound : float
        Half-width of the phase-space grid along the position (q) axis.
    y_bound : float
        Half-width of the phase-space grid along the momentum (p) axis.
    title : str, optional
        Figure-level title (``suptitle``). If ``None``, no title is added.
    grid_points : int, optional
        Number of grid points per axis used when evaluating the Wigner
        function, by default 200. Higher values increase resolution and cost.
    add_colorbar : bool, optional
        If ``True`` (default), append a shared vertical colorbar labeled
        ``W(q, p)`` in an extra axis to the right of the panels.
    labels : list of str, optional
        Panel titles, one per snapshot. Defaults to the time fractions
        ``t/T* = ...``.

    Returns
    -------
    Axes
        The array of Matplotlib axes created by ``plt.subplots``. The first
        ``n_snapshots`` entries hold the Wigner panels (left to right in time);
        the final entry is the colorbar axis.
    """

    fracs = np.linspace(0.0, 1.0, n_snapshots)
    slices = [int(np.ceil(frac * (len(states) - 1))) for frac in fracs]
    gs = {"width_ratios": [1] * len(fracs) + [0.07]}

    if labels is None:
        labels = [rf"$t/T^*$ = {frac:.2f}" for frac in fracs]
    elif len(labels) != len(fracs):
        raise ValueError(f"expected {len(fracs)} labels, got {len(labels)}")

    # 14 in fits five panels; wider rows grow so panels keep their size.
    fig, ax = plt.subplots(
        1,
        len(fracs) + 1,
        figsize=(max(14.0, 2.8 * len(fracs)), 3.75),
        gridspec_kw=gs,
        constrained_layout=True,
    )

    norm = _wigner_norm(psi_f, x_bound, y_bound, grid_points)

    # compute wigner frams
    x, y, frames = wigner_trajectory(states, x_bound, y_bound, grid_points, slices)
    cf = None
    for i in range(len(fracs)):
        cf = ax[i].pcolormesh(
            x,
            y,
            frames[i],
            cmap="RdBu_r",
            norm=norm,
            shading="gouraud",  # smooth interpolation; use "auto" for pixelated
            rasterized=True,  # important if saving to PDF
        )

        # add axes labels
        ax[i].set_title(labels[i])
        ax[i].set_xlabel("q")
        ax[i].set_xticks(np.linspace(-x_bound, x_bound, 5))
        ax[i].set_yticks(np.linspace(-y_bound, y_bound, 5))

        if i == 0:
            ax[i].set_ylabel("p")
        else:
            ax[i].set_yticklabels([])

    if add_colorbar:
        cbar = fig.colorbar(cf, cax=ax[-1])
        cbar.set_label(r"$W(q,p)$")

    if title:
        fig.suptitle(title)

    return ax


def plot_wigner_grid(
    states,
    psi_f: StateLike,
    x_bound: float,
    y_bound: float,
    *,
    labels: list[str] | None = None,
    annotations: list[str] | None = None,
    curve=None,
    curve_label: str = "Fidelity",
    n_cols: int | None = None,
    grid_points: int = 120,
    title: str | None = None,
):
    """Plot a wrapped grid of Wigner functions, one square panel per state.

    The many-panel counterpart of :func:`plot_wigner_snapshots`: panels flow
    left to right and top to bottom, so the figure stays near page proportions
    however long the sequence is. All panels share one colour scale, set by
    the peak Wigner value of ``psi_f``, and one colourbar.

    Parameters
    ----------
    states : sequence
        States to draw, each a ket or density matrix (anything
        :func:`utils.wigner_trajectory` accepts).
    psi_f : StateLike
        Reference state that fixes the shared colour scale.
    x_bound, y_bound : float
        Half-widths of the phase-space window along q and p.
    labels : list of str, optional
        One short label per panel, drawn inside its top-left corner.
    annotations : list of str, optional
        One line per panel, drawn inside its bottom-left corner (e.g. a fidelity).
    curve : sequence of float, optional
        One value per panel. When given, a strip under the grid plots it against
        the panel index, to show how a quantity evolves along the sequence.
    curve_label : str
        Axis label of that strip.
    n_cols : int, optional
        Panels per row. Defaults to the most even split with at most 6 columns.
    grid_points : int
        Samples per phase-space axis.
    title : str, optional
        Figure-level title.

    Returns
    -------
    matplotlib.figure.Figure
        The panels are the first ``len(states)`` entries of ``fig.axes``.
    """
    n = len(states)
    if n == 0:
        raise ValueError("no states to plot")
    for name, items in (("labels", labels), ("annotations", annotations), ("curve", curve)):
        if items is not None and len(items) != n:
            raise ValueError(f"expected {n} {name}, got {len(items)}")
    if n_cols is None:
        n_cols = int(np.ceil(n / np.ceil(n / 6)))  # fewest rows of <= 6, split evenly
    n_cols = max(1, min(int(n_cols), n))
    n_rows = int(np.ceil(n / n_cols))

    norm = _wigner_norm(psi_f, x_bound, y_bound, grid_points)
    _, _, frames = wigner_trajectory(states, x_bound, y_bound, grid_points)

    strip = curve is not None
    fig = plt.figure(
        figsize=(2.2 * n_cols + 1.0, 2.25 * n_rows + (2.0 if strip else 0.0) + 0.4),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        n_rows + strip,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.07],
        height_ratios=[1.0] * n_rows + [0.75] * strip,
    )

    image = None
    for i in range(n):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(grid[row, col])
        image = ax.imshow(
            frames[i],
            extent=(-x_bound, x_bound, -y_bound, y_bound),
            origin="lower",
            cmap="RdBu_r",
            norm=norm,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        ax.set_xticks([-x_bound, 0, x_bound])
        ax.set_yticks([-y_bound, 0, y_bound])
        # Tick labels only where a panel has no neighbour to the left / below.
        if col == 0:
            ax.set_ylabel("p")
        else:
            ax.set_yticklabels([])
        if i + n_cols >= n:
            ax.set_xlabel("q")
        else:
            ax.set_xticklabels([])
        box = {"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5}
        if labels is not None:
            ax.text(0.04, 0.96, labels[i], transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, bbox=box)  # fmt: skip
        if annotations is not None:
            ax.text(0.04, 0.04, annotations[i], transform=ax.transAxes, va="bottom", ha="left",
                    fontsize=9, bbox=box)  # fmt: skip

    cbar = fig.colorbar(image, cax=fig.add_subplot(grid[:n_rows, -1]))
    cbar.set_label(r"$W(q,p)$")

    if strip:
        ax = fig.add_subplot(grid[n_rows, :n_cols])
        ax.plot(np.arange(n), np.asarray(curve, dtype=float), "o-", ms=4, lw=1.2, color="0.2")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylabel(curve_label)
        ax.grid(alpha=0.3)
        every = int(np.ceil(n / 30))  # keep the tick labels legible on long sequences
        ticks = np.arange(0, n, every)
        ax.set_xticks(ticks)
        if labels is not None:
            ax.set_xticklabels([labels[k] for k in ticks], rotation=60, ha="right", fontsize=8)

    if title:
        fig.suptitle(title)
    return fig
