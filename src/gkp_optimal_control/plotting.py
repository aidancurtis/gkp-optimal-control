"""Plotting utilities for bosonic states (pure JAX, no qutip)."""

import jax.numpy as jnp
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def _is_ket(state: jnp.ndarray) -> bool:
    """Return True if ``state`` is a 1D ket or column-vector ket."""
    if state.ndim == 1:
        return True
    if state.ndim == 2 and state.shape[1] == 1:
        return True
    return False


def _photon_probs(state: jnp.ndarray) -> np.ndarray:
    """Return the photon-number probability distribution ``P(n)`` as a numpy array.

    Accepts either a 1D ket ``(n_fock,)``, a column-vector ket ``(n_fock, 1)``,
    or a density matrix ``(n_fock, n_fock)``. For kets, ``P(n) = |c_n|^2``.
    For density matrices, ``P(n) = rho_{nn}``.
    """
    if _is_ket(state):
        ket = jnp.asarray(state).reshape(-1)
        probs = jnp.abs(ket) ** 2
    else:
        probs = jnp.real(jnp.diag(jnp.asarray(state)))
    return np.asarray(probs)


def plot_wigner(
    state: jnp.ndarray | None = None,
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
    state : jnp.ndarray, optional
        Ket (1D or column-vector) or density matrix of the bosonic state.
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

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

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
        cbar = ax.figure.colorbar(cf, ax=ax, cax=cax)
        cbar.set_label(r"$W(\alpha)$")

    if title:
        ax.set_title(title)

    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_aspect("equal")

    return ax


def plot_photon_number(
    state: jnp.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    y_lim: float | None = None,
    max_n: int | None = None,
) -> Axes:
    r"""Plot the photon-number probability distribution :math:`P(n)` of a state.

    Parameters
    ----------
    state : jnp.ndarray
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
