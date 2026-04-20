import os

import matplotlib.animation as animation
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def animate_wigner(
    wigner_states: list[ArrayLike],
    xvec: np.ndarray,
    yvec: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | None = None,
    interval: int = 40,
    dpi: int = 200,
    fps: int = 25,
    n_contours: int = 100,
    fixed_scale: bool = True,
    add_colorbar: bool = True,
) -> animation.FuncAnimation:
    r"""Animate a sequence of Wigner distributions as a filled contour movie.

    Parameters
    ----------
    wigner_states : list of array_like
        Sequence of 2D Wigner distributions, one per animation frame. Each
        entry must have shape ``(len(yvec), len(xvec))``.
    xvec : array_like
        1D grid of :math:`q`-axis samples.
    yvec : array_like, optional
        1D grid of :math:`p`-axis samples. If ``None``, ``xvec`` is used for
        both axes.
    title : str, optional
        Axes title. If ``None``, no title is set.
    save_path : str, optional
        Output file path. If the extension is ``.gif``, the pillow writer is
        used; otherwise ffmpeg is used. If ``None``, the animation is not
        saved.
    interval : int, default 40
        Delay between frames in milliseconds (affects interactive playback
        only).
    dpi : int, default 200
        Resolution in dots per inch for saved output.
    fps : int, default 25
        Frames per second for saved output.
    n_contours : int, default 100
        Number of contour levels used in each frame.
    fixed_scale : bool, default True
        If ``True``, use a shared symmetric color scale across all frames
        based on the global maximum :math:`|W|`. If ``False``, each frame is
        scaled independently.
    add_colorbar : bool, default True
        If ``True``, attach a colorbar to the figure.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object. Keep a reference to it to prevent garbage
        collection from stopping interactive playback.

    Raises
    ------
    ValueError
        If ``wigner_states`` is empty.
    """
    if len(wigner_states) == 0:
        raise ValueError("wigner_states is empty; nothing to animate.")

    if yvec is None:
        yvec = xvec

    wigner_states = [np.real(w) for w in wigner_states]
    x, y = np.meshgrid(xvec, yvec)

    if fixed_scale:
        wmax = max(float(np.abs(w).max()) for w in wigner_states)
        if wmax == 0.0:
            wmax = 1e-12
        norm = mpl_colors.TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)
        levels = np.linspace(-wmax, wmax, n_contours)
    else:
        norm = None
        levels = n_contours

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_xlim(xvec[0], xvec[-1])
    ax.set_ylim(yvec[0], yvec[-1])
    if title:
        ax.set_title(title)

    cf_holder = [ax.contourf(x, y, wigner_states[0], levels=levels, cmap="RdBu_r", norm=norm)]

    if add_colorbar:
        fig.colorbar(cf_holder[0], ax=ax)

    def update(frame_idx):
        cf_holder[0].remove()
        cf_holder[0] = ax.contourf(
            x, y, wigner_states[frame_idx], levels=levels, cmap="RdBu_r", norm=norm
        )
        return (cf_holder[0],)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(wigner_states),
        interval=interval,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)
        plt.close(fig)

    return anim
