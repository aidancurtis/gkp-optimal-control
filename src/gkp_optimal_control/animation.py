"""Animate a sequence of Wigner distributions.

Pure JAX/numpy/matplotlib -- no qutip. Inputs may come as a 3D array from
:func:`utils.wigner_trajectory` (the fast path) or as a Python sequence of
2D frames; both are normalized to a contiguous float32 numpy stack before
rendering so per-frame conversions don't dominate animation cost.
"""

import os

import matplotlib.animation as animation
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _to_numpy_frames(wigner_states) -> np.ndarray:
    """Normalize input to a contiguous float32 numpy array of shape ``(T, H, W)``.

    Accepts a 3D array (numpy or JAX) of shape ``(T, H, W)``, or a sequence
    of 2D arrays from either library. Complex input has its real part taken.
    The returned array is C-contiguous and float32 -- matplotlib's
    ``set_array`` copies anyway, so we pay the conversion cost once here.
    """
    if hasattr(wigner_states, "ndim") and wigner_states.ndim == 3:
        arr = np.asarray(wigner_states)
    else:
        if len(wigner_states) == 0:
            raise ValueError("wigner_states is empty; nothing to animate.")
        arr = np.stack([np.asarray(w) for w in wigner_states], axis=0)

    arr = np.real(arr)

    return np.ascontiguousarray(arr, dtype=np.float32)


def animate_wigner(
    wigner_states,
    xvec,
    yvec=None,
    title: str | None = None,
    save_path: str | None = None,
    interval: int = 40,
    dpi: int = 150,
    fps: int = 25,
    fixed_scale: bool = True,
    add_colorbar: bool = True,
    *,
    use_contour: bool = False,
    n_contours: int = 60,
) -> animation.FuncAnimation:
    r"""Animate a sequence of Wigner distributions.

    Parameters
    ----------
    wigner_states : numpy.ndarray, jnp.ndarray, or sequence of 2D arrays
        Frames to animate. Most commonly the 3D output of
        :func:`utils.wigner_trajectory`, shape ``(T, len(yvec), len(xvec))``.
    xvec, yvec : array_like
        1D grid arrays. If ``yvec`` is ``None``, ``xvec`` is used for both
        axes.
    title : str, optional
        Plot title.
    save_path : str, optional
        Output filename. ``.gif`` uses Pillow; other extensions use ffmpeg
        (libx264, yuv420p, preset=fast, CRF=18).
    interval : int, default 40
        Delay between frames in milliseconds (display only; ``fps`` controls
        the saved file).
    dpi : int, default 150
        Output resolution.
    fps : int, default 25
        Frames per second for saved animations.
    fixed_scale : bool, default True
        If ``True``, use a global symmetric colour scale derived from the
        whole trajectory. If ``False``, rescale per frame (slower; useful
        for trajectories where Wigner magnitude changes over orders of
        magnitude).
    add_colorbar : bool, default True
        Attach a colorbar to the figure.
    use_contour : bool, default False
        Use ``contourf`` rather than ``pcolormesh``. Contour mode produces
        sharper level lines but is much slower because each frame redraws
        the contour set; ``pcolormesh`` mode supports blitting.
    n_contours : int, default 60
        Number of contour levels when ``use_contour`` is True.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation. If ``save_path`` was given, the figure has been
        closed and the file written; the returned animation is still valid
        if you want to display it inline.
    """
    frames = _to_numpy_frames(wigner_states)

    xvec = np.asarray(xvec)
    yvec = np.asarray(yvec) if yvec is not None else xvec

    # Global color scale.
    if fixed_scale:
        wmax = float(np.abs(frames).max())
        if wmax == 0.0:
            wmax = 1e-12
        norm = mpl_colors.TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_xlim(xvec[0], xvec[-1])
    ax.set_ylim(yvec[0], yvec[-1])
    if title:
        ax.set_title(title)

    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

    x, y = np.meshgrid(xvec, yvec)

    if use_contour:
        if fixed_scale:
            levels = np.linspace(-wmax, wmax, n_contours)
        else:
            levels = n_contours

        cf_holder = [ax.contourf(x, y, frames[0], levels=levels, cmap="RdBu_r", norm=norm)]

        if add_colorbar:
            cbar = fig.colorbar(cf_holder[0], ax=ax, cax=cax)
            cbar.set_label(r"$W(\alpha)$")

        def update_contour(frame_idx):
            cf_holder[0].remove()
            cf_holder[0] = ax.contourf(
                x, y, frames[frame_idx], levels=levels, cmap="RdBu_r", norm=norm
            )
            return (cf_holder[0],)

        update = update_contour
        blit = False
    else:
        mesh = ax.pcolormesh(
            x,
            y,
            frames[0],
            cmap="RdBu_r",
            norm=norm,
            shading="gouraud",
            rasterized=True,
        )

        if add_colorbar:
            cbar = fig.colorbar(mesh, ax=ax, cax=cax)
            cbar.set_label(r"$W(\alpha)$")

        def update_mesh(frame_idx):
            mesh.set_array(frames[frame_idx].ravel())
            if not fixed_scale:
                local_max = float(np.abs(frames[frame_idx]).max()) or 1e-12
                mesh.set_norm(mpl_colors.TwoSlopeNorm(vmin=-local_max, vcenter=0.0, vmax=local_max))
            return (mesh,)

        update = update_mesh
        blit = True

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=blit,
        repeat=False,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=-1,
                extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18"],
            )
            anim.save(save_path, writer=writer, dpi=dpi)
        plt.close(fig)

    return anim
