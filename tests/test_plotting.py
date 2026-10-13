"""Layout of the Wigner grid and snapshot figures (drawn off-screen)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import gkp_optimal_control  # noqa: F401, E402  (enables 64-bit JAX)
from gkp_optimal_control.plotting import plot_wigner_grid, plot_wigner_snapshots  # noqa: E402
from gkp_optimal_control.states import fock_state  # noqa: E402

N_FOCK = 12
GRID = 21


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _states(n: int) -> list:
    return [fock_state(N_FOCK, k % 4) for k in range(n)]


def _panels(fig) -> list:
    return [ax for ax in fig.axes if ax.images]


def _rows_cols(fig) -> tuple[int, int]:
    spans = [ax.get_subplotspec() for ax in _panels(fig)]
    return max(s.rowspan.stop for s in spans), max(s.colspan.stop for s in spans)


@pytest.mark.parametrize(
    "n, shape", [(1, (1, 1)), (5, (1, 5)), (6, (1, 6)), (7, (2, 4)), (13, (3, 5)), (26, (5, 6))]
)
def test_grid_wraps_into_balanced_rows(n, shape):
    fig = plot_wigner_grid(_states(n), fock_state(N_FOCK, 1), 3.0, 3.0, grid_points=GRID)
    assert len(_panels(fig)) == n
    assert _rows_cols(fig) == shape
    assert fig.axes[:n] == _panels(fig)  # panels come first, in order
    for ax in _panels(fig):
        assert ax.get_aspect() == 1.0
        assert ax.get_xlim() == (-3.0, 3.0) and ax.get_ylim() == (-3.0, 3.0)
    # Taller, not wider, as the sequence grows.
    width, height = fig.get_size_inches()
    assert width <= 2.2 * 6 + 1.0 and height == pytest.approx(2.25 * shape[0] + 0.4)


def test_column_override():
    fig = plot_wigner_grid(_states(10), fock_state(N_FOCK, 1), 3.0, 3.0, n_cols=3, grid_points=GRID)
    assert _rows_cols(fig) == (4, 3)
    fig = plot_wigner_grid(_states(2), fock_state(N_FOCK, 1), 3.0, 3.0, n_cols=9, grid_points=GRID)
    assert _rows_cols(fig) == (1, 2)  # never wider than the number of panels


def test_panels_share_one_colour_scale_and_colourbar():
    reference = fock_state(N_FOCK, 1)
    fig = plot_wigner_grid(_states(7), reference, 3.0, 3.0, grid_points=GRID)
    norms = {ax.images[0].norm for ax in _panels(fig)}
    assert len(norms) == 1
    norm = norms.pop()
    # |W| of Fock |1> peaks at 1/pi at the origin; the scale is symmetric about zero.
    assert norm.vmax == pytest.approx(1 / np.pi, rel=1e-6) and norm.vmin == -norm.vmax
    colourbars = [ax for ax in fig.axes if ax.get_ylabel() == r"$W(q,p)$"]
    assert len(colourbars) == 1


def test_tick_labels_only_on_outer_panels():
    fig = plot_wigner_grid(_states(13), fock_state(N_FOCK, 1), 3.0, 3.0, grid_points=GRID)  # 3 x 5
    panels = _panels(fig)
    for i, ax in enumerate(panels):
        row, col = divmod(i, 5)
        assert bool(ax.get_ylabel()) == (col == 0)
        assert any(t.get_text() for t in ax.get_yticklabels()) == (col == 0)
        # Bottom of its column: the last row, or above an empty cell in the last row.
        assert bool(ax.get_xlabel()) == (i + 5 >= 13)


def test_labels_annotations_and_curve():
    n = 8
    labels = [f"G {k}" for k in range(n)]
    notes = [f"F = 0.{k}" for k in range(n)]
    curve = np.linspace(0.1, 1.0, n)
    fig = plot_wigner_grid(
        _states(n), fock_state(N_FOCK, 1), 3.0, 3.0, labels=labels, annotations=notes,
        curve=curve, curve_label="Fidelity", title="Run", grid_points=GRID,
    )  # fmt: skip
    for ax, label, note in zip(_panels(fig), labels, notes, strict=True):
        assert [t.get_text() for t in ax.texts] == [label, note]
    strip = [ax for ax in fig.axes if ax.lines]
    assert len(strip) == 1
    np.testing.assert_allclose(strip[0].lines[0].get_ydata(), curve)
    np.testing.assert_array_equal(strip[0].lines[0].get_xdata(), np.arange(n))
    assert strip[0].get_ylabel() == "Fidelity"
    assert [t.get_text() for t in strip[0].get_xticklabels()] == labels
    assert fig.get_suptitle() == "Run"
    # The strip adds height; without it there is no line plot at all.
    plain = plot_wigner_grid(_states(n), fock_state(N_FOCK, 1), 3.0, 3.0, grid_points=GRID)
    assert not [ax for ax in plain.axes if ax.lines]
    assert fig.get_size_inches()[1] > plain.get_size_inches()[1]


def test_long_sequences_thin_the_strip_tick_labels():
    n = 70
    fig = plot_wigner_grid(
        _states(n), fock_state(N_FOCK, 1), 3.0, 3.0, labels=[str(k) for k in range(n)],
        curve=np.zeros(n), grid_points=11,
    )  # fmt: skip
    assert _rows_cols(fig) == (12, 6)
    strip = next(ax for ax in fig.axes if ax.lines)
    assert len(strip.get_xticks()) <= 30


def test_density_matrices_and_kets_mix():
    ket = np.asarray(fock_state(N_FOCK, 2))
    fig = plot_wigner_grid([np.outer(ket, ket.conj())] * 3, ket, 3.0, 3.0, grid_points=GRID)
    as_dm = _panels(fig)[0].images[0].get_array()
    as_ket = (
        _panels(plot_wigner_grid([ket], ket, 3.0, 3.0, grid_points=GRID))[0].images[0].get_array()
    )
    np.testing.assert_allclose(as_dm, as_ket, atol=1e-10)


def test_argument_validation():
    target = fock_state(N_FOCK, 1)
    with pytest.raises(ValueError, match="no states"):
        plot_wigner_grid([], target, 3.0, 3.0)
    for name in ("labels", "annotations", "curve"):
        with pytest.raises(ValueError, match=f"expected 3 {name}"):
            plot_wigner_grid(_states(3), target, 3.0, 3.0, grid_points=GRID, **{name: ["a", "b"]})


def test_snapshot_row_is_unchanged():
    axes = plot_wigner_snapshots(_states(9), fock_state(N_FOCK, 1), 3.0, 3.0, grid_points=GRID)
    assert len(axes) == 6  # five panels and the colourbar axis
    assert [ax.get_title() for ax in axes[:5]] == [
        rf"$t/T^*$ = {f:.2f}" for f in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    labelled = plot_wigner_snapshots(
        _states(3), fock_state(N_FOCK, 1), 3.0, 3.0, grid_points=GRID, n_snapshots=3,
        labels=["a", "b", "c"],
    )  # fmt: skip
    assert [ax.get_title() for ax in labelled[:3]] == ["a", "b", "c"]
