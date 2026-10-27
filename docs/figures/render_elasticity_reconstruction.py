"""Render elasticity reconstructions in the style of the manuscript's field figures.

Example::

    dmf-reconstruct --config configs/elasticity_gl_rbf_enh.yaml \
        --checkpoint checkpoints/elasticity_gl_rbf_enh.pt --sample-index 28 \
        --output runs/elasticity_demo/case_28.npz
    python docs/figures/render_elasticity_reconstruction.py \
        runs/elasticity_demo/case_28.npz --output runs/elasticity_demo/reconstruction.png

Each input is one full-grid reconstruction written by ``dmf-reconstruct``. For
each case, the top row shows the reference von Mises stress and the DMF-Gen
reconstruction (the mean of the sampled draws); the bottom row shows the stress
sensors on the reference geometry and the signed error, reconstruction minus
reference, with its relative L2 error over the material beneath it. All cases
share one stress scale and one symmetric error scale. The void is drawn with
the spline outline used for the README geometry figure and the manuscript.
Requires ``pip install -e '.[physics]'`` (SciPy and Matplotlib).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Polygon
from render_geometry_examples import ELASTICITY_OUTLINE, _outline
from scipy import ndimage

from dmf_gen.metrics import support_relative_l2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colours and line weights of the manuscript's field figures.
FIELD_CMAP = "Spectral_r"
ERROR_CMAP = "RdBu_r"
INK = "#202020"
VOID_COLOR = "#d8d8d8"
FRAME_WIDTH = 1.0
OUTLINE_WIDTH = 0.9
# Layout in inches: map side, gaps, and the colour-bar column.
CELL = 1.05
COLUMN_GAP = 0.08
CASE_GAP = 0.32
ROW_GAP = 0.30
MARGIN = (0.08, 0.26, 0.26)  # left, top, bottom
BAR_GAP, BAR_WIDTH, BAR_LABELS = 0.12, 0.07, 0.62


def _style() -> None:
    # Arial as in the manuscript, else a metric-compatible substitute; listing
    # only installed families avoids Matplotlib's missing-font warnings.
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in ("Arial", "Liberation Sans", "Helvetica") if name in installed]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [*families, "DejaVu Sans"],
            "font.size": 8,
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _case(path: Path) -> dict[str, object]:
    """Reference and reconstructed stress, support, and sensors on the (ny, nx) grid."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata["query_subset"]:
            raise ValueError(f"{path} holds a query subset; reconstruct without --query-points")
        names = [str(name) for name in data["field_names"]]
        shape = tuple(metadata["logical_shape"])
        order = data["query_indices"]
        target = np.empty_like(data["target"])
        target[order] = data["target"]
        prediction = np.empty_like(data["prediction_mean"])
        prediction[order] = data["prediction_mean"]
        # Sensors sit on query points: match each one's model coordinate.
        sensors = data["obs_coords"][data["obs_mask"]]
        offsets = sensors[:, None, :] - data["query_coords"][None, :, :]
        sensor_points = order[np.argmin((offsets**2).sum(axis=-1), axis=1)]
    stress, mask = names.index("sigma"), names.index("mask")
    support = target[:, mask] > 0.5
    return {
        "sample_id": metadata["sample_id"],
        "checkpoint_epoch": metadata["checkpoint_epoch"],
        "sensor_seed": metadata["sensor_seed"],
        "truth": target[:, stress].reshape(shape),
        "prediction": prediction[:, stress].reshape(shape),
        "support": support.reshape(shape),
        "sensors": np.divmod(sensor_points, shape[1]),
        "relative_l2": support_relative_l2(target[:, stress], prediction[:, stress], support),
    }


def _frame(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_WIDTH)
        spine.set_color(INK)


def _map(ax, field, case, outline, cmap: str, vmin: float, vmax: float) -> object:
    # Points in the void take the nearest material value, so the bilinear image
    # stays continuous up to the outline that covers the void.
    support = case["support"]
    nearest = ndimage.distance_transform_edt(~support, return_distances=False, return_indices=True)
    artist = ax.imshow(
        field[tuple(nearest)], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="bilinear",
    )
    ax.add_patch(
        Polygon(outline, facecolor="white", edgecolor=INK, linewidth=OUTLINE_WIDTH,
                joinstyle="round", zorder=2)
    )
    _frame(ax)
    return artist


def _sensors(ax, case, outline) -> None:
    ny, nx = case["support"].shape
    ax.add_patch(
        Polygon(outline, facecolor=VOID_COLOR, edgecolor=INK, linewidth=OUTLINE_WIDTH,
                joinstyle="round", zorder=2)
    )
    rows, columns = case["sensors"]
    ax.scatter(columns, rows, s=3.0, c=INK, linewidths=0, zorder=3)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{rows.size} sensors", fontsize=8, pad=3)
    _frame(ax)


def _colorbar(fig, box, artist, label: str) -> None:
    bar = fig.colorbar(artist, cax=fig.add_axes(box))
    low, high = artist.get_clim()
    # Only the two ends are ticked: the maps share one scale, so the bar states its range.
    bar.set_ticks([low, high])
    bar.set_ticklabels([f"{low:.0f}", f"{high:.0f}"])
    bar.ax.tick_params(labelsize=8, width=FRAME_WIDTH, length=2, pad=1.5, colors=INK)
    bar.outline.set_linewidth(0.45)
    bar.outline.set_edgecolor(INK)
    bar.set_label(label, fontsize=8, labelpad=2)


def render(cases: list[dict[str, object]], output: Path) -> None:
    truth = np.concatenate([case["truth"][case["support"]] for case in cases])
    vmin, vmax = (float(value) for value in np.percentile(truth, [1, 99]))
    errors = np.concatenate(
        [(case["prediction"] - case["truth"])[case["support"]] for case in cases]
    )
    limit = float(np.percentile(np.abs(errors), 99))

    block = 2 * CELL + COLUMN_GAP
    left, top, bottom = MARGIN
    width = left + len(cases) * block + (len(cases) - 1) * CASE_GAP
    width += BAR_GAP + BAR_WIDTH + BAR_LABELS
    height = top + 2 * CELL + ROW_GAP + bottom
    fig = plt.figure(figsize=(width, height))

    def box(x: float, y: float, w: float = CELL, h: float = CELL) -> list[float]:
        return [x / width, y / height, w / width, h / height]

    upper, lower = bottom + CELL + ROW_GAP, bottom
    field = error = None
    for index, case in enumerate(cases):
        x = left + index * (block + CASE_GAP)
        outline = _outline(case["support"], *ELASTICITY_OUTLINE)
        ax = fig.add_axes(box(x, upper))
        field = _map(ax, case["truth"], case, outline, FIELD_CMAP, vmin, vmax)
        ax.set_title("Truth", fontsize=9, pad=3)
        ax = fig.add_axes(box(x + CELL + COLUMN_GAP, upper))
        _map(ax, case["prediction"], case, outline, FIELD_CMAP, vmin, vmax)
        ax.set_title("DMF-Gen", fontsize=9, pad=3)

        ax = fig.add_axes(box(x, lower))
        _sensors(ax, case, outline)
        if index == 0:
            ax.text(0.5, -0.05, r"rel. $L_2$", transform=ax.transAxes, ha="center", va="top")
        ax = fig.add_axes(box(x + CELL + COLUMN_GAP, lower))
        signed = case["prediction"] - case["truth"]
        error = _map(ax, signed, case, outline, ERROR_CMAP, -limit, limit)
        ax.text(0.5, -0.05, f"{case['relative_l2']:.3f}", transform=ax.transAxes,
                ha="center", va="top")

    bar_x = width - BAR_LABELS - BAR_WIDTH
    _colorbar(fig, box(bar_x, upper, BAR_WIDTH, CELL), field, r"$\sigma_{vM}$ (Pa)")
    _colorbar(fig, box(bar_x, lower, BAR_WIDTH, CELL), error, r"$\Delta\sigma_{vM}$ (Pa)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="NPZ files from dmf-reconstruct")
    parser.add_argument("--output", type=Path, help="Image path (default: beside the first input)")
    args = parser.parse_args()
    output = args.output or args.inputs[0].with_name("elasticity_reconstruction.png")
    _style()
    cases = [_case(path) for path in args.inputs]
    render(cases, output)
    summary = {
        "output": str(output),
        "cases": [
            {
                "sample_id": case["sample_id"],
                "checkpoint_epoch": case["checkpoint_epoch"],
                "sensor_seed": case["sensor_seed"],
                "sigma_relative_l2": round(case["relative_l2"], 4),
            }
            for case in cases
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
