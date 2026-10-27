"""Render held-out changing-geometry examples for the DMF-Gen README.

Example::

    python docs/figures/render_geometry_examples.py

The script reads three held-out elasticity unit cells and three held-out
airfoils through the installed dataset adapters. It does not read checkpoints
or predict fields. Sensor positions are one draw of each evaluation rule (16
stress sensors next to the void, 32 pressure sensors on the fixed ellipse);
they are not the manuscript's saved evaluation manifest.

As in the manuscript's figures, fields are interpolated bilinearly and each
void or airfoil is drawn as a closed smoothing spline of its mask boundary.
Requires ``pip install -e '.[physics]'`` (SciPy and Matplotlib).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from contourpy import contour_generator
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from scipy.interpolate import splev, splprep

from dmf_gen.data import AirfoilDataset, ElasticityDataset, make_observation_batch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOID_COLOR = "#d5dade"  # Elasticity void or airfoil body.
OUTLINE_COLOR = "#202529"
FRAME_COLOR = "#87929b"
# The held-out cases shown in the manuscript's Fig. 1 domain-shape panel.
ELASTICITY_EXAMPLES = (232, 30, 354)
AIRFOIL_EXAMPLES = (207, 196, 104)
# Airfoil view in physical units: chord, sensor ring, and near field.
AIRFOIL_WINDOW = ((-0.35, 1.35), (-0.45, 0.45))
# Outline settings of the manuscript's figures: Gaussian blur of the mask in
# grid cells, then spline smoothing per boundary vertex. The airfoil is only a
# few cells thick, so its staircase is smoothed away; the voids keep a
# near-interpolating fit.
ELASTICITY_OUTLINE = (0.0, 0.05)
AIRFOIL_OUTLINE = (0.6, 1.0)
INTERPRETATION = (
    "Raw dataset fields only; no model reconstruction or quantitative performance is shown."
)


def _source_label(path: Path) -> str:
    """Record a useful input label without publishing a contributor's absolute path."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def _cases(dataset, indices, observed: str, sensors: int, seed: int) -> list[dict[str, object]]:
    """Physical fields, support, candidates, and one evaluation-rule sensor draw."""
    cases = []
    for index in indices:
        sample = dataset[index]
        batch = make_observation_batch([sample], [observed], sensors, seed=seed + index)
        active = batch.obs_indices[0, batch.obs_mask[0]]
        cases.append(
            {
                "index": int(index),
                "case_id": int(sample.case_id),
                "sensor_seed": seed + int(index),
                "values": sample.values.numpy(),
                "support": sample.support.numpy(),
                "candidates": sample.sensor_mask.numpy(),
                "sensors": batch.query_indices[0, active].numpy(),
                "xy": sample.coordinates_raw[:, :2].numpy(),
            }
        )
    return cases


def _outline(support: np.ndarray, blur: float, smoothing: float) -> np.ndarray:
    """Closed smoothing spline along the edge of the support, in grid-index units."""
    excluded = (~support).astype(np.float64)
    if blur:
        excluded = ndimage.gaussian_filter(excluded, blur)
    loops = contour_generator(z=excluded).lines(0.5)
    if len(loops) != 1 or not np.allclose(loops[0][0], loops[0][-1]):
        raise ValueError("the region outside the support must have one closed boundary")
    spline, _ = splprep(loops[0].T, s=smoothing * (len(loops[0]) - 1), per=1)
    return np.column_stack(splev(np.linspace(0.0, 1.0, 400), spline))


def _panel(ax, case, grid, channel: int, outline, cmap: str, norm) -> object:
    support = case["support"].reshape(grid.shape)
    field = case["values"][:, channel].reshape(grid.shape)
    # Points outside the support take the nearest support value, so the
    # interpolated image stays continuous up to the smoothed outline.
    nearest = ndimage.distance_transform_edt(~support, return_distances=False, return_indices=True)
    (x0, x1), (y0, y1) = grid.x_range, grid.y_range
    step = np.array([(x1 - x0) / (grid.shape[1] - 1), (y1 - y0) / (grid.shape[0] - 1)])
    artist = ax.imshow(
        field[tuple(nearest)],
        origin="lower",
        extent=(x0 - step[0] / 2, x1 + step[0] / 2, y0 - step[1] / 2, y1 + step[1] / 2),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )
    boundary = np.array([x0, y0]) + step * _outline(support, *outline)
    ax.add_patch(
        Polygon(
            boundary, facecolor=VOID_COLOR, edgecolor=OUTLINE_COLOR,
            linewidth=0.8, joinstyle="round", zorder=2,
        )
    )
    candidates = case["xy"][case["candidates"]]
    ax.scatter(
        candidates[:, 0], candidates[:, 1], s=1.2, c="white", alpha=0.55, linewidths=0, zorder=3
    )
    sensors = case["xy"][case["sensors"]]
    ax.scatter(
        sensors[:, 0], sensors[:, 1], s=16, facecolors="white",
        edgecolors=OUTLINE_COLOR, linewidths=0.6, zorder=4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(FRAME_COLOR)
    return artist


def _figure(cases, *, grid, channel, outline, window, cmap, label, output: Path) -> None:
    values = np.concatenate([case["values"][case["support"], channel] for case in cases])
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    width = window[0][1] - window[0][0]
    height = window[1][1] - window[1][0]
    size = (11.5, 11.5 / len(cases) * height / width + 0.8)
    fig, axes = plt.subplots(1, len(cases), figsize=size, layout="constrained")
    artist = None
    for ax, case in zip(axes, cases, strict=True):
        artist = _panel(ax, case, grid, channel, outline, cmap, norm)
        ax.set_xlim(*window[0])
        ax.set_ylim(*window[1])
        ax.set_aspect("equal")
        ax.set_title(f"held-out case {case['case_id']}", fontsize=10, pad=5)
    bar = fig.colorbar(
        artist, ax=axes, orientation="horizontal", shrink=0.6, pad=0.04, fraction=0.06
    )
    bar.set_label(label, fontsize=9)
    bar.locator = MaxNLocator(nbins=5)
    bar.update_ticks()
    bar.ax.tick_params(labelsize=8, length=2)
    bar.outline.set_linewidth(0.5)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _manifest_entry(
    dataset, cases, *, source: str, observed: str, rule: str, outline, **details
) -> dict:
    blur, smoothing = outline
    return {
        "source": source,
        "split": "held-out cases of the seeded 80/20 case split, seed 42",
        "heldout_indices": [case["index"] for case in cases],
        "case_ids": [case["case_id"] for case in cases],
        "grid": list(dataset.grid.shape),
        "field_order": list(dataset.field_names),
        "observed_field": observed,
        "sensor_rule": rule,
        **details,
        "illustrative_sensor_count": int(len(cases[0]["sensors"])),
        "illustrative_sensor_seeds": [case["sensor_seed"] for case in cases],
        "sensor_positions_are_paper_manifest": False,
        "displayed_outline": {
            "method": "closed cubic smoothing spline of the support-mask boundary",
            "blur_cells": blur,
            "smoothing_per_vertex": smoothing,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--elasticity", type=Path, default=PROJECT_ROOT / "dataset/elasticity")
    parser.add_argument("--airfoil", type=Path, default=PROJECT_ROOT / "dataset/airfoil")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sensor-seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    elasticity = ElasticityDataset(
        args.elasticity,
        split="test",
        stats_path=_existing(PROJECT_ROOT / "dataset/elasticity.stats.pt"),
    )
    airfoil = AirfoilDataset(
        args.airfoil, split="test", stats_path=_existing(PROJECT_ROOT / "dataset/airfoil.stats.pt")
    )
    elastic_cases = _cases(elasticity, ELASTICITY_EXAMPLES, "sigma", 16, args.sensor_seed)
    airfoil_cases = _cases(airfoil, AIRFOIL_EXAMPLES, "p", 32, args.sensor_seed)

    _figure(
        elastic_cases,
        grid=elasticity.grid,
        channel=0,
        outline=ELASTICITY_OUTLINE,
        window=((0.0, 1.0), (0.0, 1.0)),
        cmap="magma",
        label=r"von Mises stress  $\sigma_\mathrm{vM}$",
        output=args.output_dir / "geometry_elasticity.png",
    )
    _figure(
        airfoil_cases,
        grid=airfoil.grid,
        channel=3,
        outline=AIRFOIL_OUTLINE,
        window=AIRFOIL_WINDOW,
        cmap="viridis",
        label=r"static pressure  $p$  (freestream-normalized)",
        output=args.output_dir / "geometry_airfoil.png",
    )

    manifest_path = args.output_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest["elasticity"] = _manifest_entry(
        elasticity,
        elastic_cases,
        source=f"{_source_label(args.elasticity)}/Interp/"
        "Random_UnitCell_{sigma,mask}_10_interp.npy",
        observed="sigma",
        rule=f"material points within {elasticity.sensor_band_cells} grid steps of the void",
        outline=ELASTICITY_OUTLINE,
    )
    manifest["airfoil"] = _manifest_entry(
        airfoil,
        airfoil_cases,
        source=f"{_source_label(args.airfoil)}/naca_interp_5f/NACA_{{Q,mask,X,Y}}_interp.npy",
        observed="p",
        rule="fixed elliptic ring in model coordinates",
        outline=AIRFOIL_OUTLINE,
        sensor_ring=airfoil.ellipse,
    )
    manifest.pop("interpretation", None)
    manifest["interpretation"] = INTERPRETATION
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "elasticity_cases": manifest["elasticity"]["case_ids"],
        "airfoil_cases": manifest["airfoil"]["case_ids"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
