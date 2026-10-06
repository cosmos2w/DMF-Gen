# ruff: noqa: E501 -- demonstration prose stays on one line.
"""Render two dataset-only examples for the DMF-Gen README.

Example::

    python docs/figures/render_dataset_examples.py

The script reads one held-out combustion frame and one held-out CFD case/time from the local dataset links. It does not read checkpoints, predict fields, or copy dataset files. The 256 temperature locations illustrate the Cond-T sensor budget; they are not the manuscript's saved evaluation manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("CH4", "CO", "T", "U_1", "p")
LEVELS = {"L": 32, "M": 64, "H": 128}


def _source_label(path: Path) -> str:
    """Record a useful input label without publishing a contributor's absolute path."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def _field_names(raw: object) -> tuple[str, ...]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("processed HDF5 requires selected_fields metadata")
    return tuple(part.strip() for part in raw.split(","))


def _on_grid(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place stored point values on their exact x/y Cartesian product without interpolation."""
    x, ix = np.unique(coords[:, 0], return_inverse=True)
    y, iy = np.unique(coords[:, 1], return_inverse=True)
    slots = iy * len(x) + ix
    if len(x) * len(y) != len(coords) or len(np.unique(slots)) != len(coords):
        raise ValueError("coordinates do not form one complete rectilinear grid")
    grid = np.empty((len(y), len(x), values.shape[1]), dtype=np.float64)
    grid[iy, ix] = values
    if not np.isfinite(grid).all():
        raise ValueError("source field contains non-finite values")
    return x, y, grid


def _combustion_state(path: Path, split_seed: int) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        fields = handle["fields"]
        if _field_names(fields.attrs.get("selected_fields")) != FIELDS:
            raise ValueError("combustion field order must be CH4,CO,T,U_1,p")
        if fields.shape != (1, 10000, 40300, 1, 1, 5):
            raise ValueError("expected the paper's five-field, 40300-point combustion file")
        # Mirror the release loader's seeded 90/10 frame split before choosing test state zero.
        permutation = np.random.default_rng(split_seed).permutation(fields.shape[1])
        heldout = np.sort(permutation[int(np.floor(0.9 * fields.shape[1])) :])
        frame_id = int(heldout[0])
        coords = np.asarray(handle["coordinates"], dtype=np.float64).reshape(-1, 3)
        values = np.asarray(fields[0, frame_id, :, 0, 0, :], dtype=np.float64)
        time = float(handle["time"][frame_id]) if "time" in handle else None
    x, y, grid = _on_grid(coords, values)
    if grid.shape != (100, 403, 5):
        raise ValueError(f"unexpected combustion logical grid {grid.shape}")
    return {
        "x": x,
        "y": y,
        "grid": grid,
        "coords": coords,
        "values": values,
        "frame_id": frame_id,
        "time": time,
        "heldout_count": len(heldout),
    }


def _cfd_state(root: Path, case_id: int, frame_id: int) -> dict[str, object]:
    fields_by_level: dict[str, np.ndarray] = {}
    times: dict[str, float] = {}
    for level, side in LEVELS.items():
        with h5py.File(root / f"CFD_{level}_res.h5", "r") as handle:
            fields = handle["fields"]
            if _field_names(fields.attrs.get("selected_fields")) != (
                "Vx",
                "Vy",
                "density",
                "pressure",
            ):
                raise ValueError(f"{level} processed field zero is not verified Vx")
            if fields.shape != (10000, 21, side * side, 1, 1, 4):
                raise ValueError(f"unexpected {level} CFD dimensions {fields.shape}")
            if not 9000 <= case_id < fields.shape[0] or not 10 <= frame_id <= 20:
                raise ValueError("choose a held-out case and a frame in the common 10–20 window")
            coords = np.asarray(handle["coordinates"], dtype=np.float64).reshape(-1, 3)
            values = np.asarray(fields[case_id, frame_id, :, 0, 0, 0], dtype=np.float64)
            x, y, grid = _on_grid(coords, values[:, None])
            if grid.shape != (side, side, 1):
                raise ValueError(f"unexpected {level} CFD logical grid {grid.shape}")
            fields_by_level[level] = grid[..., 0]
            times[level] = float(handle["time"][frame_id])
            if not np.allclose(x, y, atol=1.0e-7):
                raise ValueError(f"{level} CFD x/y coordinate grids differ")
    if len({round(value, 7) for value in times.values()}) != 1:
        raise ValueError("CFD frames do not share the same physical time")
    # Processing defines L and M as nonoverlapping spatial averages of H.
    coarse_max_errors: dict[str, float] = {}
    high = fields_by_level["H"]
    for level, factor in (("M", 2), ("L", 4)):
        averaged = high.reshape(128 // factor, factor, 128 // factor, factor).mean(axis=(1, 3))
        error = float(np.max(np.abs(fields_by_level[level] - averaged)))
        if error > 1.0e-6:
            raise ValueError(f"{level} is not an average of the selected H state: {error}")
        coarse_max_errors[level] = error
    return {
        "fields": fields_by_level,
        "time": times["H"],
        "case_id": case_id,
        "frame_id": frame_id,
        "coarse_max_errors": coarse_max_errors,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _colorbar(fig: plt.Figure, artist: object, ax: plt.Axes) -> None:
    bar = fig.colorbar(
        artist, ax=ax, orientation="horizontal", pad=0.035, fraction=0.07, shrink=0.87
    )
    bar.locator = MaxNLocator(nbins=3)
    bar.update_ticks()
    bar.ax.tick_params(labelsize=8, length=2)
    bar.outline.set_linewidth(0.5)


def _combustion_figure(state: dict[str, object], sensor_seed: int, output: Path) -> np.ndarray:
    x = state["x"]
    y = state["y"]
    grid = state["grid"]
    coords = state["coords"]
    values = state["values"]
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert isinstance(grid, np.ndarray) and isinstance(coords, np.ndarray)
    assert isinstance(values, np.ndarray)
    # These locations explain the Cond-T observation pattern; no saved paper sensor manifest is implied.
    selected = np.sort(
        np.random.default_rng(sensor_seed).choice(len(coords), size=256, replace=False)
    )
    temperature_norm = Normalize(vmin=float(grid[..., 2].min()), vmax=float(grid[..., 2].max()))
    specifications = (
        (0, "Methane · CH₄", "viridis", None),
        (1, "Carbon monoxide · CO", "viridis", None),
        (2, "Temperature · T (K)", "inferno", temperature_norm),
        (None, "Temperature observations · 256", "inferno", temperature_norm),
        (3, "Streamwise velocity · U₁ (m/s)", "RdBu_r", None),
        (4, "Pressure · p (kPa)", "viridis", None),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 7.7), layout="constrained")
    for ax, (channel, title, cmap, norm) in zip(axes.flat, specifications, strict=True):
        if channel is None:
            ax.set_facecolor("#f3f5f6")
            artist = ax.scatter(
                coords[selected, 0],
                coords[selected, 1],
                c=values[selected, 2],
                cmap=cmap,
                norm=norm,
                s=14,
                linewidths=0.25,
                edgecolors="#202529",
                zorder=3,
            )
        else:
            shown = grid[..., channel] / (1000.0 if channel == 4 else 1.0)
            if channel == 3:
                peak = float(np.max(np.abs(shown)))
                norm = TwoSlopeNorm(vmin=-peak, vcenter=0.0, vmax=peak)
            artist = ax.pcolormesh(
                x, y, shown, shading="nearest", cmap=cmap, norm=norm, rasterized=True
            )
        ax.set_title(title, loc="left", fontsize=10, pad=5)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(float(y[0]), float(y[-1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("#87929b")
        _colorbar(fig, artist, ax)
    fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return selected


def _cfd_figure(state: dict[str, object], output: Path) -> None:
    fields = state["fields"]
    assert isinstance(fields, dict)
    peak = max(float(np.max(np.abs(field))) for field in fields.values())
    norm = TwoSlopeNorm(vmin=-peak, vcenter=0.0, vmax=peak)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), layout="constrained")
    artist = None
    for ax, (level, side) in zip(axes, LEVELS.items(), strict=True):
        artist = ax.imshow(
            fields[level],
            origin="lower",
            extent=(-1, 1, -1, 1),
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
        )
        ax.set_title(f"{level}  ·  {side} × {side}", fontsize=11, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#87929b")
    assert artist is not None
    bar = fig.colorbar(
        artist, ax=axes, orientation="horizontal", shrink=0.66, pad=0.08, fraction=0.05
    )
    bar.set_label("Streamwise velocity  Vₓ", fontsize=9)
    bar.locator = MaxNLocator(nbins=5)
    bar.update_ticks()
    bar.ax.tick_params(labelsize=8, length=2)
    bar.outline.set_linewidth(0.5)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--combustion", type=Path, default=PROJECT_ROOT / "dataset/combustion.h5")
    parser.add_argument(
        "--cfd-root", type=Path, default=PROJECT_ROOT / "dataset/pdebench/Processed"
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cfd-case", type=int, default=9000)
    parser.add_argument("--cfd-frame", type=int, default=10)
    parser.add_argument("--sensor-seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    combustion = _combustion_state(args.combustion, split_seed=42)
    cfd = _cfd_state(args.cfd_root, case_id=args.cfd_case, frame_id=args.cfd_frame)
    sensor_indices = _combustion_figure(
        combustion, args.sensor_seed, args.output_dir / "combustion_multifield.png"
    )
    _cfd_figure(cfd, args.output_dir / "cfd_resolution_hierarchy.png")
    manifest = {
        "combustion": {
            "source": _source_label(args.combustion),
            "split": "held-out frame split, seed 42",
            "heldout_state_index": 0,
            "frame_id": combustion["frame_id"],
            "frame_time_as_stored": combustion["time"],
            "grid": [100, 403],
            "field_order": list(FIELDS),
            "conditioning_example": "T only",
            "illustrative_sensor_count": len(sensor_indices),
            "illustrative_sensor_seed": args.sensor_seed,
            "sensor_positions_are_paper_manifest": False,
        },
        "cfd": {
            "source": f"{_source_label(args.cfd_root)}/CFD_{{L,M,H}}_res.h5",
            "split": "held-out cases 9000-9999",
            "case_id": cfd["case_id"],
            "frame_id": cfd["frame_id"],
            "frame_time_as_stored": cfd["time"],
            "field": "Vx",
            "grid_sizes": {level: [side, side] for level, side in LEVELS.items()},
            "max_absolute_coarse_average_difference": cfd["coarse_max_errors"],
        },
        "interpretation": "Raw dataset fields only; no model reconstruction or quantitative performance is shown.",
    }
    # Preserve the entries written by render_geometry_examples.py.
    manifest_path = args.output_dir / "source_manifest.json"
    merged = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    merged.pop("interpretation", None)
    merged.update(manifest)
    manifest_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "combustion_frame": combustion["frame_id"],
                "cfd_case": cfd["case_id"],
                "cfd_frame": cfd["frame_id"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
