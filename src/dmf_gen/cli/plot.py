"""Make a diagnostic truth/prediction/error plot from a reconstruction NPZ.

Example::

    dmf-plot runs/smoke/reconstruction.npz --output runs/smoke/reconstruction.png

This PNG helps inspect a saved reconstruction; manuscript figures use their
separate documented rendering workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def plot_field_comparisons(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    coords: np.ndarray,
    field_names: tuple[str, ...],
    sample_id: str,
    checkpoint_epoch: int | None,
    output_dir: Path,
    prefix: str,
    sensor_coords: np.ndarray | None = None,
    sensor_field_ids: np.ndarray | None = None,
) -> list[str]:
    """Save one physical-unit truth/reconstruction/error panel per field."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if target.shape != prediction.shape or target.shape != (len(coords), len(field_names)):
        raise ValueError("field arrays, coordinates, and names must align")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("field plotting requires at least two spatial coordinates")
    xy = coords[:, :2]
    # The combustion point cloud is irregular; reuse one mesh across fields.
    # A scatter fallback also handles tiny or degenerate smoke-check subsets.
    try:
        triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1]) if len(xy) >= 3 else None
    except (RuntimeError, ValueError):
        triangulation = None
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for channel, name in enumerate(field_names):
        truth = target[:, channel]
        estimate = prediction[:, channel]
        error = np.abs(estimate - truth)
        field_min = float(min(truth.min(), estimate.min()))
        field_max = float(max(truth.max(), estimate.max()))
        if field_min == field_max:
            field_max = field_min + 1.0e-12
        error_max = max(float(error.max()), 1.0e-12)
        relative_l2 = float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1.0e-12))
        figure, axes = plt.subplots(3, 1, figsize=(12, 9), layout="constrained")
        for row, (values, title) in enumerate(
            ((truth, "Ground truth"), (estimate, "Reconstruction"), (error, "|Error|"))
        ):
            axis = axes[row]
            limits = (0.0, error_max) if row == 2 else (field_min, field_max)
            cmap = "magma" if row == 2 else "coolwarm"
            if triangulation is None:
                artist = axis.scatter(
                    xy[:, 0], xy[:, 1], c=values, s=3, cmap=cmap,
                    vmin=limits[0], vmax=limits[1], linewidths=0,
                )
            else:
                levels = np.linspace(limits[0], limits[1], 65)
                artist = axis.tricontourf(
                    triangulation, values, levels=levels, cmap=cmap,
                    vmin=limits[0], vmax=limits[1],
                )
            if row == 0 and sensor_coords is not None and sensor_field_ids is not None:
                selected = sensor_coords[sensor_field_ids == channel]
                if len(selected):
                    axis.scatter(
                        selected[:, 0], selected[:, 1], s=8, facecolors="none",
                        edgecolors="green", linewidths=0.7,
                    )
            axis.set_title(title)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(artist, ax=axis, pad=0.01, shrink=0.8)
        figure.suptitle(
            f"{name} | {sample_id} | epoch {checkpoint_epoch} | "
            f"relative L2 {relative_l2:.3e}"
        )
        filename = f"{prefix}_field_{name}.png"
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)
        files.append(filename)
    return files


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="NPZ written by dmf-reconstruct")
    parser.add_argument("--output", help="PNG path (default: beside input NPZ)")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def plot_reconstruction(
    input_path: str | Path, output_path: str | Path | None = None, *, dpi: int = 160
) -> Path:
    """Plot physical-unit fields. This is a diagnostic, not a manuscript figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install the plotting extra with pip install -e '.[plot]'") from exc

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_suffix(".png")
    )
    with np.load(source, allow_pickle=False) as archive:
        target = np.asarray(archive["target"], dtype=np.float64)
        prediction = np.asarray(archive["prediction_mean"], dtype=np.float64)
        coords = np.asarray(archive["query_coords_raw"], dtype=np.float64)
        names = tuple(str(name) for name in archive["field_names"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must both be [queries, fields]")
    if target.shape[1] != len(names) or coords.shape[0] != target.shape[0]:
        raise ValueError("field names and coordinates must align with the predictions")
    if coords.shape[1] < 2:
        raise ValueError("plotting requires at least two spatial coordinates")

    # Only a complete rectangular query set can be shown as an image; sampled
    # query subsets and irregular coordinates use a spatial scatter instead.
    logical_shape = metadata.get("logical_shape")
    grid = (
        isinstance(logical_shape, list)
        and len(logical_shape) == 2
        and int(np.prod(logical_shape)) == len(coords)
    )
    figure, axes = plt.subplots(3, len(names), figsize=(3.1 * len(names), 8.1), squeeze=False)
    for column, name in enumerate(names):
        truth = target[:, column]
        estimate = prediction[:, column]
        error = np.abs(estimate - truth)
        common_min = float(min(truth.min(), estimate.min()))
        common_max = float(max(truth.max(), estimate.max()))
        for row, (values, title) in enumerate(
            ((truth, "Reference"), (estimate, "Reconstruction"), (error, "Absolute error"))
        ):
            axis = axes[row, column]
            if grid:
                artist = axis.imshow(
                    values.reshape(logical_shape),
                    origin="lower",
                    cmap="magma" if row == 2 else "viridis",
                    vmin=0.0 if row == 2 else common_min,
                    vmax=common_max if row != 2 else None,
                    interpolation="nearest",
                )
            else:
                artist = axis.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=values,
                    s=max(0.3, min(15.0, 25000.0 / len(values))),
                    cmap="magma" if row == 2 else "viridis",
                    vmin=0.0 if row == 2 else common_min,
                    vmax=common_max if row != 2 else None,
                    linewidths=0,
                )
                axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{name}: {title}")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{metadata['sample_id']} · checkpoint epoch {metadata.get('checkpoint_epoch')}"
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> None:
    args = _arguments()
    print(plot_reconstruction(args.input, args.output, dpi=args.dpi))


if __name__ == "__main__":
    main()
