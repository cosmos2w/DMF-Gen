"""Evaluate physical field errors from a selected DMF-Gen checkpoint.

Example::

    dmf-evaluate --config configs/gl_rbf_base.yaml --checkpoint runs/smoke/best.pt \
        --max-samples 1 --query-points 64 --consistency none --device cpu

Full-grid evaluation can use hard observed-value replacement. A query subset
may omit sensors, so this smoke example uses ``--consistency none``.
A changing-geometry dataset names its ``support_field`` and provides
``case_metrics``: physical fields are scored on the reference support, and the
support mask and derived quantities go to ``per_sample_task.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..metrics import clamped_relative_errors, support_relative_l2
from .common import (
    load_config,
    load_model_checkpoint,
    make_batch,
    make_dataset,
    project_root,
    sample_model,
)
from .plot import plot_field_comparisons


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML profile matching the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Saved DMF-Gen or paper checkpoint")
    parser.add_argument("--output-dir", help="Metrics directory; defaults to ignored runs/")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, help="Override evaluation.max_samples")
    parser.add_argument("--query-points", type=int, help="Sample a query subset for a smoke check")
    parser.add_argument("--draws", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--consistency", help="Observation consistency mode; use none for subsets")
    parser.add_argument("--query-chunk-size", type=int)
    parser.add_argument("--seed", type=int, default=42, help="Sensor and source sampling seed")
    parser.add_argument(
        "--plot-samples", type=int, help="Save field panels for this many test samples"
    )
    return parser.parse_args()


def _observed_fields(config: dict[str, Any], field_names: tuple[str, ...]) -> set[str]:
    raw = config["observations"]["fields"]
    result = set()
    for value in raw:
        result.add(field_names[value] if isinstance(value, int) else str(value))
    return result


def _physical_l2(
    predicted: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor | None = None,
) -> list[float]:
    """Per-channel relative L2 over the queried points, or only over ``support``."""
    if support is None:
        delta = predicted - target
        numerator = torch.linalg.vector_norm(delta, dim=0)
        denominator = torch.linalg.vector_norm(target, dim=0).clamp_min(1.0e-12)
        return (numerator / denominator).tolist()
    return [
        support_relative_l2(target[:, channel].numpy(), predicted[:, channel].numpy(), support)
        for channel in range(target.shape[1])
    ]


def _finite_json(value: Any) -> Any:
    """Replace NaN and infinity with ``None`` so the summary stays valid JSON."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summarize_task_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/median of each per-case quantity and clamped errors for true/pred pairs."""
    keys = [key for key in rows[0] if key != "sample_id"]
    summary: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = {
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "cases": int(finite.size),
        }
    paired = {}
    for key in keys:
        stem = key.removesuffix("_true")
        if stem != key and f"{stem}_pred" in keys:
            paired[stem] = clamped_relative_errors(
                [row[key] for row in rows], [row[f"{stem}_pred"] for row in rows]
            )
    return {"per_case": summary, "relative_errors": paired}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Write per-state/per-field physical errors and a compact JSON summary."""
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root() / "runs" / f"evaluation_{checkpoint_path.stem}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # The checkpoint carries the training normalization. Loading the dataset
    # against the configured stats path also verifies its field order.
    dataset = make_dataset(config, "test")
    model, checkpoint = load_model_checkpoint(
        checkpoint_path, config, dataset=dataset, device=args.device
    )
    if args.max_samples is None:
        requested = int(config.get("evaluation", {}).get("max_samples", len(dataset)))
    else:
        requested = args.max_samples
    if requested <= 0:
        raise ValueError("max-samples must be positive")
    count = min(requested, len(dataset))
    evaluation_config = config.get("evaluation", {})
    plot_count = int(evaluation_config.get("plot_samples", 0))
    if args.plot_samples is not None:
        plot_count = args.plot_samples
    if plot_count < 0:
        raise ValueError("plot-samples must be nonnegative")
    plot_count = min(plot_count, count)
    observed = _observed_fields(config, tuple(dataset.field_names))
    field_names = tuple(dataset.field_names)
    support_field = getattr(dataset, "support_field", None)
    # The support mask is scored as a shape in the task metrics, not as a field.
    scored = [channel for channel, name in enumerate(field_names) if name != support_field]
    case_metrics = getattr(dataset, "case_metrics", None)

    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    visualizations: list[dict[str, Any]] = []
    for sample_index in range(count):
        sample = dataset[sample_index]
        query_count = (
            None if args.query_points is None else min(args.query_points, sample.values.shape[0])
        )
        batch = make_batch(
            [sample],
            config,
            query_count=query_count,
            seed=args.seed + sample_index,
            observation_set="evaluation",
        )
        samples = sample_model(
            model,
            batch,
            config,
            device=args.device,
            steps=args.steps,
            draws=args.draws,
            consistency=args.consistency,
            query_chunk_size=args.query_chunk_size,
            seed=args.seed + 100000 + sample_index * 1000,
        ).detach().cpu()
        # Average normalized draws before inversion, as the manuscript does.
        # For affine fields this equals averaging physical draws; log-space
        # fields (airfoil density and pressure) give their geometric mean.
        mean_prediction = batch.to_physical(samples.mean(dim=0))[0]
        target = batch.to_physical()[0]
        support = None
        if support_field is not None:
            support = target[:, field_names.index(support_field)].numpy() > 0.5
        field_errors = _physical_l2(mean_prediction[:, scored], target[:, scored], support)
        spread = batch.to_physical(samples).std(dim=0, unbiased=False)[0]
        if support is not None:
            spread = spread[torch.from_numpy(support)]
        if callable(case_metrics):
            task_rows.append(
                {
                    "sample_id": sample.sample_id,
                    **case_metrics(
                        sample,
                        target.numpy(),
                        mean_prediction.numpy(),
                        batch.query_indices[0].numpy(),
                    ),
                }
            )
        if sample_index < plot_count:
            query_indices = batch.query_indices[0]
            sensor_indices = batch.obs_indices[0] if batch.obs_indices is not None else None
            sensor_mask = batch.obs_mask[0]
            sensor_coords = (
                sample.coordinates_raw[query_indices[sensor_indices[sensor_mask]]].numpy()
                if sensor_indices is not None else None
            )
            solver = evaluation_config.get("solver", "euler")
            steps = args.steps or evaluation_config.get("steps", 2)
            prefix = f"sample_{sample_index:04d}_{solver}_nfe{steps}"
            files = plot_field_comparisons(
                target=target.numpy(),
                prediction=mean_prediction.numpy(),
                coords=sample.coordinates_raw[query_indices].numpy(),
                field_names=tuple(batch.field_names),
                sample_id=sample.sample_id,
                checkpoint_epoch=checkpoint.get("epoch"),
                output_dir=output_dir,
                prefix=prefix,
                sensor_coords=sensor_coords,
                sensor_field_ids=(
                    batch.obs_field_ids[0, sensor_mask].numpy()
                    if sensor_indices is not None else None
                ),
            )
            visualizations.append(
                {"sample_index": sample_index, "sample_id": sample.sample_id, "files": files}
            )
        for error, channel in zip(field_errors, scored, strict=True):
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "resolution": sample.resolution or "native",
                    "field": field_names[channel],
                    "observed": field_names[channel] in observed,
                    "relative_l2": error,
                    "mean_draw_spread": float(spread[:, channel].mean()),
                    "n_queries": int(batch.query_coords.shape[1]),
                }
            )

    table_path = output_dir / "per_sample_field.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scored_names = [field_names[channel] for channel in scored]
    by_field = {
        name: sum(row["relative_l2"] for row in rows if row["field"] == name) / count
        for name in scored_names
    }
    unobserved = [name for name in scored_names if name not in observed]
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": str(config["config_source"]),
        "samples": count,
        "query_subset": args.query_points is not None,
        "observed_fields": sorted(observed),
        "mean_physical_relative_l2_by_field": by_field,
        "mean_unobserved_field_relative_l2": (
            sum(by_field[name] for name in unobserved) / len(unobserved)
            if unobserved
            else None
        ),
        "draws": args.draws or config.get("evaluation", {}).get("draws", 1),
        "steps": args.steps or config.get("evaluation", {}).get("steps", 2),
        "sensor_seed": args.seed,
        "sensor_selection": "deterministic per dataset index; not the paper manifest",
        "visualizations": visualizations,
    }
    if support_field is not None:
        summary["support_field"] = support_field
        summary["field_error_points"] = f"reference support ({support_field} > 0.5)"
    if task_rows:
        task_path = output_dir / "per_sample_task.csv"
        with task_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
            writer.writeheader()
            writer.writerows(task_rows)
        summary["task_metrics"] = _summarize_task_metrics(task_rows)
    summary = _finite_json(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    close = getattr(dataset, "close", None)
    if close is not None:
        close()
    return summary


def main() -> None:
    summary = evaluate(_arguments())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
