"""Reconstruct one state and save aligned physical arrays for diagnostics.

Example::

    dmf-reconstruct --config configs/gl_rbf_base.yaml \
        --checkpoint runs/smoke/best.pt --sample-index 0 --query-points 64 \
        --consistency none --output runs/smoke/reconstruction.npz

The NPZ records query indices, physical fields, marked sensors, and provenance.
Query subsets are for execution checks, not paper figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    load_config,
    load_model_checkpoint,
    make_batch,
    make_dataset,
    project_root,
    sample_model,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", help="Compressed NPZ path; defaults to ignored runs/")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-points", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--consistency", help="Use none when sensors are outside query subset")
    parser.add_argument("--query-chunk-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reconstruct(args: argparse.Namespace) -> dict[str, Any]:
    """Produce one query-aligned NPZ in physical units from a selected checkpoint."""
    config = load_config(args.config)
    dataset = make_dataset(config, args.split)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset) - 1}]")
    sample = dataset[args.sample_index]
    model, checkpoint = load_model_checkpoint(
        args.checkpoint, config, dataset=dataset, device=args.device
    )
    query_count = (
        None if args.query_points is None else min(args.query_points, sample.values.shape[0])
    )
    batch = make_batch(
        [sample],
        config,
        query_count=query_count,
        seed=args.seed,
        observation_set="evaluation",
    )
    draws = sample_model(
        model,
        batch,
        config,
        device=args.device,
        steps=args.steps,
        draws=args.draws,
        consistency=args.consistency,
        query_chunk_size=args.query_chunk_size,
        seed=args.seed + 100000,
    ).detach().cpu()
    # Save physical units for plotting and raw query coordinates separately
    # from the normalized coordinates passed to the model.
    physical_draws = batch.to_physical(draws)[:, 0]
    physical_target = batch.to_physical()[0]
    mean_prediction = physical_draws.mean(dim=0)
    query_indices = batch.query_indices[0]

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else project_root()
        / "runs"
        / f"reconstruction_{checkpoint_path.stem}_{args.sample_index}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_id": sample.sample_id,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": str(config["config_source"]),
        "resolution": sample.resolution,
        "logical_shape": sample.logical_shape,
        "n_queries": int(query_indices.numel()),
        "query_subset": args.query_points is not None,
        "sensor_seed": args.seed,
        "sensor_selection": "deterministic; not the paper manifest",
    }
    np.savez_compressed(
        output,
        query_coords=batch.query_coords[0].numpy(),
        query_coords_raw=sample.coordinates_raw[query_indices].numpy(),
        query_indices=query_indices.numpy(),
        target=physical_target.numpy(),
        prediction_mean=mean_prediction.numpy(),
        prediction_draws=physical_draws.numpy(),
        obs_coords=batch.obs_coords[0].numpy(),
        obs_values=batch.observations_to_physical()[0].numpy(),
        obs_field_ids=batch.obs_field_ids[0].numpy(),
        obs_mask=batch.obs_mask[0].numpy(),
        field_names=np.asarray(batch.field_names),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    close = getattr(dataset, "close", None)
    if close is not None:
        close()
    return {"output": str(output), **metadata}


def main() -> None:
    result = reconstruct(_arguments())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
