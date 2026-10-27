"""Train a GL-RBF-family model from an explicit YAML task profile.

Example::

    dmf-train --config configs/gl_rbf_base.yaml --epochs 1 --batch-size 1 \
        --query-points 64 --max-steps 1 --validation-samples 1 --device cpu

The resolved config, metrics, and checkpoints go under ignored ``runs/``.
This short example checks execution, not manuscript accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..data import FieldSample, group_samples_by_resolution
from ..models import build_pointcloud_model
from .common import (
    batch_to_device,
    load_config,
    make_batch,
    make_dataset,
    project_root,
    save_model_checkpoint,
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmf-train",
        description="Train GL-RBF-family DMF-Gen models from configs/*.yaml.",
    )
    parser.add_argument("--config", required=True, help="YAML config path, relative to repo root")
    parser.add_argument("--epochs", type=_positive_int, help="Override configured epoch count")
    parser.add_argument("--batch-size", type=_positive_int, help="Override configured batch size")
    parser.add_argument(
        "--query-points", type=_positive_int, help="Override query points per sample"
    )
    parser.add_argument(
        "--max-steps", type=_positive_int, help="Stop after this many optimizer updates"
    )
    parser.add_argument(
        "--validation-samples",
        type=_positive_int,
        help="Override the number of heldout samples evaluated each epoch",
    )
    parser.add_argument("--device", help="Torch device (default: CUDA when available, else CPU)")
    parser.add_argument(
        "--output-dir",
        help="Run directory under runs/ (default: runs/<config>-<timestamp>)",
    )
    return parser


def _select_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_directory(
    requested: str | Path | None,
    config: Mapping[str, Any],
    root: Path,
) -> Path:
    """Keep generated training output inside the independent checkout's runs tree."""
    runs_root = (root / "runs").resolve()
    if requested is None:
        training = config.get("training", {})
        configured = training.get("output_dir") if isinstance(training, Mapping) else None
        if configured:
            candidate = Path(configured).expanduser()
            path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (runs_root / candidate).resolve()
            )
        else:
            source = Path(str(config.get("config_source", "config.yaml")))
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = (runs_root / f"{source.stem}-{stamp}").resolve()
    else:
        candidate = Path(requested).expanduser()
        if candidate.is_absolute():
            path = candidate.resolve()
        elif candidate.parts and candidate.parts[0] == "runs":
            path = (root / candidate).resolve()
        else:
            path = (runs_root / candidate).resolve()

    if path != runs_root and runs_root not in path.parents:
        raise ValueError(f"run output must be inside {runs_root}, got {path}")
    return path


def _configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("dmf_gen.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class _ExponentialMovingAverage:
    """Exponential moving average of trainable parameters; buffers are not averaged."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("training.ema_decay must be between 0 and 1")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)

    @contextmanager
    def applied(self, model: torch.nn.Module) -> Iterator[None]:
        """Temporarily load the averaged parameters into ``model``."""
        backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])

    def averaged_state(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """A full state dictionary with averaged parameters and current buffers."""
        with self.applied(model):
            return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _resolution_batches(
    dataset: Any,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[str | None, np.ndarray]]:
    """Shuffle batches while keeping every mixed-resolution batch homogeneous."""
    by_resolution = getattr(dataset, "indices_by_resolution", None)
    if isinstance(by_resolution, Mapping):
        rank = {"L": 0, "M": 1, "H": 2}
        buckets = sorted(
            by_resolution.items(),
            key=lambda item: (rank.get(str(item[0]), 3), str(item[0])),
        )
    else:
        buckets = [(None, np.arange(len(dataset), dtype=np.int64))]

    batches: list[tuple[str | None, np.ndarray]] = []
    for resolution, values in buckets:
        indices = np.asarray(values, dtype=np.int64)
        if indices.size == 0:
            continue
        shuffled = rng.permutation(indices)
        for start in range(0, shuffled.size, batch_size):
            selected = shuffled[start : start + batch_size]
            if selected.size:
                batches.append((None if resolution is None else str(resolution), selected))
    rng.shuffle(batches)
    return batches


def _validation_batches(
    dataset: Any,
    n_samples: int,
    batch_size: int,
    seed: int,
) -> list[list[FieldSample]]:
    """Select a deterministic held-out subset for this epoch and group by grid size."""
    if len(dataset) == 0:
        raise ValueError("validation split is empty")
    count = min(int(n_samples), len(dataset))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset), size=count, replace=False))
    samples = [dataset[int(index)] for index in selected]
    grouped = group_samples_by_resolution(samples)
    result: list[list[FieldSample]] = []
    for resolution in sorted(grouped, key=lambda value: (str(value) != "L", str(value))):
        group = grouped[resolution]
        result.extend(
            group[start : start + batch_size] for start in range(0, len(group), batch_size)
        )
    return result


def _save_config(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")


def _write_loss_artifacts(run_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Keep the numeric history and its diagnostic PNG together in the run directory."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("loss plotting requires pip install -e '.[plot]'") from exc

    columns = (
        "epoch",
        "global_step",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "epoch_seconds",
    )
    with (run_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in records)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["epoch"] for row in records], [row["train_loss"] for row in records], label="Train"
    )
    validated = [row for row in records if row["validation_loss"] is not None]
    if validated:
        axis.plot(
            [row["epoch"] for row in validated],
            [row["validation_loss"] for row in validated],
            label="Validation",
        )
    axis.set(xlabel="Epoch", ylabel="Flow matching loss", title="Training history")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "loss_history.png", dpi=160)
    plt.close(figure)


def _run_live_evaluation(
    *,
    config_path: str,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    epoch: int,
    steps: int | None = None,
) -> None:
    """Pause training at a saved checkpoint for full physical-field evaluation."""
    command = [
        sys.executable,
        "-m",
        "dmf_gen.cli.evaluate",
        "--config",
        config_path,
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--device",
        str(device),
    ]
    if steps is not None:
        command.extend(("--steps", str(steps)))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluate.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(
            f"physical evaluation at epoch {epoch} failed with exit code {result.returncode}; "
            f"see {output_dir / 'evaluate.log'}"
        )


def _evaluate_epoch(
    model: torch.nn.Module,
    dataset: Any,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    query_points: int,
    validation_samples: int,
    batch_size: int,
    seed: int,
    epoch: int,
    observation_set: str = "evaluation",
) -> float:
    """Measure the same stochastic flow loss with the evaluation sensor budget."""
    batches = _validation_batches(dataset, validation_samples, batch_size, seed + epoch)
    prior_training = model.training
    model.eval()
    weighted_loss = 0.0
    seen = 0
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.no_grad():
            for batch_id, samples in enumerate(batches):
                point_count = samples[0].values.shape[0]
                obs = make_batch(
                    samples,
                    config,
                    query_count=min(query_points, point_count),
                    seed=seed + epoch * 1_000_003 + batch_id,
                    observation_set=observation_set,
                )
                tensors = batch_to_device(obs, device)
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(seed + epoch * 1_000_033 + batch_id)
                    loss, _ = model.training_loss(
                        x1=tensors["target_fields"],
                        coords=tensors["query_coords"],
                        obs_coords=tensors["obs_coords"],
                        obs_values=tensors["obs_values"],
                        obs_mask=tensors["obs_mask"],
                        obs_field_ids=tensors["obs_field_ids"],
                        obs_indices=tensors.get("obs_indices"),
                    )
                batch_count = len(samples)
                weighted_loss += float(loss.detach().cpu()) * batch_count
                seen += batch_count
    finally:
        model.train(prior_training)
    return weighted_loss / max(seen, 1)


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a profile, train by native resolution, and save reviewable artifacts."""
    args = _parser().parse_args(argv)
    root = project_root().resolve()
    # An absolute config from an installed wheel should still resolve its data
    # paths and output directory against the checkout that contains that config.
    supplied_config = Path(args.config).expanduser().resolve()
    config_checkout = supplied_config.parent.parent
    if supplied_config.parent.name == "configs" and (config_checkout / "pyproject.toml").is_file():
        root = config_checkout
    config = load_config(args.config, repo_root=root)
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("config must contain a 'training' mapping")
    effective_training = dict(training)
    for arg_name, config_name in (
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("query_points", "query_points"),
        ("validation_samples", "validation_samples"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            effective_training[config_name] = value
    epochs = int(effective_training.get("epochs", 1))
    batch_size = int(effective_training.get("batch_size", 1))
    query_points = int(effective_training.get("query_points", 1))
    validation_samples = int(effective_training.get("validation_samples", 1))
    validation_every_epochs = int(effective_training.get("validation_every_epochs", 1))
    save_every_epochs = int(effective_training.get("save_every_epochs", 1))
    plot_every_epochs = int(effective_training.get("plot_every_epochs", 0))
    scheduler_step_unit = str(effective_training.get("scheduler_step_unit", "optimizer_step"))
    validation_observation_set = str(
        effective_training.get("validation_observation_set", "evaluation")
    )
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise ValueError("config.evaluation must be a mapping when provided")
    live_evaluation_every_epochs = int(evaluation.get("every_epochs", 0))
    benchmark_steps = tuple(int(value) for value in evaluation.get("benchmark_steps", ()))
    if len(set(benchmark_steps)) != len(benchmark_steps) or any(
        value <= 0 for value in benchmark_steps
    ):
        raise ValueError("evaluation.benchmark_steps must contain unique positive integers")
    if (
        min(
            epochs,
            batch_size,
            query_points,
            validation_samples,
            validation_every_epochs,
            save_every_epochs,
        )
        <= 0
    ):
        raise ValueError(
            "epochs, batch_size, query_points, validation_samples, validation_every_epochs, "
            "and save_every_epochs must be positive"
        )
    if plot_every_epochs < 0 or live_evaluation_every_epochs < 0:
        raise ValueError("plot_every_epochs and evaluation.every_epochs must be nonnegative")
    if scheduler_step_unit not in {"optimizer_step", "epoch"}:
        raise ValueError("scheduler_step_unit must be 'optimizer_step' or 'epoch'")
    if validation_observation_set not in {"training", "evaluation"}:
        raise ValueError("validation_observation_set must be 'training' or 'evaluation'")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    config["training"] = effective_training

    device = _select_device(args.device)
    run_dir = _run_directory(args.output_dir, config, root)
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("config.runtime must be a mapping when provided")
    config["runtime"] = {
        **dict(runtime),
        "device": str(device),
        "max_steps": args.max_steps,
        "output_dir": str(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(run_dir)
    _save_config(run_dir / "run_config.yaml", config)

    seed = int(effective_training.get("seed", 42))
    _seed_everything(seed, device)
    # Fit/load statistics once on training data, then reuse exactly those
    # channel statistics for held-out samples and checkpoint metadata.
    train_dataset = make_dataset(config, "train")
    stats = train_dataset.stats
    validation_dataset = make_dataset(config, "test", stats=stats)
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    model_section = config.get("model")
    if not isinstance(model_section, Mapping):
        raise ValueError("config must contain a 'model' mapping with an explicit backbone")
    if model_section.get("model_name") is None and model_section.get("backbone") is None:
        raise ValueError("release training configs must select model.model_name or model.backbone")
    model = build_pointcloud_model(
        model_section,
        n_fields=len(train_dataset.field_names),
        device=device,
    ).to(device)

    learning_rate = float(effective_training.get("learning_rate", 1.0e-4))
    weight_decay = float(effective_training.get("weight_decay", 1.0e-6))
    gradient_clip = float(effective_training.get("gradient_clip", 1.0))
    if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
        raise ValueError(
            "learning_rate and gradient_clip must be positive; weight_decay nonnegative"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Optional parameter averaging: validation and saved ``model`` weights use
    # the average; ``model_raw`` keeps the optimizer iterate.
    ema_decay = effective_training.get("ema_decay")
    ema = None if ema_decay is None else _ExponentialMovingAverage(model, float(ema_decay))

    planned_batches = len(
        _resolution_batches(train_dataset, batch_size, np.random.default_rng(seed))
    )
    planned_steps = max(1, planned_batches * epochs)
    if args.max_steps is not None:
        planned_steps = min(planned_steps, args.max_steps)
    # Historical paper runs stepped the cosine once per epoch with T_max=10000.
    # The default public trainer still steps per optimizer update unless a
    # profile explicitly chooses the historical schedule.
    scheduler_t_max = int(
        effective_training.get(
            "scheduler_t_max", epochs if scheduler_step_unit == "epoch" else planned_steps
        )
    )
    if scheduler_t_max <= 0:
        raise ValueError("scheduler_t_max must be positive")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    logger.info(
        "start backbone=%s fields=%s train=%d validation=%d device=%s output=%s",
        model_section.get("backbone", model_section.get("model_name", "unspecified")),
        tuple(train_dataset.field_names),
        len(train_dataset),
        len(validation_dataset),
        device,
        run_dir,
    )
    metrics_path = run_dir / "metrics.jsonl"
    best_validation_loss = math.inf
    global_step = 0
    stopped = False
    history: list[dict[str, Any]] = []

    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            epoch_rng = np.random.default_rng(seed + epoch)
            # Native L/M/H point counts differ, so every tensor batch contains
            # one resolution even while the epoch order is globally shuffled.
            batches = _resolution_batches(train_dataset, batch_size, epoch_rng)
            train_weighted_loss = 0.0
            train_seen = 0
            for batch_id, (_, indices) in enumerate(batches):
                samples = [train_dataset[int(index)] for index in indices]
                point_count = samples[0].values.shape[0]
                obs = make_batch(
                    samples,
                    config,
                    query_count=min(query_points, point_count),
                    seed=seed + epoch * 1_000_003 + batch_id,
                    observation_set="training",
                )
                tensors = batch_to_device(obs, device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = model.training_loss(
                    x1=tensors["target_fields"],
                    coords=tensors["query_coords"],
                    obs_coords=tensors["obs_coords"],
                    obs_values=tensors["obs_values"],
                    obs_mask=tensors["obs_mask"],
                    obs_field_ids=tensors["obs_field_ids"],
                    obs_indices=tensors.get("obs_indices"),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch}, batch {batch_id}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                if scheduler_step_unit == "optimizer_step":
                    scheduler.step()
                global_step += 1
                batch_count = len(samples)
                train_weighted_loss += float(loss.detach().cpu()) * batch_count
                train_seen += batch_count
                if args.max_steps is not None and global_step >= args.max_steps:
                    stopped = True
                    break

            if scheduler_step_unit == "epoch":
                scheduler.step()
            train_loss = train_weighted_loss / max(train_seen, 1)
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            # The held-out flow objective is a checkpoint-selection signal;
            # it is not a substitute for physical per-field paper evaluation.
            validation_loss: float | None = None
            if epoch == 1 or epoch % validation_every_epochs == 0 or epoch == epochs or stopped:
                with ema.applied(model) if ema is not None else nullcontext():
                    validation_loss = _evaluate_epoch(
                        model,
                        validation_dataset,
                        config,
                        device,
                        query_points=query_points,
                        validation_samples=validation_samples,
                        batch_size=batch_size,
                        seed=seed,
                        epoch=epoch,
                        observation_set=validation_observation_set,
                    )
                if not math.isfinite(validation_loss):
                    raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            history.append(record)

            improved = validation_loss is not None and validation_loss < best_validation_loss
            if improved:
                assert validation_loss is not None
                best_validation_loss = validation_loss
            save_args = {
                "config": config,
                "stats": stats,
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "best_validation_loss": best_validation_loss,
            }
            # Always leave a usable last checkpoint on first, final, or
            # step-limited epochs even when the periodic interval is longer.
            save_last = (
                epoch == 1
                or epoch % save_every_epochs == 0
                or (live_evaluation_every_epochs and epoch % live_evaluation_every_epochs == 0)
                or epoch == epochs
                or stopped
            )
            if ema is not None and (save_last or improved):
                save_args["averaged_state"] = ema.averaged_state(model)
                save_args["ema_decay"] = ema.decay
            if save_last:
                save_model_checkpoint(run_dir / "last.pt", model, **save_args)
            if improved:
                save_model_checkpoint(run_dir / "best.pt", model, **save_args)
            if plot_every_epochs and (
                epoch == 1 or epoch % plot_every_epochs == 0 or epoch == epochs or stopped
            ):
                _write_loss_artifacts(run_dir, history)
            logger.info(
                "epoch=%d step=%d train_loss=%.7g validation_loss=%s lr=%.4g seconds=%.1f%s",
                epoch,
                global_step,
                train_loss,
                "-" if validation_loss is None else f"{validation_loss:.7g}",
                optimizer.param_groups[0]["lr"],
                record["epoch_seconds"],
                " best" if improved else "",
            )
            # Evaluate the saved state before starting the next epoch, so each
            # milestone's physical errors correspond exactly to its checkpoint.
            if live_evaluation_every_epochs and epoch % live_evaluation_every_epochs == 0:
                milestone_dir = run_dir / "evaluation" / f"epoch_{epoch:04d}"
                steps_to_evaluate = benchmark_steps or (None,)
                for step_index, steps in enumerate(steps_to_evaluate):
                    step_dir = (
                        milestone_dir if step_index == 0 else milestone_dir / f"nfe{steps}"
                    )
                    logger.info(
                        "physical evaluation epoch=%d steps=%s started output=%s",
                        epoch, steps or evaluation.get("steps", 2), step_dir,
                    )
                    _run_live_evaluation(
                        config_path=str(config["config_source"]),
                        checkpoint_path=run_dir / "last.pt",
                        output_dir=step_dir,
                        device=device,
                        epoch=epoch,
                        steps=steps,
                    )
                logger.info("physical evaluation epoch=%d complete", epoch)
            if stopped:
                break
    finally:
        for dataset in (train_dataset, validation_dataset):
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
