"""Connect YAML profiles, lazy datasets, observation batches, and checkpoints.

Example::

    config = load_config("configs/gl_rbf_base.yaml")
    dataset = make_dataset(config, "train")
    batch = make_batch([dataset[0]], config, query_count=64)

Relative data paths resolve against the checkout containing the config.
Checkpoint loading uses strict model keys and checks normalization statistics.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset

from ..data import (
    AIRFOIL_FIELDS,
    ELASTICITY_FIELDS,
    PAPER_COMBUSTION_FIELDS,
    PDEBENCH_CFD_FIELDS,
    AirfoilDataset,
    CombustionH5Dataset,
    ElasticityDataset,
    FieldSample,
    NormalizationStats,
    ObservationBatch,
    PDEBenchCFDDataset,
    make_observation_batch,
)
from ..models import PointCloudFFM, build_pointcloud_model


def project_root() -> Path:
    """Find the checkout for editable or wheel installs, falling back to cwd."""
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    return cwd


def _resolved_path(value: str | Path, root: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (root / path).resolve())


def load_config(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML mapping and resolve data/output paths against the project root."""
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else project_root()
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        from_cwd = (Path.cwd() / config_path).resolve()
        config_path = from_cwd if from_cwd.is_file() else (root / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    if repo_root is None:
        candidate_root = config_path.parent.parent
        if config_path.parent.name == "configs" and (candidate_root / "pyproject.toml").is_file():
            root = candidate_root
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected a YAML mapping in {config_path}")
    config = copy.deepcopy(dict(loaded))
    config["config_source"] = str(config_path)
    config["project_root"] = str(root)

    for section_name in ("data", "training", "evaluation"):
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            continue
        resolved_section = dict(section)
        for key in ("path", "root", "stats_path", "output_dir", "checkpoint", "checkpoint_path"):
            value = resolved_section.get(key)
            if isinstance(value, (str, Path)) and value:
                resolved_section[key] = _resolved_path(value, root)
        config[section_name] = resolved_section
    return config


def _dataset_data(config: Mapping[str, Any]) -> Mapping[str, Any]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("config must contain a 'data' mapping")
    return data


def make_dataset(
    config: Mapping[str, Any],
    split: str,
    stats: NormalizationStats | Mapping[str, object] | str | Path | None = None,
) -> Dataset[FieldSample]:
    """Build the selected lazy data adapter with one shared normalization contract."""
    data = _dataset_data(config)
    kind = str(data.get("kind", "")).lower()
    # The elasticity and airfoil splits hold out 20 % of cases; the others hold out 10 %.
    default_fraction = 0.8 if kind in {"elasticity", "airfoil"} else 0.9
    train_fraction = float(data.get("train_fraction", default_fraction))
    split_seed = int(data.get("split_seed", 42))
    stats_path = None if stats is not None else data.get("stats_path")

    if kind in {"combustion", "turbulent_combustion"}:
        path = data.get("path")
        if not path:
            raise ValueError("combustion data config requires data.path")
        return CombustionH5Dataset(
            path,
            split=split,
            train_fraction=train_fraction,
            split_seed=split_seed,
            stats=stats,
            stats_path=stats_path,
            stats_frame_chunk=int(data.get("stats_frame_chunk", 16)),
        )

    if kind in {"pdebench", "pdebench_cfd"}:
        root = data.get("root")
        if not root:
            raise ValueError("PDEBench config requires data.root")
        frame_indices = data.get("frame_indices")
        if frame_indices is None and ("frame_start" in data or "frame_end" in data):
            start = int(data.get("frame_start", 0))
            end = int(data.get("frame_end", start))
            if end < start:
                raise ValueError("data.frame_end must be at least data.frame_start")
            frame_indices = tuple(range(start, end + 1))
        resolution_fractions = data.get("resolution_fractions") if split == "train" else None
        return PDEBenchCFDDataset(
            root,
            split=split,
            train_fraction=train_fraction,
            eval_resolution=str(data.get("eval_resolution", "H")),
            frame_indices=frame_indices,
            stats=stats,
            stats_path=stats_path,
            stats_case_chunk=int(data.get("stats_case_chunk", 8)),
            selected_fields=data.get("selected_fields"),
            resolution_fractions=resolution_fractions,
            resolution_seed=int(data.get("resolution_seed", split_seed)),
        )

    if kind in {"elasticity", "airfoil"}:
        root = data.get("root")
        if not root:
            raise ValueError(f"{kind} config requires data.root")
        options = {
            "split": split,
            "train_fraction": train_fraction,
            "split_seed": split_seed,
            "stats": stats,
            "stats_path": stats_path,
        }
        if kind == "elasticity":
            return ElasticityDataset(
                root, **options, sensor_band_cells=int(data.get("sensor_band_cells", 3))
            )
        return AirfoilDataset(
            root,
            **options,
            ellipse_center=tuple(data.get("ellipse_center", (0.5, 0.5))),
            ellipse_semi_axes=tuple(data.get("ellipse_semi_axes", (0.30, 0.12))),
            ellipse_ring_halfwidth=float(data.get("ellipse_ring_halfwidth", 0.08)),
            stats_case_chunk=int(data.get("stats_case_chunk", 64)),
        )

    raise ValueError(
        f"unknown data.kind {kind!r}; expected 'combustion', 'pdebench_cfd', "
        "'elasticity', or 'airfoil'"
    )


def make_batch(
    samples: Sequence[FieldSample],
    config: Mapping[str, Any],
    query_count: int | None = None,
    seed: int | None = None,
    *,
    observation_set: str = "training",
) -> ObservationBatch:
    """Sample observed fields and query targets for one homogeneous resolution batch.

    Training count ranges are passed through to ``make_observation_batch`` so it
    can sample independently per sample/channel and pad with its validity mask.
    Mixed-resolution callers must group samples first, using
    ``PDEBenchCFDDataset.indices_by_resolution``.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("samples must not be empty")
    if observation_set not in {"training", "evaluation"}:
        raise ValueError("observation_set must be 'training' or 'evaluation'")
    resolutions = {sample.resolution for sample in samples}
    if len(resolutions) != 1:
        raise ValueError("group mixed-resolution samples before constructing a batch")

    observations = config.get("observations", {})
    if not isinstance(observations, Mapping):
        raise ValueError("config.observations must be a mapping")
    resolution = samples[0].resolution
    if observation_set == "training" and "train_sensors_per_field" in observations:
        sensors_per_channel = observations["train_sensors_per_field"]
    elif observation_set == "evaluation" and "evaluation_sensors_per_field" in observations:
        sensors_per_channel = observations["evaluation_sensors_per_field"]
    elif "evaluation_sensors_per_resolution" in observations and observation_set == "evaluation":
        per_resolution = observations["evaluation_sensors_per_resolution"]
        sensors_per_channel = per_resolution.get(resolution, per_resolution.get(str(resolution)))
    elif "sensors_per_resolution" in observations:
        per_resolution = observations["sensors_per_resolution"]
        if not isinstance(per_resolution, Mapping):
            raise ValueError("observations.sensors_per_resolution must be a mapping")
        sensors_per_channel = per_resolution.get(resolution, per_resolution.get(str(resolution)))
        if sensors_per_channel is None:
            raise ValueError(f"no sensor budget configured for resolution {resolution!r}")
    elif "sensors_per_field" in observations:
        sensors_per_channel = observations["sensors_per_field"]
    else:
        raise ValueError(
            "config.observations requires train_sensors_per_field, "
            "evaluation_sensors_per_field, sensors_per_resolution, or sensors_per_field"
        )

    training = config.get("training", {})
    default_seed = int(training.get("seed", 42)) if isinstance(training, Mapping) else 42
    if query_count is None and observation_set == "training" and isinstance(training, Mapping):
        configured_queries = training.get("query_points")
        query_count = None if configured_queries is None else int(configured_queries)
    selected_seed = default_seed if seed is None else int(seed)
    observed_fields = observations.get("fields")
    if observed_fields is None:
        raise ValueError("config.observations.fields is required")
    return make_observation_batch(
        samples,
        observed_channels=observed_fields,
        sensors_per_channel=sensors_per_channel,
        query_count=query_count,
        seed=selected_seed,
    )


def batch_to_device(
    batch: ObservationBatch,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Move model tensors while leaving normalization and sample metadata on CPU."""
    target_device = torch.device(device)
    values = {
        "target_fields": batch.target_fields.to(target_device),
        "query_coords": batch.query_coords.to(target_device),
        "obs_coords": batch.obs_coords.to(target_device),
        "obs_values": batch.obs_values.to(target_device),
        "obs_mask": batch.obs_mask.to(target_device),
        "obs_field_ids": batch.obs_field_ids.to(target_device),
    }
    if batch.obs_indices is not None:
        values["obs_indices"] = batch.obs_indices.to(target_device)
    return values


def _stats_mapping(
    stats: NormalizationStats | Mapping[str, object] | None,
) -> dict[str, object] | None:
    if stats is None:
        return None
    if isinstance(stats, NormalizationStats):
        mapping: dict[str, object] = {
            "mean": stats.mean.detach().cpu(),
            "std": stats.std.detach().cpu(),
            "field_names": tuple(stats.field_names),
        }
        if stats.transforms is not None:
            mapping["transforms"] = tuple(stats.transforms)
        return mapping
    return dict(stats)


def save_model_checkpoint(
    checkpoint_path: str | Path,
    model: PointCloudFFM,
    *,
    config: Mapping[str, Any],
    stats: NormalizationStats | Mapping[str, object] | None,
    epoch: int,
    global_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    best_validation_loss: float | None = None,
    averaged_state: Mapping[str, torch.Tensor] | None = None,
    ema_decay: float | None = None,
) -> None:
    """Atomically save weights, optimizer state, profile, and field statistics.

    With ``averaged_state``, ``model`` holds those averaged weights (the ones
    validated) and ``model_raw`` holds the optimizer's current weights.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "dmf-gen-checkpoint-v1",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict() if averaged_state is None else dict(averaged_state),
        "config": copy.deepcopy(dict(config)),
        "stats": _stats_mapping(stats),
        "best_validation_loss": best_validation_loss,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if averaged_state is not None:
        payload["model_raw"] = model.state_dict()
        payload["ema_decay"] = ema_decay
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _field_names_for_model(
    config: Mapping[str, Any], dataset: Dataset[FieldSample] | None
) -> tuple[str, ...]:
    if dataset is not None:
        names = getattr(dataset, "field_names", None)
        if names is not None:
            return tuple(str(name) for name in names)
    data = _dataset_data(config)
    selected = data.get("selected_fields")
    if selected is not None:
        return tuple(str(name) for name in selected)
    kind = str(data.get("kind", "")).lower()
    if kind in {"combustion", "turbulent_combustion"}:
        return tuple(PAPER_COMBUSTION_FIELDS)
    if kind in {"pdebench", "pdebench_cfd"}:
        return tuple(PDEBENCH_CFD_FIELDS)
    if kind == "elasticity":
        return tuple(ELASTICITY_FIELDS)
    if kind == "airfoil":
        return tuple(AIRFOIL_FIELDS)
    raise ValueError("field names require a dataset or data.selected_fields in config")


def _load_payload(path: Path, device: torch.device) -> Any:
    # Checkpoints may come from outside this project (including paper artifacts),
    # so never enable arbitrary pickle loading in the CLI.
    return torch.load(path, map_location=device, weights_only=True)


def _state_dict_from_payload(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise ValueError("checkpoint must contain a model state dictionary")


def _stats_from_payload(
    payload: Any,
    field_names: Sequence[str],
) -> NormalizationStats | None:
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("stats", payload.get("normalization_stats"))
    if isinstance(data, NormalizationStats):
        return data
    if isinstance(data, Mapping):
        return NormalizationStats.from_mapping(data, field_names=field_names)
    # Legacy paper checkpoints sometimes store these tensors directly beside
    # ``model`` instead of nesting them under a ``stats`` key.
    if "mean" in payload and "std" in payload:
        return NormalizationStats.from_mapping(
            {"mean": payload["mean"], "std": payload["std"]},
            field_names=field_names,
        )
    return None


def load_model_checkpoint(
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    *,
    dataset: Dataset[FieldSample] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[PointCloudFFM, dict[str, Any]]:
    """Strictly load a new checkpoint or a legacy ``best.pt``/``last.pt`` payload.

    Legacy paper checkpoints expose the model state under ``model`` and need the
    matching architecture in ``config``. New checkpoints also carry config and
    normalization statistics. When a dataset is given, its field names and
    statistics are checked against saved metadata.
    """
    target_device = torch.device(device)
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    loaded = _load_payload(path, target_device)
    if isinstance(loaded, Mapping):
        payload = dict(loaded)
    else:
        payload = {"model": loaded}

    # Release checkpoints carry their exact model profile. Original paper
    # checkpoints do not, so their caller must supply the matching YAML.
    saved_config = payload.get("config")
    configured_model = config.get("model")
    model_config = configured_model if isinstance(configured_model, Mapping) else config
    if isinstance(saved_config, Mapping):
        saved_model_config = saved_config.get("model")
        if isinstance(saved_model_config, Mapping):
            model_config = saved_model_config
        elif "model_name" in saved_config or "backbone" in saved_config:
            model_config = saved_config
    field_names = _field_names_for_model(config, dataset)
    if dataset is not None and tuple(getattr(dataset, "field_names", ())) != field_names:
        raise ValueError("dataset field_names could not be resolved consistently")

    model = build_pointcloud_model(
        model_config,
        n_fields=len(field_names),
        device=target_device,
    )
    model.load_state_dict(_state_dict_from_payload(payload), strict=True)

    saved_stats = _stats_from_payload(payload, field_names)
    dataset_stats = getattr(dataset, "stats", None) if dataset is not None else None
    if saved_stats is not None and dataset_stats is not None:
        if saved_stats.field_names != dataset_stats.field_names:
            raise ValueError("checkpoint and dataset normalization field order differs")
        if not torch.allclose(saved_stats.mean, dataset_stats.mean, rtol=1e-6, atol=1e-7):
            raise ValueError("checkpoint and dataset normalization means differ")
        if not torch.allclose(saved_stats.std, dataset_stats.std, rtol=1e-6, atol=1e-7):
            raise ValueError("checkpoint and dataset normalization scales differ")
        if saved_stats.transforms is None:
            # Paper checkpoints store mean/std only; the dataset adapter
            # defines which fields were standardized in log space.
            saved_stats = dataset_stats
        elif saved_stats.transforms != dataset_stats.transforms:
            raise ValueError("checkpoint and dataset normalization transforms differ")
    payload["normalization_stats"] = saved_stats or dataset_stats
    return model, payload


def sample_model(
    model: PointCloudFFM,
    batch: ObservationBatch,
    config: Mapping[str, Any],
    device: torch.device | str = "cpu",
    *,
    steps: int | None = None,
    draws: int | None = None,
    consistency: str | None = None,
    query_chunk_size: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Draw normalized fields as ``[draws, batch, queries, fields]``."""
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise ValueError("config.evaluation must be a mapping")
    n_steps = int(evaluation.get("steps", 2) if steps is None else steps)
    n_draws = int(evaluation.get("draws", 1) if draws is None else draws)
    mode = str(
        evaluation.get("observation_consistency", "default_hard")
        if consistency is None
        else consistency
    )
    gather_mode = str(getattr(getattr(model, "model", None), "gather_mode", "rbf"))
    default_execution = (
        "cached_streamed" if gather_mode in {"topk_rbf", "topk_rbf_glres"} else "legacy_full"
    )
    execution_mode = str(evaluation.get("reconstruction_execution_mode", default_execution))
    chunk_size = int(
        evaluation.get("query_chunk_size", 8192) if query_chunk_size is None else query_chunk_size
    )
    if n_steps < 1 or n_draws < 1 or chunk_size < 1:
        raise ValueError("steps, draws, and query_chunk_size must be positive")

    target_device = torch.device(device)
    model.to(target_device)
    tensors = batch_to_device(batch, target_device)
    if mode in {"default_hard", "hard", "endpoint"} and "obs_indices" not in tensors:
        raise ValueError(
            "hard observation consistency requires a query set containing every observed point"
        )
    was_training = model.training
    model.eval()
    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            target_device.index if target_device.index is not None else torch.cuda.current_device()
        ]
    results = []
    try:
        for draw in range(n_draws):
            rng_context = (
                torch.random.fork_rng(devices=cuda_devices) if seed is not None else nullcontext()
            )
            with rng_context:
                if seed is not None:
                    torch.manual_seed(int(seed) + draw)
                    if target_device.type == "cuda":
                        torch.cuda.manual_seed_all(int(seed) + draw)
                results.append(
                    model.sample(
                        coords=tensors["query_coords"],
                        obs_coords=tensors["obs_coords"],
                        obs_values=tensors["obs_values"],
                        obs_mask=tensors["obs_mask"],
                        obs_field_ids=tensors["obs_field_ids"],
                        n_steps=n_steps,
                        clamp_indices=tensors.get("obs_indices"),
                        ode_solver=str(evaluation.get("solver", "euler")),
                        obs_consistency_mode=mode,
                        reconstruction_query_chunk_size=chunk_size,
                        reconstruction_execution_mode=execution_mode,
                    )
                )
    finally:
        model.train(was_training)
    return torch.stack(results, dim=0)


__all__ = [
    "batch_to_device",
    "load_config",
    "load_model_checkpoint",
    "make_batch",
    "make_dataset",
    "project_root",
    "sample_model",
    "save_model_checkpoint",
]
