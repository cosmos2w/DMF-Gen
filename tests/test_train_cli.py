"""Small end-to-end coverage for the public training CLI helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from dmf_gen.cli import train as train_cli
from dmf_gen.cli.common import (
    load_config,
    load_model_checkpoint,
    make_batch,
    make_dataset,
    sample_model,
)

FIELDS = ("CH4", "CO", "T", "U_1", "p")


def test_mixed_resolution_epoch_batches_are_homogeneous() -> None:
    class MixedDataset:
        indices_by_resolution = {
            "L": np.array([0, 1, 2]),
            "M": np.array([3, 4]),
            "H": np.array([5, 6, 7, 8]),
        }

        def __len__(self) -> int:
            return 9

    batches = train_cli._resolution_batches(
        MixedDataset(), batch_size=2, rng=np.random.default_rng(29)
    )
    expected = {
        "L": {0, 1, 2},
        "M": {3, 4},
        "H": {5, 6, 7, 8},
    }
    observed: dict[str, set[int]] = {key: set() for key in expected}
    for resolution, indices in batches:
        assert resolution in expected
        observed[resolution].update(int(index) for index in indices)
    assert observed == expected


def _write_small_combustion(path: Path) -> None:
    n_frames, n_points = 8, 12
    frame = np.arange(n_frames, dtype=np.float32)[:, None, None]
    point = np.arange(n_points, dtype=np.float32)[None, :, None]
    channel = np.arange(len(FIELDS), dtype=np.float32)[None, None, :]
    values = frame * 3.0 + point * 0.2 + channel * 0.7
    x = np.linspace(-1.0, 1.0, n_points, dtype=np.float32)
    coordinates = np.stack((x, x**2, np.sin(x)), axis=-1)[:, None, None, :]
    with h5py.File(path, "w") as handle:
        fields = handle.create_dataset(
            "fields",
            data=values[None, :, :, None, None, :],
            chunks=(1, 1, n_points, 1, 1, len(FIELDS)),
        )
        fields.attrs["selected_fields"] = ",".join(FIELDS)
        handle.create_dataset("coordinates", data=coordinates)
        handle.create_dataset("time", data=np.arange(n_frames, dtype=np.float32))


def _tiny_config(dataset_path: Path) -> dict[str, object]:
    return {
        "data": {
            "kind": "combustion",
            "path": str(dataset_path),
            "train_fraction": 0.75,
            "split_seed": 19,
        },
        "model": {
            "model_name": "GL_rbf",
            "backbone": "GL_rbf",
            "coord_dim": 3,
            "hidden_dim": 16,
            "cond_dim": 8,
            "field_embed_dim": 4,
            "latent_dim": 16,
            "num_latents": 4,
            "num_heads": 4,
            "num_latent_blocks": 1,
            "ff_mult": 2,
            "summary_type": "mean",
            "gather_mode": "rbf",
            "rbf_sigma": 0.2,
            "prior": "iid",
        },
        "observations": {
            "fields": ["T"],
            "train_sensors_per_field": {"min": 2, "max": 4},
            "evaluation_sensors_per_field": 2,
        },
        "training": {
            "seed": 23,
            "epochs": 5,
            "batch_size": 2,
            "query_points": 6,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "validation_samples": 2,
        },
        "evaluation": {
            "steps": 1,
            "draws": 1,
            "solver": "euler",
            "observation_consistency": "default_hard",
            "query_chunk_size": 4,
        },
    }


def test_train_cli_writes_checkpoints_and_reconstructs_with_padded_observations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path
    dataset_dir = root / "dataset"
    config_dir = root / "configs"
    dataset_dir.mkdir()
    config_dir.mkdir()
    dataset_path = dataset_dir / "tiny.h5"
    _write_small_combustion(dataset_path)
    config_path = config_dir / "tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(_tiny_config(dataset_path), sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(train_cli, "project_root", lambda: root)
    output_dir = root / "runs" / "tiny-smoke"

    status = train_cli.main(
        [
            "--config",
            str(config_path),
            "--epochs",
            "5",
            "--batch-size",
            "2",
            "--query-points",
            "6",
            "--max-steps",
            "2",
            "--validation-samples",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "run_config.yaml").is_file()
    assert (output_dir / "train.log").is_file()
    assert (output_dir / "metrics.jsonl").is_file()

    config = load_config(config_path, repo_root=root)
    training_dataset = make_dataset(config, "train")
    training_samples = [training_dataset[i] for i in range(len(training_dataset))]
    padded = make_batch(
        training_samples,
        config,
        query_count=6,
        seed=17,
        observation_set="training",
    )
    assert torch.any(~padded.obs_mask)
    assert torch.all(padded.obs_field_ids[~padded.obs_mask] == -1)
    model, payload = load_model_checkpoint(
        output_dir / "last.pt", config, dataset=training_dataset, device="cpu"
    )
    assert payload["epoch"] == 1
    assert payload["global_step"] == 2

    legacy_path = output_dir / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "mean": training_dataset.stats.mean,
            "std": training_dataset.stats.std,
        },
        legacy_path,
    )
    legacy_model, legacy_payload = load_model_checkpoint(
        legacy_path, config, dataset=training_dataset, device="cpu"
    )
    assert set(legacy_model.state_dict()) == set(model.state_dict())
    assert legacy_payload["normalization_stats"].field_names == FIELDS

    validation_dataset = make_dataset(config, "test", stats=training_dataset.stats)
    evaluation_batch = make_batch(
        [validation_dataset[0]], config, observation_set="evaluation", seed=31
    )
    draws = sample_model(
        model,
        evaluation_batch,
        config,
        device="cpu",
        steps=1,
        draws=2,
        consistency="default_hard",
        query_chunk_size=4,
        seed=41,
    )
    assert draws.shape == (2, 1, 12, len(FIELDS))
    assert torch.isfinite(draws).all()
    training_dataset.close()
    validation_dataset.close()


def test_last_checkpoint_respects_save_interval_and_final_epoch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    config_dir = tmp_path / "configs"
    dataset_dir.mkdir()
    config_dir.mkdir()
    dataset_path = dataset_dir / "tiny.h5"
    _write_small_combustion(dataset_path)
    config = _tiny_config(dataset_path)
    config["training"].update({"epochs": 3, "batch_size": 6, "save_every_epochs": 100})
    config_path = config_dir / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(train_cli, "project_root", lambda: tmp_path)

    last_epochs: list[int] = []
    original_save = train_cli.save_model_checkpoint

    def record_save(path, model, **kwargs):
        if Path(path).name == "last.pt":
            last_epochs.append(kwargs["epoch"])
        return original_save(path, model, **kwargs)

    monkeypatch.setattr(train_cli, "save_model_checkpoint", record_save)
    output_dir = tmp_path / "runs" / "checkpoint-interval"
    assert (
        train_cli.main(
            ["--config", str(config_path), "--device", "cpu", "--output-dir", str(output_dir)]
        )
        == 0
    )
    assert last_epochs == [1, 3]
    assert torch.load(output_dir / "last.pt", weights_only=False)["epoch"] == 3


def test_a0_style_cadence_writes_curve_and_physical_milestones(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise scheduled validation, epoch cosine stepping, plotting, and live evaluation."""
    dataset_dir = tmp_path / "dataset"
    config_dir = tmp_path / "configs"
    dataset_dir.mkdir()
    config_dir.mkdir()
    dataset_path = dataset_dir / "tiny.h5"
    _write_small_combustion(dataset_path)
    config = _tiny_config(dataset_path)
    config["training"].update(
        {
            "epochs": 4,
            "batch_size": 6,
            "validation_samples": 1,
            "validation_every_epochs": 2,
            "validation_observation_set": "training",
            "scheduler_step_unit": "epoch",
            "scheduler_t_max": 10,
            "save_every_epochs": 100,
            "plot_every_epochs": 2,
        }
    )
    config["evaluation"].update(
        {"every_epochs": 2, "max_samples": 1, "plot_samples": 1, "benchmark_steps": [1, 2]}
    )
    config_path = config_dir / "a0-style.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(train_cli, "project_root", lambda: tmp_path)
    run_dir = tmp_path / "runs" / "a0-style"

    assert (
        train_cli.main(
            ["--config", str(config_path), "--device", "cpu", "--output-dir", str(run_dir)]
        )
        == 0
    )
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert records[2]["validation_loss"] is None
    assert records[1]["validation_loss"] is not None
    assert records[0]["learning_rate"] > records[-1]["learning_rate"]
    with (run_dir / "loss_history.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert (run_dir / "loss_history.png").is_file()
    for epoch in (2, 4):
        result = json.loads(
            (run_dir / "evaluation" / f"epoch_{epoch:04d}" / "summary.json").read_text()
        )
        assert result["checkpoint_epoch"] == epoch
        assert result["samples"] == 1
        assert len(result["visualizations"]) == 1
        assert len(result["visualizations"][0]["files"]) == len(FIELDS)
        for filename in result["visualizations"][0]["files"]:
            assert (run_dir / "evaluation" / f"epoch_{epoch:04d}" / filename).is_file()
        second = json.loads(
            (run_dir / "evaluation" / f"epoch_{epoch:04d}" / "nfe2" / "summary.json").read_text()
        )
        assert result["steps"] == 1
        assert second["steps"] == 2
        assert len(second["visualizations"][0]["files"]) == len(FIELDS)
