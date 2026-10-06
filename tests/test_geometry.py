"""Changing-geometry adapters: channel order, masking, sensors, transforms, and CLI paths."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dmf_gen import build_pointcloud_model
from dmf_gen.cli import evaluate as evaluate_cli
from dmf_gen.cli import reconstruct as reconstruct_cli
from dmf_gen.cli import train as train_cli
from dmf_gen.cli.common import load_config, load_model_checkpoint, make_batch, make_dataset
from dmf_gen.data import (
    AirfoilDataset,
    ElasticityDataset,
    EnclosingGrid,
    FieldSample,
    NormalizationStats,
    SupportGeometry,
    SupportMaskAdapter,
    make_observation_batch,
)
from dmf_gen.data.geometry import dilate_four_connected, ellipse_ring
from dmf_gen.data.io import seeded_case_split
from dmf_gen.metrics import (
    airfoil_drag_coefficients,
    airfoil_mach,
    clamped_relative_errors,
    geometry_errors,
    support_relative_l2,
)

AIRFOIL_TRANSFORMS = ("log", "identity", "identity", "log", "identity")


def _write_elasticity(root: Path, *, n_cases: int = 10, side: int = 11) -> dict[str, np.ndarray]:
    folder = root / "Interp"
    folder.mkdir(parents=True)
    y, x = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side), indexing="ij")
    material = np.ones((side, side, n_cases), dtype=np.int64)
    stress = np.zeros((side, side, n_cases))
    for case in range(n_cases):
        radius = 0.15 + 0.02 * (case % 4)
        void = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2
        material[..., case] = (~void).astype(np.int64)
        stress[..., case] = np.where(void, 0.0, 100.0 + 50.0 * x + 10.0 * case)
    np.save(folder / "Random_UnitCell_sigma_10_interp.npy", stress)
    np.save(folder / "Random_UnitCell_mask_10_interp.npy", material)
    return {"stress": stress, "material": material}


def _write_airfoil(
    root: Path, *, n_cases: int = 10, side: int = 21, swap_pressure_and_mach: bool = False
) -> dict[str, np.ndarray]:
    folder = root / "naca_interp_5f"
    folder.mkdir(parents=True)
    x = np.linspace(-0.5, 1.5, side)
    y = np.linspace(-1.0, 1.0, side)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    unit_x, unit_y = (grid_x + 0.5) / 2.0, (grid_y + 1.0) / 2.0
    channels = np.zeros((n_cases, 5, side, side), dtype=np.float32)
    fluid = np.ones((n_cases, side, side), dtype=bool)
    for case in range(n_cases):
        body = (np.abs(unit_x - 0.5) < 0.2) & (np.abs(unit_y - 0.5) < 0.02 + 0.03 * (case % 2))
        fluid[case] = ~body
        rho = 1.0 + 0.05 * np.sin(3 * grid_x) + 0.01 * case
        u = 0.8 + 0.05 * np.cos(2 * grid_y)
        v = 0.02 * np.sin(grid_x + grid_y)
        p = 1.0 + 0.1 * np.cos(grid_x) - 0.02 * case
        mach = airfoil_mach(rho, u, v, p)
        for channel, value in enumerate((rho, u, v, p, mach)):
            channels[case, channel] = np.where(body, 0.0, value)
    if swap_pressure_and_mach:
        channels[:, [3, 4]] = channels[:, [4, 3]]
    np.save(folder / "NACA_Q_interp.npy", channels)
    np.save(folder / "NACA_mask_interp.npy", fluid)
    np.save(folder / "NACA_X_interp.npy", np.broadcast_to(grid_x, fluid.shape).astype(np.float32))
    np.save(folder / "NACA_Y_interp.npy", np.broadcast_to(grid_y, fluid.shape).astype(np.float32))
    return {"channels": channels, "fluid": fluid}


def test_case_split_reproduces_the_paper_held_out_cases() -> None:
    # Sizes and first held-out IDs of the manuscript's elasticity and airfoil splits.
    for n_cases, n_test, first_held_out in (
        (2000, 400, [5, 13, 16, 19, 32]),
        (2490, 498, [5, 13, 37, 41, 42]),
    ):
        train, test = seeded_case_split(n_cases, 0.8, 42)
        assert test.size == n_test and np.union1d(train, test).size == n_cases
        np.testing.assert_array_equal(test[:5], first_held_out)


def test_four_connected_dilation_matches_scipy() -> None:
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(3)
    for iterations in (1, 2, 3):
        mask = rng.random((17, 23)) > 0.93
        expected = ndimage.binary_dilation(mask, iterations=iterations)
        np.testing.assert_array_equal(dilate_four_connected(mask, iterations), expected)


def test_elasticity_channels_support_sensors_and_statistics(tmp_path: Path) -> None:
    raw = _write_elasticity(tmp_path)
    train = ElasticityDataset(tmp_path, split="train", train_fraction=0.8, split_seed=42)
    test = ElasticityDataset(tmp_path, split="test", stats=train.stats)
    assert len(train) == 8 and len(test) == 2
    assert set(train.case_ids).isdisjoint(test.case_ids)

    expected_train = np.stack(
        [raw["stress"][..., train.case_ids], raw["material"][..., train.case_ids]], axis=-1
    ).reshape(-1, 2)
    np.testing.assert_allclose(train.stats.mean, expected_train.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(train.stats.std, expected_train.std(axis=0), rtol=1e-6)

    sample = test[0]
    case = int(test.case_ids[0])
    assert sample.field_names == ("sigma", "mask")
    assert sample.support_field == "mask"
    assert sample.logical_shape == (11, 11)
    torch.testing.assert_close(
        sample.values[:, 0], torch.from_numpy(raw["stress"][..., case].reshape(-1)).float()
    )
    material = torch.from_numpy(raw["material"][..., case].reshape(-1).astype(bool))
    torch.testing.assert_close(sample.support, material)
    assert torch.all(sample.values[~material, 0] == 0)

    # Sensor candidates: material points within three grid steps (L1) of the void.
    in_material = raw["material"][..., case].astype(bool)
    void_points = np.argwhere(~in_material)
    rows, cols = np.indices(in_material.shape)
    steps = np.abs(rows[..., None] - void_points[:, 0])
    steps = steps + np.abs(cols[..., None] - void_points[:, 1])
    band = in_material & (steps.min(axis=-1) <= 3)
    np.testing.assert_array_equal(sample.sensor_mask.numpy(), band.reshape(-1))

    items = [test[0], test[1]]
    batch = make_observation_batch(items, ["sigma"], {"min": 3, "max": 5}, seed=5)
    for row, item in enumerate(items):
        active = batch.obs_indices[row, batch.obs_mask[row]]
        sensors = batch.query_indices[row, active]
        assert torch.all(item.sensor_mask[sensors])
    with pytest.raises(ValueError, match="hidden support target"):
        make_observation_batch([sample], ["mask"], 2)
    with pytest.raises(ValueError, match="sensor candidates"):
        make_observation_batch([sample], ["sigma"], int(sample.sensor_mask.sum()) + 1)


def test_airfoil_transforms_fill_values_and_ring(tmp_path: Path) -> None:
    raw = _write_airfoil(tmp_path)
    train = AirfoilDataset(tmp_path, split="train")
    test = AirfoilDataset(tmp_path, split="test", stats=train.stats)
    assert train.stats.transforms == AIRFOIL_TRANSFORMS
    assert train.grid.x_range == (-0.5, 1.5) and train.grid.y_range == (-1.0, 1.0)

    sample = test[0]
    case = int(test.case_ids[0])
    fluid = torch.from_numpy(raw["fluid"][case].reshape(-1))
    assert sample.field_names == ("rho", "u", "v", "p", "mask")
    source = torch.from_numpy(raw["channels"][case, :4].reshape(4, -1).T)
    torch.testing.assert_close(sample.values[fluid, :4], source[fluid])
    torch.testing.assert_close(sample.values[~fluid, 0], torch.ones(int((~fluid).sum())))
    torch.testing.assert_close(sample.values[~fluid, 3], torch.ones(int((~fluid).sum())))
    assert torch.all(sample.values[~fluid, 1:3] == 0)
    torch.testing.assert_close(sample.values[:, 4], fluid.float())

    normalized = sample.normalized_values
    expected_rho = (torch.log(sample.values[:, 0]) - train.stats.mean[0]) / train.stats.std[0]
    torch.testing.assert_close(normalized[:, 0], expected_rho)
    torch.testing.assert_close(sample.to_physical(normalized), sample.values, rtol=1e-5, atol=1e-6)
    arbitrary = sample.to_physical(torch.randn(normalized.shape) * 50)
    assert torch.all(arbitrary[:, [0, 3]] > 0)

    batch = make_observation_batch([sample], ["p"], 6, seed=3)
    active = batch.obs_indices[0, batch.obs_mask[0]]
    sensors = batch.query_indices[0, active]
    assert torch.all(torch.from_numpy(train.sensor_ring)[sensors])
    observed = batch.observations_to_physical()[0, :6, 0]
    torch.testing.assert_close(observed, sample.values[sensors, 3])
    assert torch.equal(test[1].sensor_mask, sample.sensor_mask)
    ring = ellipse_ring(train.grid, (0.5, 0.5), (0.30, 0.12), 0.08)
    np.testing.assert_array_equal(train.sensor_ring, ring)


def test_airfoil_rejects_a_wrong_channel_order(tmp_path: Path) -> None:
    _write_airfoil(tmp_path, swap_pressure_and_mach=True)
    with pytest.raises(ValueError, match="rho, u, v, p, Ma"):
        AirfoilDataset(tmp_path, split="train")


def test_statistics_files_adopt_or_reject_transforms(tmp_path: Path) -> None:
    _write_airfoil(tmp_path)
    fitted = AirfoilDataset(tmp_path, split="train").stats
    plain = tmp_path / "plain.pt"
    torch.save({"mean": fitted.mean, "std": fitted.std}, plain)
    adopted = AirfoilDataset(tmp_path, split="test", stats_path=plain).stats
    assert adopted.transforms == AIRFOIL_TRANSFORMS
    wrong = tmp_path / "wrong.pt"
    torch.save(
        {"mean": fitted.mean, "std": fitted.std, "transforms": ("identity",) * 5}, wrong
    )
    with pytest.raises(ValueError, match="transforms"):
        _ = AirfoilDataset(tmp_path, split="test", stats_path=wrong).stats


def test_support_mask_adapter_contract() -> None:
    grid = EnclosingGrid(shape=(3, 4), x_range=(-1.0, 1.0), y_range=(0.0, 2.0))
    stats = NormalizationStats(
        torch.tensor([0.5, 0.0]), torch.tensor([1.0, 2.0]), ("mask", "p"), ("identity", "log")
    )
    support = np.ones(12, dtype=bool)
    support[5] = False
    candidates = np.zeros(12, dtype=bool)
    candidates[[0, 11]] = True
    with pytest.raises(ValueError, match="positive fill"):
        SupportMaskAdapter(stats)
    with pytest.raises(ValueError, match="on the physical support"):
        SupportGeometry(grid, support=support, sensor_candidates=~support)
    adapter = SupportMaskAdapter(stats, fill_values={"p": 1.0})
    physical = FieldSample(
        sample_id="synthetic:0",
        values=torch.full((12, 1), 3.0),
        coordinates=torch.from_numpy(grid.normalized_coordinates()),
        coordinates_raw=torch.from_numpy(grid.physical_coordinates()),
        field_names=("p",),
        stats=stats.subset(("p",)),
        case_id=7,
    )
    adapted = adapter.adapt(physical, SupportGeometry(grid, support, candidates, "fluid"))
    assert adapted.field_names == ("mask", "p")
    assert adapted.values[5].tolist() == [0.0, 1.0]
    assert adapted.values[0].tolist() == [1.0, 3.0]
    assert adapted.metadata["support_name"] == "fluid" and adapted.case_id == 7
    torch.testing.assert_close(adapted.coordinates_raw[-1], torch.tensor([1.0, 2.0, 0.0]))
    assert adapted.sensor_mask.nonzero().flatten().tolist() == [0, 11]


def test_geometry_and_field_metrics() -> None:
    support = np.array([True, True, False, False, True])
    predicted_mask = np.array([0.9, 0.2, 0.1, 0.7, 1.0])
    errors = geometry_errors(support, predicted_mask)
    assert errors["geometry_relative_l2"] == pytest.approx(np.sqrt(2 / 2))
    assert errors["geometry_iou"] == pytest.approx(1 / 3)
    target = np.array([1.0, 2.0, 50.0, 50.0, 2.0])
    prediction = np.array([1.0, 2.0, 0.0, 0.0, 1.0])
    assert support_relative_l2(target, prediction, support) == pytest.approx(1 / 3)
    summary = clamped_relative_errors([2.0, 0.001, -1.0], [2.2, 0.101, -1.0])
    floor = 0.15 * np.sqrt((4 + 1e-6 + 1) / 3)
    assert summary["denominator_floor"] == pytest.approx(floor)
    assert summary["clamped_cases"] == 1
    assert summary["mean_absolute_relative_error"] == pytest.approx((0.1 + 0.1 / floor) / 3)


def test_drag_coefficient_vanishes_for_uniform_pressure_and_signs_streamwise_load() -> None:
    pytest.importorskip("scipy")
    side = 41
    yy, xx = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side), indexing="ij")
    unit_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    fluid = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2 > 0.15**2).ravel()
    fields = np.zeros((side * side, 5))
    fields[:, 0], fields[:, 1], fields[:, 3], fields[:, 4] = 1.0, 0.8, 1.0, fluid
    channels = {"rho": 0, "u": 1, "v": 2, "p": 3, "mask": 4}
    ranges = ((-0.5, 1.5), (-1.0, 1.0))
    loaded = fields.copy()
    loaded[:, 3] = 1.0 + 0.2 * (0.5 - unit_xy[:, 0])
    drag = airfoil_drag_coefficients(fields, loaded, unit_xy, (side, side), ranges, channels)
    assert drag is not None
    assert drag[0] == pytest.approx(0.0, abs=1e-12)
    # The loaded pressure falls by 0.1 per unit x; on a cylinder of radius r = 0.3
    # with q = 0.32 and reference length 2r this gives C_D = 0.1 * pi * r / (2 q).
    assert drag[1] == pytest.approx(0.1 * np.pi * 0.3 / (2 * 0.32), rel=0.05)


def _tiny_airfoil_config(root: Path) -> dict[str, object]:
    return {
        "data": {"kind": "airfoil", "root": str(root), "train_fraction": 0.8, "split_seed": 42},
        "model": {
            "model_name": "GL_rbf_CQ",
            "backbone": "GL_rbf_ENH_CQ",
            "coord_dim": 3,
            "hidden_dim": 16,
            "cond_dim": 8,
            "field_embed_dim": 4,
            "latent_dim": 16,
            "num_latents": 4,
            "num_heads": 4,
            "num_latent_blocks": 1,
            "ff_mult": 2,
            "summary_type": "cls",
            "gather_mode": "topk_rbf_glres",
            "gather_topk": 4,
            "rbf_sigma": 0.06,
            "learnable_rbf_sigma": True,
            "cq_query_dim": 16,
            "cq_readout_rank": 8,
            "cq_readout_heads": 4,
            "cq_time_conditioning": "sinusoidal_film",
            "cq_time_embed_dim": 16,
            "cq_measurement_support_mode": "rbf_value_support",
            "prior": "rff",
            "rff_features": 8,
        },
        "observations": {
            "fields": ["p"],
            "train_sensors_per_field": {"min": 4, "max": 6},
            "evaluation_sensors_per_field": 4,
        },
        "training": {
            "seed": 3,
            "epochs": 2,
            "batch_size": 4,
            "query_points": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "ema_decay": 0.9,
            "validation_samples": 2,
        },
        "evaluation": {"steps": 2, "draws": 2, "solver": "euler", "query_chunk_size": 128},
    }


def test_airfoil_train_evaluate_and_reconstruct_cli(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("scipy")
    dataset_root = tmp_path / "dataset" / "airfoil"
    _write_airfoil(dataset_root)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "airfoil_tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(_tiny_airfoil_config(dataset_root), sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(train_cli, "project_root", lambda: tmp_path)
    run_dir = tmp_path / "runs" / "airfoil-tiny"
    status = train_cli.main(
        [
            "--config",
            str(config_path),
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert status == 0
    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    assert checkpoint["stats"]["transforms"] == AIRFOIL_TRANSFORMS
    assert set(checkpoint["model_raw"]) == set(checkpoint["model"])
    assert any(
        not torch.equal(checkpoint["model"][key], checkpoint["model_raw"][key])
        for key in checkpoint["model"]
    )
    assert checkpoint["ema_decay"] == pytest.approx(0.9)

    config = load_config(config_path, repo_root=tmp_path)
    dataset = make_dataset(config, "test")
    _, payload = load_model_checkpoint(run_dir / "best.pt", config, dataset=dataset)
    assert payload["normalization_stats"].transforms == AIRFOIL_TRANSFORMS
    batch = make_batch([dataset[0]], config, observation_set="evaluation", seed=1)
    assert int(batch.obs_mask.sum()) == 4 and batch.obs_indices is not None

    arguments = [
        "dmf-evaluate",
        "--config",
        str(config_path),
        "--checkpoint",
        str(run_dir / "best.pt"),
        "--output-dir",
        str(run_dir / "evaluation"),
        "--device",
        "cpu",
        "--max-samples",
        "2",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    summary = evaluate_cli.evaluate(evaluate_cli._arguments())
    assert summary["support_field"] == "mask"
    assert summary["observed_fields"] == ["p"]
    assert set(summary["mean_physical_relative_l2_by_field"]) == {"rho", "u", "v", "p"}
    task = summary["task_metrics"]
    assert task["per_case"]["Ma_relative_l2"]["cases"] == 2
    assert task["per_case"]["drag_coefficient_true"]["cases"] == 2
    assert task["relative_errors"]["drag_coefficient"]["cases"] == 2
    with (run_dir / "evaluation" / "per_sample_field.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["field"] for row in rows} == {"rho", "u", "v", "p"}
    assert json.loads((run_dir / "evaluation" / "summary.json").read_text())["samples"] == 2

    arguments = [
        "dmf-reconstruct",
        "--config",
        str(config_path),
        "--checkpoint",
        str(run_dir / "best.pt"),
        "--output",
        str(run_dir / "reconstruction.npz"),
    ]
    monkeypatch.setattr("sys.argv", arguments)
    reconstruct_cli.reconstruct(reconstruct_cli._arguments())
    with np.load(run_dir / "reconstruction.npz") as archive:
        assert tuple(archive["field_names"]) == ("rho", "u", "v", "p", "mask")
        assert archive["target"].shape == (21 * 21, 5)
        assert np.all(archive["prediction_mean"][:, [0, 3]] > 0)
        assert np.all(archive["obs_values"][archive["obs_mask"], 0] > 0)


def test_legacy_checkpoint_stats_take_the_dataset_transforms(tmp_path: Path) -> None:
    dataset_root = tmp_path / "airfoil"
    _write_airfoil(dataset_root)
    config = _tiny_airfoil_config(dataset_root)
    dataset = AirfoilDataset(dataset_root, split="test")
    model = build_pointcloud_model(config["model"], n_fields=5)
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {"model": model.state_dict(), "mean": dataset.stats.mean, "std": dataset.stats.std},
        legacy,
    )
    _, payload = load_model_checkpoint(legacy, config, dataset=dataset)
    assert payload["normalization_stats"].transforms == AIRFOIL_TRANSFORMS


def test_elasticity_config_and_case_metrics(tmp_path: Path) -> None:
    _write_elasticity(tmp_path)
    dataset = make_dataset({"data": {"kind": "elasticity", "root": str(tmp_path)}}, "test")
    assert (dataset.train_case_ids.size, len(dataset)) == (8, 2)
    sample = dataset[0]
    target = sample.values.numpy()
    prediction = target.copy()
    prediction[:, 0] *= 0.5
    metrics = dataset.case_metrics(sample, target, prediction, np.arange(len(target)))
    material = target[:, 1] > 0.5
    expected = target[material, 0].max() / target[material, 0].mean()
    assert metrics["stress_concentration_true"] == pytest.approx(expected)
    # A uniform scale changes the peak stress but not the concentration factor.
    assert metrics["stress_concentration_pred"] == pytest.approx(expected)
    assert metrics["peak_stress_pred"] == pytest.approx(0.5 * metrics["peak_stress_true"])
    assert metrics["geometry_relative_l2"] == 0.0 and metrics["geometry_iou"] == 1.0


def test_exponential_moving_average_updates_applies_and_restores() -> None:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = train_cli._ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    with ema.applied(model):
        torch.testing.assert_close(model.weight, torch.full((1, 2), 2.0))
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))
    torch.testing.assert_close(ema.averaged_state(model)["weight"], torch.full((1, 2), 2.0))
