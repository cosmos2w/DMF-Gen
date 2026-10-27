from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dmf_gen.data import (
    CombustionH5Dataset,
    FieldSample,
    NormalizationStats,
    PDEBenchCFDDataset,
    UnsupportedGeometryAdapter,
    group_samples_by_resolution,
    make_observation_batch,
)

COMBUSTION_FIELDS = ("CH4", "CO", "T", "U_1", "p")
CFD_FIELDS = ("Vx", "Vy", "density", "pressure")


def _coordinates(n_points: int) -> np.ndarray:
    first_axis = np.linspace(-1.0, 1.0, n_points, dtype=np.float32)
    return np.stack((first_axis, first_axis**2, np.zeros_like(first_axis)), axis=-1)[
        :, None, None, :
    ]


def _write_combustion(path: Path, *, n_frames: int = 6, n_points: int = 8) -> np.ndarray:
    frame = np.arange(n_frames, dtype=np.float32)[:, None, None]
    point = np.arange(n_points, dtype=np.float32)[None, :, None]
    channel = np.arange(5, dtype=np.float32)[None, None, :]
    values = frame * 10 + point + channel * 0.25
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "fields", data=values[None, :, :, None, None, :], chunks=(1, 1, n_points, 1, 1, 5)
        )
        dataset.attrs["selected_fields"] = ",".join(COMBUSTION_FIELDS)
        handle.create_dataset("coordinates", data=_coordinates(n_points))
        handle.create_dataset("time", data=np.arange(n_frames, dtype=np.float32))
    return values


def _write_pdebench(root: Path, *, n_cases: int = 6, n_frames: int = 2) -> dict[str, np.ndarray]:
    processed = root / "Processed"
    processed.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    points_by_level = {"L": 4, "M": 9, "H": 16}
    for level, n_points in points_by_level.items():
        case = np.arange(n_cases, dtype=np.float32)[:, None, None, None]
        frame = np.arange(n_frames, dtype=np.float32)[None, :, None, None]
        point = np.arange(n_points, dtype=np.float32)[None, None, :, None]
        channel = np.arange(4, dtype=np.float32)[None, None, None, :]
        values = case * 100 + frame * 10 + point + channel * 0.5
        arrays[level] = values
        with h5py.File(processed / f"CFD_{level}_res.h5", "w") as handle:
            dataset = handle.create_dataset(
                "fields",
                data=values[:, :, :, None, None, :],
                chunks=(1, 1, n_points, 1, 1, 4),
            )
            dataset.attrs["selected_fields"] = ",".join(CFD_FIELDS)
            handle.create_dataset("coordinates", data=_coordinates(n_points))
            handle.create_dataset("time", data=np.arange(n_frames, dtype=np.float32))
    return arrays


def test_combustion_lazy_frame_split_and_train_statistics(tmp_path: Path) -> None:
    path = tmp_path / "combustion.h5"
    values = _write_combustion(path)
    train = CombustionH5Dataset(path, split="train", train_fraction=0.5, split_seed=7)
    assert train._handle is None
    assert len(train) == 3
    assert not set(train.train_frame_ids).intersection(train.test_frame_ids)

    stats = train.stats
    expected = values[train.train_frame_ids].reshape(-1, 5)
    np.testing.assert_allclose(stats.mean.numpy(), expected.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(stats.std.numpy(), expected.std(axis=0), rtol=1e-6)

    sample = train[0]
    assert sample.values.shape == (8, 5)
    assert sample.coordinates.shape == (8, 3)
    assert sample.field_names == COMBUSTION_FIELDS
    torch.testing.assert_close(sample.to_physical(sample.normalized_values), sample.values)
    assert torch.all(sample.coordinates >= 0) and torch.all(sample.coordinates <= 1)
    train.close()


def test_combustion_cached_torch_stats_must_match_fields(tmp_path: Path) -> None:
    path = tmp_path / "combustion.h5"
    _write_combustion(path)
    stats_path = tmp_path / "stats.pt"
    torch.save({"mean": torch.zeros(5), "std": torch.ones(5)}, stats_path)
    dataset = CombustionH5Dataset(path, stats_path=stats_path)
    torch.testing.assert_close(dataset.stats.mean, torch.zeros(5))
    torch.testing.assert_close(dataset.stats.std, torch.ones(5))

    bad_path = tmp_path / "bad_stats.pt"
    torch.save({"mean": torch.zeros(1), "std": torch.ones(1)}, bad_path)
    with pytest.raises(ValueError, match="same nonzero length"):
        invalid_dataset = CombustionH5Dataset(path, stats_path=bad_path)
        _ = invalid_dataset.stats


def test_pdebench_records_case_split_resolution_and_h_stats(tmp_path: Path) -> None:
    arrays = _write_pdebench(tmp_path)
    train = PDEBenchCFDDataset(
        tmp_path,
        split="train",
        train_fraction=0.5,
        resolution_fractions={"L": 1, "M": 1, "H": 1},
        resolution_seed=13,
        stats_case_chunk=2,
    )
    assert train._handle_by_resolution == {}
    assert len(train.sample_records) == 6
    assert {level: len(indices) for level, indices in train.indices_by_resolution.items()} == {
        "L": 2,
        "M": 2,
        "H": 2,
    }
    assert len({record.case_id for record in train.sample_records}) == 3
    assert all(record.resolution in {"L", "M", "H"} for record in train.sample_records)

    h_train = arrays["H"][:3].reshape(-1, 4)
    np.testing.assert_allclose(train.stats.mean.numpy(), h_train.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(train.stats.std.numpy(), h_train.std(axis=0), rtol=1e-6)
    low_record = next(record for record in train.sample_records if record.resolution == "L")
    low_sample = train[low_record.dataset_index]
    assert low_sample.values.shape == (4, 4)
    assert low_sample.logical_shape == (2, 2)
    assert low_sample.stats is train.stats
    torch.testing.assert_close(low_sample.to_physical(), low_sample.values)

    test = PDEBenchCFDDataset(tmp_path, split="test", train_fraction=0.5)
    assert {record.case_id for record in train.sample_records}.isdisjoint(
        record.case_id for record in test.sample_records
    )
    assert {record.resolution for record in test.sample_records} == {"H"}
    assert len(test.indices_by_resolution["H"]) == len(test)


def test_pdebench_loads_compatible_one_field_cached_stats(tmp_path: Path) -> None:
    _write_pdebench(tmp_path)
    stats_path = tmp_path / "vx_stats.pt"
    torch.save({"mean": torch.tensor([2.0]), "std": torch.tensor([4.0])}, stats_path)
    dataset = PDEBenchCFDDataset(
        tmp_path, selected_fields=("Vx",), stats_path=stats_path, frame_indices=(0,)
    )
    assert dataset.stats.field_names == ("Vx",)
    sample = dataset[0]
    torch.testing.assert_close(sample.normalized_values, (sample.values - 2) / 4)


def test_observation_batch_is_deterministic_and_maps_only_coincident_queries() -> None:
    stats = NormalizationStats(torch.zeros(2), torch.ones(2), ("a", "b"))
    sample = FieldSample(
        sample_id="synthetic:0",
        values=torch.arange(16, dtype=torch.float32).reshape(8, 2),
        coordinates=torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24,
        coordinates_raw=torch.arange(24, dtype=torch.float32).reshape(8, 3),
        field_names=("a", "b"),
        stats=stats,
        resolution="H",
        logical_shape=(2, 4),
    )
    full_a = make_observation_batch([sample], ("a", "b"), 2, seed=11)
    full_b = make_observation_batch([sample], ("a", "b"), 2, seed=11)
    torch.testing.assert_close(full_a.obs_coords, full_b.obs_coords)
    torch.testing.assert_close(full_a.obs_values, full_b.obs_values)
    assert full_a.obs_indices is not None
    assert full_a.obs_indices.shape == (1, 4)
    for slot, source_index in enumerate(full_a.query_indices[0]):
        assert torch.equal(full_a.target_fields[0, slot], sample.normalized_values[source_index])
    for slot, source_index in enumerate(full_a.obs_indices[0]):
        assert torch.equal(
            full_a.obs_coords[0, slot], full_a.query_coords[0, source_index]
        )
    assert full_a.to_physical().shape == (1, 8, 2)
    assert full_a.observations_to_physical().shape == (1, 4, 1)

    partial = make_observation_batch(
        [sample], ("a", "b"), {"a": 2, "b": 1}, query_count=1, seed=2
    )
    if partial.obs_indices is not None:
        for slot, source_index in enumerate(partial.obs_indices[0]):
            assert torch.equal(
                partial.obs_coords[0, slot], partial.query_coords[0, source_index]
            )

    different_resolution = FieldSample(
        sample_id="synthetic:L",
        values=sample.values,
        coordinates=sample.coordinates,
        coordinates_raw=sample.coordinates_raw,
        field_names=sample.field_names,
        stats=stats,
        resolution="L",
        logical_shape=sample.logical_shape,
    )
    assert set(group_samples_by_resolution([sample, different_resolution])) == {"H", "L"}
    with pytest.raises(ValueError, match="same native resolution"):
        make_observation_batch([sample, different_resolution], "a", 1)


def test_variable_sensor_ranges_pad_each_batch_row() -> None:
    stats = NormalizationStats(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]), ("a", "b"))
    sample = FieldSample(
        sample_id="synthetic:0",
        values=torch.arange(16, dtype=torch.float32).reshape(8, 2),
        coordinates=torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24,
        coordinates_raw=torch.arange(24, dtype=torch.float32).reshape(8, 3),
        field_names=("a", "b"),
        stats=stats,
        resolution="H",
        logical_shape=(2, 4),
    )
    batch = make_observation_batch(
        [sample] * 8,
        observed_channels=("a", "b"),
        sensors_per_channel={"min": 1, "max": 3},
        seed=17,
    )

    assert batch.obs_mask.shape[0] == 8
    assert torch.any(~batch.obs_mask)
    assert batch.obs_indices is not None
    assert torch.all(batch.obs_field_ids[~batch.obs_mask] == -1)
    assert torch.all(batch.obs_indices[~batch.obs_mask] == -1)
    for row in range(batch.obs_mask.shape[0]):
        active_ids = batch.obs_field_ids[row, batch.obs_mask[row]]
        assert torch.all((active_ids == 0) | (active_ids == 1))
        assert 1 <= int((active_ids == 0).sum()) <= 3
        assert 1 <= int((active_ids == 1).sum()) <= 3
        active_indices = batch.obs_indices[row, batch.obs_mask[row]]
        assert torch.all(active_indices >= 0)
        for obs_slot, query_slot in zip(
            torch.nonzero(batch.obs_mask[row], as_tuple=False).flatten(),
            active_indices,
            strict=True,
        ):
            torch.testing.assert_close(
                batch.obs_coords[row, obs_slot], batch.query_coords[row, query_slot]
            )
    assert torch.all(batch.observations_to_physical()[~batch.obs_mask] == 0)

    per_channel = make_observation_batch(
        [sample] * 3,
        observed_channels=("a", "b"),
        sensors_per_channel={"a": {"min": 1, "max": 2}, "b": 1},
        seed=21,
    )
    for row in range(per_channel.obs_mask.shape[0]):
        active_ids = per_channel.obs_field_ids[row, per_channel.obs_mask[row]]
        assert 1 <= int((active_ids == 0).sum()) <= 2
        assert int((active_ids == 1).sum()) == 1


def test_geometry_adapter_is_an_explicit_stub() -> None:
    stats = NormalizationStats(torch.zeros(1), torch.ones(1), ("field",))
    sample = FieldSample(
        sample_id="synthetic",
        values=torch.ones(2, 1),
        coordinates=torch.zeros(2, 1),
        coordinates_raw=torch.zeros(2, 1),
        field_names=("field",),
        stats=stats,
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        UnsupportedGeometryAdapter().adapt(sample, geometry=object())
