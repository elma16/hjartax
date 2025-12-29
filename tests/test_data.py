import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from hjartax.data import (
    WorkoutDataset,
    WorkoutDatasetConfig,
    WorkoutDataLoader,
    make_dataloaders,
    workout_dataset_collate_fn,
)


def _make_df(workout_lengths, subject_ids, start_times, include_weather=False):
    rows = []
    for idx, (length, subject_id, start_time) in enumerate(
        zip(workout_lengths, subject_ids, start_times)
    ):
        time_grid = np.arange(length, dtype=float)
        heart_rate = np.linspace(60, 80, length)
        heart_rate_normalized = heart_rate / 200.0
        speed = np.linspace(0.0, 1.0, length)
        row = {
            "subject_id": subject_id,
            "workout_id": idx + 1,
            "time_grid": time_grid,
            "time_start": np.datetime64(start_time),
            "heart_rate": heart_rate,
            "heart_rate_normalized": heart_rate_normalized,
            "speed": speed,
        }
        if include_weather:
            row["temp"] = np.linspace(10.0, 12.0, length)
        rows.append(row)
    return pd.DataFrame(rows)


def test_workout_dataset_config_helpers():
    config = WorkoutDatasetConfig(
        activity_columns=["a", "b"],
        weather_columns=["w"],
    )
    assert config.n_activity_channels() == 2
    assert config.history_dim() == 5
    assert config.n_weather_channels() == 1


def test_workout_dataset_chunking():
    df = _make_df([10], [1], ["2020-01-01"])
    config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=4,
        stride=3,
    )
    dataset = WorkoutDataset(df, config)
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["heart_rate"].shape[0] == 4
    assert sample["subject_index"] == 0


def test_workout_dataset_history():
    df = _make_df([6, 6], [1, 1], ["2020-01-01", "2020-01-02"])
    config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=5,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, config)
    first = dataset[0]["history"]
    second = dataset[1]["history"]
    assert first.shape[1] == config.history_dim()
    assert np.all(first == -1)
    assert second.shape[1] == config.history_dim()
    assert not np.all(second == -1)


def test_workout_dataset_index_helpers():
    df = _make_df([4, 4, 4], [1, 2, 1], ["2020-01-01", "2020-01-02", "2020-01-03"])
    config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, config)
    idxs = dataset.get_indices_by_subject_ids([1])
    assert len(idxs) == 2
    idxs = dataset.get_indices_by_workout_ids([2])
    assert idxs == [1]


def test_workout_dataset_collate_fn_padding():
    df = _make_df([3, 5], [1, 2], ["2020-01-01", "2020-01-02"])
    config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, config)
    batch = workout_dataset_collate_fn([dataset[0], dataset[1]])
    assert batch["heart_rate"].shape == (2, 5)
    assert jnp.all(batch["heart_rate"][0, 3:] == 0.0)
    assert isinstance(batch["activity"], jax.Array)


def test_workout_dataloader_len_and_iter():
    df = _make_df([4, 4, 4], [1, 1, 1], ["2020-01-01", "2020-01-02", "2020-01-03"])
    config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, config)
    loader = WorkoutDataLoader(dataset, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(loader) == 2
    assert len(batches) == 2
    assert batches[0]["heart_rate"].shape[0] == 2


def test_make_dataloaders():
    df = _make_df([4, 4], [1, 2], ["2020-01-01", "2020-01-02"])
    config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, config)
    train_loader, test_loader = make_dataloaders(dataset, dataset, batch_size=1)
    assert isinstance(train_loader, WorkoutDataLoader)
    assert isinstance(test_loader, WorkoutDataLoader)
