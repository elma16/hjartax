#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import tqdm
import jax.numpy as jnp


@dataclass
class WorkoutDatasetConfig:
    # noinspection PyUnresolvedReferences
    """
    Configuration of a WorkoutDataset.
    Contains the names of the columns of the pandas DataFrame containing the data.
    Also contains the hyperparameters about the data preprocessing: sequence_length, stride, history_max_length.


    Attributes
    ----------
    subject_id_column : str
        name of the column containing the subject id.
    workout_id_column : str
        name of the column containing the workout id.
    time_since_start_column : str
        name of the column containing the time since the start of the workout.
    time_of_start_column : str
        name of the column containing the time of the start of the workout.
    heart_rate_column : str
        name of the column containing the heart rate measurements.
    heart_rate_normalized_column : str
        name of the column containing the normalized heart rate measurements.
    activity_columns : List[str]
        names of the columns containing the activity measurements.
    weather_columns : List[str]
        names of the columns containing the weather measurements.
    sequence_length : int
        length of the training samples chunks. Each workout is split into smaller chunks for faster training.
    stride : int
        stride between two training samples chunks from one workout.
    history_max_length : int (optional)
        Number of measurements from past workouts to feed to the encoder. All past workouts will be concatenated.
        None is equivalent to 0 (no history).
    """

    subject_id_column: str = "subject_id"
    workout_id_column: str = "workout_id"
    time_since_start_column: str = "time_grid"
    time_of_start_column: str = "time_start"
    heart_rate_column: str = "heart_rate"
    heart_rate_normalized_column: str = "heart_rate_normalized"
    activity_columns: List[str] = field(default_factory=list)
    weather_columns: List[str] = field(default_factory=list)

    history_max_length: Optional[int] = None
    chunk_size: Optional[int] = 64
    stride: Optional[int] = 32

    def n_activity_channels(self) -> int:
        return len(self.activity_columns)

    def history_dim(self) -> int:
        # heart rate + time + channels + days between last workouts
        return self.n_activity_channels() + 3

    def n_weather_channels(self) -> int:
        return len(self.weather_columns)


class WorkoutDataset:
    """
    A dataset for workouts. Matches the original PyTorch dataset but yields NumPy arrays.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dataset_config: WorkoutDatasetConfig,
    ):
        self.dataset_config: WorkoutDatasetConfig = dataset_config
        self.data = data.copy()

        if (self.dataset_config.chunk_size is None) ^ (
            self.dataset_config.stride is None
        ):
            raise ValueError(
                "chunk_size and stride should both be None or both be given"
            )
        self.chunk_size = self.dataset_config.chunk_size
        self.stride = self.dataset_config.stride

        self.subject_ids = []
        self.subject_indices = []
        self.workout_ids = []
        self.weathers = []
        self.full_workout_lengths = []

        self.heart_rates = []
        self.activity = []
        self.times = []
        self.history = []
        self.encoder_input_dim = self.dataset_config.history_dim()

        self.workout_id_to_all_measurements = dict()
        self.workout_id_to_history = dict()
        self.subject_id_to_index = dict()

        self.n_subjects = 0
        self.n_workouts = 0
        self.len = None

        self.prepare_data()
        self.prepare_history()

    def _add_workout_entry(
        self,
        subject_id,
        subject_index,
        workout_id,
        full_workout_length,
        weather,
        times,
        heart_rates,
        activity,
        start_index=0,
        end_index=None,
    ):
        self.subject_ids.append(subject_id)
        self.subject_indices.append(subject_index)
        self.workout_ids.append(workout_id)
        self.full_workout_lengths.append(full_workout_length)
        self.weathers.append(weather)
        self.heart_rates.append(heart_rates[start_index:end_index])
        self.activity.append(activity[start_index:end_index])
        self.times.append(times[start_index:end_index])

    def prepare_data(self):
        unique_subject_ids = self.data[self.dataset_config.subject_id_column].unique()
        self.subject_id_to_index = {
            s_id: idx for idx, s_id in enumerate(unique_subject_ids)
        }
        for i in tqdm.tqdm(range(self.data.shape[0])):
            current_workout = self.data.iloc[i]
            subject_id = current_workout[self.dataset_config.subject_id_column]
            workout_id = current_workout[self.dataset_config.workout_id_column]
            subject_index = self.subject_id_to_index[subject_id]

            heart_rates = np.array(
                current_workout[self.dataset_config.heart_rate_column]
            )
            times = np.array(
                current_workout[self.dataset_config.time_since_start_column]
            )
            full_workout_length = times.shape[0]
            weather = np.array(
                [
                    current_workout[column]
                    for column in self.dataset_config.weather_columns
                ]
            )
            heart_rates_normalized = np.array(
                current_workout[self.dataset_config.heart_rate_normalized_column]
            )
            activity = np.array(
                [
                    current_workout[column]
                    for column in self.dataset_config.activity_columns
                ]
            ).T
            self.workout_id_to_all_measurements[workout_id] = {
                "times": times,
                "heart_rates_normalized": heart_rates_normalized,
                "activity": activity,
            }

            if self.chunk_size is not None:
                max_start = times.shape[0] - self.chunk_size
                if max_start < 0:
                    indices = [0]
                else:
                    indices = list(range(0, max_start, self.stride))
                    if indices:
                        if indices[-1] + self.chunk_size < times.shape[0]:
                            indices.append(times.shape[0] - self.chunk_size)
                    else:
                        indices = [0]
                for j in indices:
                    self._add_workout_entry(
                        subject_id,
                        subject_index,
                        workout_id,
                        full_workout_length,
                        weather,
                        times,
                        heart_rates,
                        activity,
                        j,
                        j + self.chunk_size,
                    )

            else:
                self._add_workout_entry(
                    subject_id,
                    subject_index,
                    workout_id,
                    full_workout_length,
                    weather,
                    times,
                    heart_rates,
                    activity,
                )

        self.subject_ids = np.array(self.subject_ids)
        self.subject_indices = np.array(self.subject_indices)
        self.workout_ids = np.array(self.workout_ids)
        self.weathers = np.array(self.weathers)
        self.full_workout_lengths = np.array(self.full_workout_lengths)

        if self.chunk_size is not None:
            # can be turned into array because all array are the same length (namely `self.chunk_size`)
            self.heart_rates = np.array(self.heart_rates)
            self.activity = np.array(self.activity)
            self.times = np.array(self.times)

        self.len = len(self.subject_ids)

    def prepare_history(self):
        """
        Prepare the history for each workout.

        The history for each workout w of subject s will include data from the workouts of s that happened before w.
        The included data are:
            normalized heart rates, activity measurements, times of each measurement in their workout, and the number
            of days between the past workout and w.
        The length of the history is determined by the history_max_length parameter in the dataset config, and only
        the most recent measurements will be included.
        """
        if (
            self.dataset_config.history_max_length is None
            or self.dataset_config.history_max_length <= 0
        ):
            return

        workouts_per_subject = defaultdict(list)
        past_workouts_per_workout = dict()
        for _, workout in self.data.sort_values(
            self.dataset_config.time_of_start_column
        ).iterrows():
            workout_id = workout[self.dataset_config.workout_id_column]
            subject_id = workout[self.dataset_config.subject_id_column]
            date = workout[self.dataset_config.time_of_start_column]
            past_workouts_per_workout[workout_id] = (
                date,
                workouts_per_subject[subject_id].copy(),
            )
            workouts_per_subject[subject_id].append((date, workout_id))

        def gather_workout_measurements(workout_all_data, workout_date, reference_date):
            delta_seconds = reference_date - workout_date
            if isinstance(delta_seconds, (datetime.timedelta, pd.Timedelta)):
                delta_seconds = delta_seconds.total_seconds()
            elif isinstance(delta_seconds, np.timedelta64):
                delta_seconds = delta_seconds.astype("timedelta64[s]").astype(float)

            return np.concatenate(
                [
                    workout_all_data["heart_rates_normalized"][:, None],
                    workout_all_data["times"][:, None],
                    workout_all_data["activity"],
                    np.ones_like(workout_all_data["times"])[:, None]
                    * np.log(delta_seconds / (24 * 60 * 60) + 1),
                ],
                axis=-1,
            )

        for w_id in tqdm.tqdm(past_workouts_per_workout):
            date, past_workout_ids = past_workouts_per_workout[w_id]
            past_workouts_data = []
            current_length = 0
            for past_d, past_w_id in past_workout_ids[::-1]:
                past_workout = gather_workout_measurements(
                    self.workout_id_to_all_measurements[past_w_id], past_d, date
                )
                past_workouts_data.append(past_workout)
                current_length += len(past_workout)
                if current_length > self.dataset_config.history_max_length:
                    break
            past_workouts_data = past_workouts_data[::-1]
            if past_workouts_data:
                past_workouts_data = np.concatenate(past_workouts_data, axis=0)
            else:
                past_workouts_data = -np.ones((1, self.encoder_input_dim))
            n_idx = self.dataset_config.history_max_length
            self.workout_id_to_history[w_id] = past_workouts_data[-n_idx:]

        for w_id in self.workout_ids:
            self.history.append(self.workout_id_to_history[w_id])

    def get_indices_by_subject_ids(self, subject_ids: Sequence):
        tmp = set(subject_ids)
        return [i for i, v in enumerate(self.subject_ids) if v in tmp]

    def get_indices_by_workout_ids(self, workout_ids: Sequence):
        tmp = set(workout_ids)
        return [i for i, v in enumerate(self.workout_ids) if v in tmp]

    def __getitem__(self, i):
        res = {
            "subject_id": self.subject_ids[i],
            "subject_index": self.subject_indices[i],
            "workout_id": self.workout_ids[i],
            "heart_rate": self.heart_rates[i],
            "activity": self.activity[i],
            "time": self.times[i],
            "weather": self.weathers[i],
            "full_workout_length": self.full_workout_lengths[i],
            "history": self.history[i] if self.history else None,
            "activity_measurements_names": self.dataset_config.activity_columns,
            "weather_measurements_names": self.dataset_config.weather_columns,
        }

        if res["history"] is not None:
            res["history_length"] = len(res["history"])
        else:
            res["history_length"] = 0

        return res

    def __len__(self):
        return self.len


def _pad_sequences(arrays: Sequence[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    max_len = max(a.shape[0] for a in arrays)
    tail_shape = arrays[0].shape[1:]
    out = np.full((len(arrays), max_len, *tail_shape), pad_value, dtype=np.float32)
    for i, arr in enumerate(arrays):
        out[i, : arr.shape[0]] = arr
    return out


def workout_dataset_collate_fn(batch: List[dict]):
    res = dict()
    for k in batch[0]:
        if batch[0][k] is None:
            res[k] = None
        elif k in ["heart_rate", "activity", "time", "history"]:
            lengths = [len(a[k]) for a in batch]
            arrays = [np.asarray(a[k], dtype=np.float32) for a in batch]
            if len(set(lengths)) == 1:
                res[k] = jnp.asarray(np.stack(arrays))
            else:
                res[k] = jnp.asarray(_pad_sequences(arrays))
        elif k in ["weather"]:
            res[k] = jnp.asarray(np.array([a[k] for a in batch]), dtype=jnp.float32)
        elif k in ["full_workout_length", "history_length", "subject_index"]:
            res[k] = jnp.asarray(np.array([a[k] for a in batch]), dtype=jnp.int32)
        elif k in ["subject_id", "workout_id"]:
            res[k] = np.array([a[k] for a in batch])
        elif k in ["activity_measurements_names", "weather_measurements_names"]:
            res[k] = batch[0][k]
        else:
            res[k] = np.array([a[k] for a in batch])
    return res


class WorkoutDataLoader:
    def __init__(
        self,
        dataset: WorkoutDataset,
        batch_size: int = 256,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)
        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield workout_dataset_collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield workout_dataset_collate_fn(batch)

    def __len__(self):
        n_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            n_batches += 1
        return n_batches


def make_dataloaders(train_dataset, test_dataset, batch_size=256, seed=0):
    train_dataloader = WorkoutDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    test_dataloader = WorkoutDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )

    return train_dataloader, test_dataloader
