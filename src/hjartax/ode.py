#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd

from hjartax.data import WorkoutDatasetConfig
from hjartax.modules_cnn import CNNEncoder
from hjartax.modules_dense_nn import DenseNN, PersonalizedScalarNN, PersonalizationType

EPSILON = 1e-3


@dataclass
class OdeConfig:
    # noinspection PyUnresolvedReferences
    """
    Configuration of a OdeModel.

    Attributes
    ----------
    data_config: WorkoutDatasetConfig
        Configuration of the dataset.
    learning_rate: float, default=1e-3
        Learning rate of the optimizer.
    n_epochs: int, default=50
        Number of epochs to train.
    seed: int, default=0
        Random seed.
    ode_step_size: float, default=1.0
        Step size of the ODE solver. 0.5 is a bit more stable than 1 but slower.
    clip_gradient: float, default=5.0
        Clip the gradient norm to this value. 0 to disable.
    subject_embedding_dim: int, default=8
        Dimension of the subject specific embedding.
    encoder_embedding_dim: int, default=8
        Dimension of the embedding of the workout history encoder.
    encoder_kernel_size: int, default=6
        Kernel size of the causal CNN encoder.
    encoder_layers: str, default="128"
        Number of channels of the causal CNN encoder. The string is of the form "dim_layer1,dim_layer_2,dim_layer3".
    embedding_reg_strength: float, default=1.0
        L2 regularization strength of the embeddings.
    decoder_reg_strength: float, default=0.0
        L2 regularization strength of the decoder networks: `activity_fn`, `weather_fn`, `fatigue_fn`.
    encoder_reg_strength: float, default=0.0
        L2 regularization strength of the encoder.
    ode_parameter_layers: str, default="32,8"
        Number of hidden units of the networks which decode the embeddings into the ODE parameters.
    ode_parameter_activation: str, default="softplus"
        Activation function of the networks which decode the embeddings into the ODE parameters.
    activity_fn_layers: str, default="128,64"
        Number of hidden units of the network which decodes the ODE parameter `I` into the demand `f(I)`.
    activity_fn_activation: str, default="softplus"
        Activation function of the network which decodes the ODE parameter `I` into the demand `f(I)`.
    activity_fn_embedding_personalization: ["none", "softmax", "concatenate"], default="softmax"
        Personalization of the demand function with the embedding.
    weather_fn_embedding_personalization: ["none", "softmax", "concatenate"], default="none"
        Personalization of the weather function with the embedding. See `activity_fn_embedding_personalization`.
    fatigue_fn_embedding_personalization: ["none", "softmax", "concatenate"], default="none"
        Personalization of the fatigue function with the embedding. See `activity_fn_embedding_personalization`.
    ranges_A_B_alpha_beta: str, default="-3 5, -3 5, 0.1 3, 0.1 3"
        Ranges of the ODE parameters `A`, `B`, `alpha`, `beta`.
        The string is of the form "A_min A_max, B_min B_max, alpha_min alpha_max, beta_min beta_max".
    range_activity_fn: str, default="30 250"
        Range of the demand function. The string is of the form "min max".
    python_start: str
        Date and time automatically generated when the model is initialized.
    """

    data_config: WorkoutDatasetConfig

    # training
    learning_rate: float = 1e-3
    n_epochs: int = 50
    seed: int = 0
    ode_step_size: float = 1.0
    clip_gradient: float = 5.0

    # embeddings
    subject_embedding_dim: int = 8
    encoder_embedding_dim: int = 8
    encoder_kernel_size: int = 6
    encoder_layers: str = "128"

    # regularization
    embedding_reg_strength: float = 1.0
    decoder_reg_strength: float = 0.0
    encoder_reg_strength: float = 0.0

    # architecture of the networks
    ode_parameter_layers: str = "32,8"
    ode_parameter_activation: str = "softplus"
    activity_fn_layers: str = "128,64"
    activity_fn_activation: str = "softplus"

    # personalization with embeddings
    activity_fn_embedding_personalization: PersonalizationType = "softmax"
    weather_fn_embedding_personalization: PersonalizationType = "none"
    fatigue_fn_embedding_personalization: PersonalizationType = "none"

    # ranges of the parameters
    ranges_A_B_alpha_beta: str = "-3 5, -3 5, 0.1 3, 0.1 3"
    range_activity_fn: str = "30 250"

    python_start: str = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


class EmbeddingStore(eqx.Module):
    subject_id_column: str = eqx.field(static=True)
    workout_id_column: str = eqx.field(static=True)
    subject_embedding_dim: int = eqx.field(static=True)
    encoder_embedding_dim: int = eqx.field(static=True)
    encoder_kernel_size: int = eqx.field(static=True)
    encoder_layers: list[int] = eqx.field(static=True)
    encoder_input_dim: int = eqx.field(static=True)

    subject_id_to_embedding_index: Dict = eqx.field(static=True)
    workout_id_to_embedding_index: Dict = eqx.field(static=True)
    n_subject_embeddings: int = eqx.field(static=True)

    subject_embeddings: Optional[jax.Array]
    encoder: Optional[CNNEncoder]
    dim_embedding: int = eqx.field(static=True)

    def __init__(
        self,
        ode_config: OdeConfig,
        workouts_info: pd.DataFrame,
        key: jax.Array,
    ):
        self.subject_id_column = ode_config.data_config.subject_id_column
        self.workout_id_column = ode_config.data_config.workout_id_column
        self.subject_embedding_dim = ode_config.subject_embedding_dim
        self.encoder_embedding_dim = ode_config.encoder_embedding_dim
        self.encoder_kernel_size = ode_config.encoder_kernel_size
        self.encoder_layers = list(map(int, ode_config.encoder_layers.split(",")))
        self.encoder_input_dim = ode_config.data_config.history_dim()

        k1, k2 = jax.random.split(key, 2)
        if self.subject_embedding_dim is None or self.subject_embedding_dim == 0:
            self.n_subject_embeddings = 0
            self.subject_id_to_embedding_index = {}
            self.workout_id_to_embedding_index = {}
            self.subject_embeddings = None
        else:
            unique_subject_ids = workouts_info[self.subject_id_column].unique()
            self.n_subject_embeddings = len(unique_subject_ids)
            self.subject_id_to_embedding_index = {
                s_id: idx for idx, s_id in enumerate(unique_subject_ids)
            }
            self.workout_id_to_embedding_index = {}
            for s_id, w_id in workouts_info[
                [self.subject_id_column, self.workout_id_column]
            ].values:
                self.workout_id_to_embedding_index[w_id] = (
                    self.subject_id_to_embedding_index[s_id]
                )
            self.subject_embeddings = jax.random.normal(
                k1, (self.n_subject_embeddings, self.subject_embedding_dim)
            )

        if self.encoder_embedding_dim is None or self.encoder_embedding_dim <= 0:
            self.encoder = None
        else:
            dims = [self.encoder_input_dim, *self.encoder_layers]
            self.encoder = CNNEncoder(
                *dims,
                dim_output_embedding=self.encoder_embedding_dim,
                kernel_size=self.encoder_kernel_size,
                key=k2,
            )

        self.dim_embedding = self.subject_embedding_dim + self.encoder_embedding_dim

    def _subject_indices_from_workout_ids(self, workout_ids) -> jax.Array:
        workout_ids_np = jax.device_get(workout_ids)
        indices = [self.workout_id_to_embedding_index[w_id] for w_id in workout_ids_np]
        return jnp.asarray(indices, dtype=jnp.int32)

    def get_embeddings_from_workout_ids(
        self,
        workout_ids,
        history=None,
        history_lengths=None,
        subject_indices: Optional[jax.Array] = None,
    ):
        embeddings = []
        if self.subject_embeddings is not None:
            if subject_indices is None:
                subject_indices = self._subject_indices_from_workout_ids(workout_ids)
            subject_embeddings = self.subject_embeddings[subject_indices]
            embeddings.append(subject_embeddings)
        if self.encoder is not None:
            encoded_embeddings = self.encoder(history, history_lengths)
            embeddings.append(encoded_embeddings)

        embeddings = jnp.concatenate(embeddings, axis=-1)
        return embeddings


def get_activation(name):
    return {"softplus": jax.nn.softplus, "tanh": jnp.tanh, "relu": jax.nn.relu}[name]


class BatchState(eqx.Module):
    intensity: jax.Array
    ode_params: Dict[str, jax.Array]
    fatigue_coefficient: jax.Array
    weather_coefficient: jax.Array
    activity_coefficient: jax.Array
    workout_embeddings: jax.Array
    initial_heart_rate_and_demand: jax.Array


class ODEModel(eqx.Module):
    config: OdeConfig = eqx.field(static=True)
    embedding_store: EmbeddingStore
    ode_parameter_functions: Dict[str, PersonalizedScalarNN]
    fatigue_fn: PersonalizedScalarNN
    weather_fn: Optional[PersonalizedScalarNN]
    activity_fn: PersonalizedScalarNN
    initial_heart_rate_activity_fn: DenseNN
    dim_activity: int = eqx.field(static=True)
    dim_embedding: int = eqx.field(static=True)

    def __init__(
        self,
        workouts_info: pd.DataFrame,
        config: OdeConfig,
        key: jax.Array,
    ):
        self.config = config
        key = jax.random.PRNGKey(config.seed) if key is None else key

        self.dim_activity = self.config.data_config.n_activity_channels()
        k_embed, k_params, k_fatigue, k_weather, k_activity, k_init = jax.random.split(
            key, 6
        )

        self.embedding_store = EmbeddingStore(self.config, workouts_info, key=k_embed)
        self.dim_embedding = self.embedding_store.dim_embedding
        ode_parameter_layers = [
            int(d) for d in self.config.ode_parameter_layers.split(",")
        ]

        parameter_ranges = list(
            map(
                float,
                filter(
                    len,
                    self.config.ranges_A_B_alpha_beta.replace(",", " ").split(" "),
                ),
            )
        )

        ode_param_keys = jax.random.split(k_params, 6)
        self.ode_parameter_functions = {}
        for param_idx, (parameter_name, low, high) in enumerate(
            [
                ("A", *parameter_ranges[0:2]),
                ("B", *parameter_ranges[2:4]),
                ("alpha", *parameter_ranges[4:6]),
                ("beta", *parameter_ranges[6:8]),
                ("hr_min", 40.0, 90.0),
                ("hr_max", 140.0, 210.0),
            ]
        ):
            self.ode_parameter_functions[parameter_name] = PersonalizedScalarNN(
                self.dim_embedding,
                *ode_parameter_layers,
                personalization="none",
                output_bounds=(low, high),
                activation=get_activation(self.config.ode_parameter_activation),
                key=ode_param_keys[param_idx],
            )

        fatigue_layers = [1, 32, 16]
        self.fatigue_fn = PersonalizedScalarNN(
            *fatigue_layers,
            personalization=self.config.fatigue_fn_embedding_personalization,
            dim_personalization=self.dim_embedding,
            output_bounds=(0.5, 1.5),
            key=k_fatigue,
        )

        weather_layers = [self.config.data_config.n_weather_channels(), 32, 16]
        if weather_layers[0] == 0:
            self.weather_fn = None
        else:
            self.weather_fn = PersonalizedScalarNN(
                *weather_layers,
                personalization=self.config.weather_fn_embedding_personalization,
                dim_personalization=self.dim_embedding,
                output_bounds=(0.5, 1.5),
                key=k_weather,
            )

        activity_fn_layers = [self.dim_activity] + [
            int(d) for d in self.config.activity_fn_layers.split(",")
        ]
        min_activity_fn, max_activity_fn = map(
            float, self.config.range_activity_fn.split(" ")
        )
        self.activity_fn = PersonalizedScalarNN(
            *activity_fn_layers,
            personalization=self.config.activity_fn_embedding_personalization,
            dim_personalization=self.dim_embedding,
            output_bounds=(min_activity_fn, max_activity_fn),
            activation=get_activation(self.config.activity_fn_activation),
            key=k_activity,
        )

        self.initial_heart_rate_activity_fn = DenseNN(
            *[self.dim_embedding, 32, 2],
            output_bounds=(50.0, 200.0),
            key=k_init,
        )

    def vector_field(self, t, x, args: BatchState):
        hr = x[..., 0]
        demand = x[..., 1]

        idx = jnp.clip(jnp.asarray(t, dtype=jnp.int32), 0, args.intensity.shape[1] - 1)
        intensity = args.intensity[:, idx]

        f_min = (
            (jnp.abs(hr - args.ode_params["hr_min"] + EPSILON) + EPSILON) / 60.0
        ) ** args.ode_params["alpha"]
        f_max = (
            (jnp.abs(args.ode_params["hr_max"] - hr + EPSILON) + EPSILON) / 60.0
        ) ** args.ode_params["beta"]

        hr_dot = jnp.where(
            hr >= args.ode_params["hr_max"],
            args.ode_params["hr_max"] - hr,
            jnp.where(
                hr <= args.ode_params["hr_min"],
                args.ode_params["hr_min"] - hr,
                jnp.exp(args.ode_params["A"]) * f_min * f_max * (demand - hr) / 60.0,
            ),
        )

        demand_dot = (intensity - demand) / 60.0 * jnp.exp(args.ode_params["B"])

        return jnp.stack([hr_dot, demand_dot], axis=-1)

    def initialize_batch(
        self,
        activity,
        times,
        workout_id,
        subject_id,
        subject_index=None,
        history=None,
        history_length=None,
        weather=None,
    ) -> BatchState:
        workout_embeddings = self.embedding_store.get_embeddings_from_workout_ids(
            workout_id,
            history,
            history_length,
            subject_indices=subject_index,
        )
        ode_params = {
            k: self.ode_parameter_functions[k](workout_embeddings)
            for k in self.ode_parameter_functions
        }
        workout_embeddings_tiled = jnp.tile(
            workout_embeddings[:, None, :], (1, activity.shape[1], 1)
        )

        fatigue_coefficient = self.fatigue_fn(
            times[..., None], workout_embeddings_tiled
        )
        if self.weather_fn is not None:
            weather_coefficient = self.weather_fn(weather, workout_embeddings)
        else:
            weather_coefficient = 1.0

        activity_coefficient = self.activity_fn(activity, workout_embeddings_tiled)

        intensity = activity_coefficient * fatigue_coefficient * weather_coefficient
        initial_heart_rate_and_demand = self.initial_heart_rate_activity_fn(
            workout_embeddings
        )

        return BatchState(
            intensity=intensity,
            ode_params=ode_params,
            fatigue_coefficient=fatigue_coefficient,
            weather_coefficient=weather_coefficient,
            activity_coefficient=activity_coefficient,
            workout_embeddings=workout_embeddings,
            initial_heart_rate_and_demand=initial_heart_rate_and_demand,
        )

    def forecast_batch(
        self,
        activity,
        times,
        workout_id,
        subject_id,
        subject_index=None,
        history=None,
        history_length=None,
        weather=None,
        step_size=1.0,
    ):
        state = self.initialize_batch(
            activity,
            times,
            workout_id,
            subject_id,
            subject_index,
            history,
            history_length,
            weather,
        )

        y0 = state.initial_heart_rate_and_demand
        seq_len = times.shape[1]
        ts = jnp.arange(0, seq_len + 1)

        term = diffrax.ODETerm(self.vector_field)
        sol = diffrax.diffeqsolve(
            term,
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=float(seq_len),
            dt0=step_size,
            y0=y0,
            args=state,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.ConstantStepSize(),
        )

        xs = sol.ys

        res = {
            "heart_rate": xs[1:, :, 0].transpose(1, 0),
            "demand": xs[1:, :, 1].transpose(1, 0),
            "intensity": state.intensity,
            "fatigue_coefficient": state.fatigue_coefficient,
            "weather_coefficient": state.weather_coefficient,
            "activity_coefficient": state.activity_coefficient,
            "ode_params": state.ode_params,
            "initial_heart_rate_and_demand": state.initial_heart_rate_and_demand,
            "workout_embedding": state.workout_embeddings,
        }
        return res

    def forecast_single_workout(
        self,
        workout,
        step_size=1.0,
        convert_to_numpy=True,
    ):
        from hjartax.data import workout_dataset_collate_fn

        workout = workout_dataset_collate_fn([workout])

        res = self.forecast_batch(
            workout["activity"],
            workout["time"],
            workout["workout_id"],
            workout["subject_id"],
            workout.get("subject_index"),
            workout["history"],
            workout["history_length"],
            workout["weather"],
            step_size=step_size,
        )
        if convert_to_numpy:
            res = _to_numpy_result(res)
        else:
            res = _squeeze_result(res)
        return res


def _squeeze_result(res):
    squeezed = {}
    for k, v in res.items():
        if isinstance(v, dict):
            squeezed[k] = {k2: v2[0] for k2, v2 in v.items()}
        elif hasattr(v, "shape"):
            if v.shape == ():
                squeezed[k] = v
            else:
                squeezed[k] = v[0]
        else:
            squeezed[k] = v
    return squeezed


def _to_numpy_result(res):
    squeezed = _squeeze_result(res)
    out = {}
    for k, v in squeezed.items():
        if isinstance(v, dict):
            out[k] = {k2: jax.device_get(v2) for k2, v2 in v.items()}
        else:
            out[k] = jax.device_get(v)
    return out
