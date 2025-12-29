import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from hjartax.data import (
    WorkoutDataset,
    WorkoutDatasetConfig,
    workout_dataset_collate_fn,
)
from hjartax.ode import (
    EmbeddingStore,
    OdeConfig,
    ODEModel,
    _squeeze_result,
    _to_numpy_result,
    get_activation,
)


def _make_fake_df(seq_len=8):
    return pd.DataFrame(
        {
            "subject_id": [1],
            "workout_id": [1],
            "time_grid": [np.arange(seq_len, dtype=float)],
            "time_start": [np.datetime64("2020-01-01")],
            "heart_rate": [np.linspace(60, 80, seq_len)],
            "heart_rate_normalized": [np.linspace(0.1, 0.2, seq_len)],
            "speed": [np.linspace(0.0, 1.0, seq_len)],
        }
    )


def test_ode_model_forecast_shapes():
    df = _make_fake_df()
    data_config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, data_config)
    batch = workout_dataset_collate_fn([dataset[0]])

    ode_config = OdeConfig(
        data_config=data_config,
        encoder_embedding_dim=0,
        subject_embedding_dim=2,
        encoder_layers="8",
    )

    model = ODEModel(df, ode_config, key=jax.random.PRNGKey(0))
    res = model.forecast_batch(
        batch["activity"],
        batch["time"],
        batch["workout_id"],
        batch["subject_id"],
        batch["subject_index"],
        batch["history"],
        batch["history_length"],
        batch["weather"],
    )

    assert res["heart_rate"].shape == (1, 8)
    assert res["demand"].shape == (1, 8)
    assert res["intensity"].shape == (1, 8)
    assert res["workout_embedding"].shape == (1, 2)


def test_get_activation_softplus():
    f = get_activation("softplus")
    x = jnp.array([-1.0, 0.0, 1.0])
    assert jnp.allclose(f(x), jax.nn.softplus(x))


def test_embedding_store_subject_embeddings_shape():
    df = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "workout_id": [11, 22],
        }
    )
    data_config = WorkoutDatasetConfig(activity_columns=["speed"])
    ode_config = OdeConfig(
        data_config=data_config,
        encoder_embedding_dim=0,
        subject_embedding_dim=3,
        encoder_layers="8",
    )
    store = EmbeddingStore(ode_config, df, key=jax.random.PRNGKey(0))
    subject_indices = jnp.array([0, 1])
    embeddings = store.get_embeddings_from_workout_ids(
        workout_ids=jnp.array([11, 22]),
        history=None,
        history_lengths=None,
        subject_indices=subject_indices,
    )
    assert embeddings.shape == (2, 3)


def test_vector_field_shape():
    df = _make_fake_df()
    data_config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, data_config)
    batch = workout_dataset_collate_fn([dataset[0]])
    ode_config = OdeConfig(
        data_config=data_config,
        encoder_embedding_dim=0,
        subject_embedding_dim=2,
        encoder_layers="8",
    )
    model = ODEModel(df, ode_config, key=jax.random.PRNGKey(0))
    state = model.initialize_batch(
        batch["activity"],
        batch["time"],
        batch["workout_id"],
        batch["subject_id"],
        batch["subject_index"],
        batch["history"],
        batch["history_length"],
        batch["weather"],
    )
    y0 = state.initial_heart_rate_and_demand
    dy = model.vector_field(0.0, y0, state)
    assert dy.shape == y0.shape


def test_forecast_single_workout_numpy():
    df = _make_fake_df()
    data_config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    dataset = WorkoutDataset(df, data_config)
    ode_config = OdeConfig(
        data_config=data_config,
        encoder_embedding_dim=0,
        subject_embedding_dim=2,
        encoder_layers="8",
    )
    model = ODEModel(df, ode_config, key=jax.random.PRNGKey(0))
    res = model.forecast_single_workout(dataset[0], convert_to_numpy=True)
    assert isinstance(res["heart_rate"], np.ndarray)
    assert res["heart_rate"].shape == (8,)


def test_result_helpers():
    res = {
        "heart_rate": jnp.ones((1, 3)),
        "ode_params": {"A": jnp.ones((1,))},
    }
    squeezed = _squeeze_result(res)
    assert squeezed["heart_rate"].shape == (3,)
    assert squeezed["ode_params"]["A"].shape == ()
    as_numpy = _to_numpy_result(res)
    assert isinstance(as_numpy["heart_rate"], np.ndarray)
