import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from hjartax.data import WorkoutDataset, WorkoutDatasetConfig, make_dataloaders
from hjartax.ode import OdeConfig, ODEModel
from hjartax.trainer import (
    _batch_for_jax,
    _decoder_l2,
    _encoder_l2,
    _loss_fn,
    evaluate,
    l2_error,
    l2_reg,
    train_ode_model,
)


def _make_df(seq_len=6, n_workouts=2):
    rows = []
    for idx in range(n_workouts):
        time_grid = np.arange(seq_len, dtype=float)
        heart_rate = np.linspace(60, 70, seq_len)
        rows.append(
            {
                "subject_id": 1,
                "workout_id": idx + 1,
                "time_grid": time_grid,
                "time_start": np.datetime64("2020-01-01") + np.timedelta64(idx, "D"),
                "heart_rate": heart_rate,
                "heart_rate_normalized": heart_rate / 200.0,
                "speed": np.linspace(0.0, 1.0, seq_len),
            }
        )
    return pd.DataFrame(rows)


def _make_model(df, encoder_embedding_dim=0):
    data_config = WorkoutDatasetConfig(
        activity_columns=["speed"],
        weather_columns=[],
        history_max_length=None,
        chunk_size=None,
        stride=None,
    )
    ode_config = OdeConfig(
        data_config=data_config,
        encoder_embedding_dim=encoder_embedding_dim,
        subject_embedding_dim=2,
        encoder_layers="8",
        n_epochs=1,
        clip_gradient=0.0,
    )
    model = ODEModel(df, ode_config, key=jax.random.PRNGKey(0))
    return model, data_config


def test_l2_reg_and_l2_error():
    params = {"a": jnp.array([1.0, 2.0])}
    assert float(l2_reg(params)) == 5.0
    assert float(l2_error(jnp.array([2.0]), jnp.array([1.0]), std=1.0)) == 1.0


def test_decoder_and_encoder_l2():
    df = _make_df()
    model, _ = _make_model(df)
    assert _decoder_l2(model) > 0.0
    assert _encoder_l2(model) == 0.0

    model_with_encoder, _ = _make_model(df, encoder_embedding_dim=2)
    assert _encoder_l2(model_with_encoder) > 0.0


def test_batch_for_jax():
    df = _make_df()
    model, data_config = _make_model(df)
    dataset = WorkoutDataset(df, data_config)
    batch = next(iter(make_dataloaders(dataset, dataset, batch_size=1)[0]))
    trimmed = _batch_for_jax(batch)
    assert "workout_id" not in trimmed
    assert "subject_id" not in trimmed
    assert trimmed["activity"].shape[0] == 1


def test_loss_fn_runs():
    df = _make_df()
    model, data_config = _make_model(df)
    dataset = WorkoutDataset(df, data_config)
    batch = next(iter(make_dataloaders(dataset, dataset, batch_size=1)[0]))
    loss = _loss_fn(model, _batch_for_jax(batch), model.config)
    assert jnp.isfinite(loss)


def test_train_ode_model_runs():
    df = _make_df()
    model, data_config = _make_model(df)
    dataset = WorkoutDataset(df, data_config)
    train_loader, test_loader = make_dataloaders(dataset, dataset, batch_size=1)
    trained_model, logs = train_ode_model(model, train_loader, test_loader)
    assert isinstance(trained_model, ODEModel)
    assert len(logs) == 1


def test_evaluate_returns_dataframe():
    df = _make_df()
    model, data_config = _make_model(df)
    dataset = WorkoutDataset(df, data_config)
    _, test_loader = make_dataloaders(dataset, dataset, batch_size=1)
    log = evaluate(model, 0, test_loader, start_time=0.0, train_workout_ids=None)
    assert "l1" in log.columns
    assert "relative" in log.columns
