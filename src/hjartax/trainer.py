#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import collections
import time
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import tqdm

from hjartax.ode import ODEModel

STD_HR = 5
STD_EMBEDDING = 1.0


def l2_reg(params):
    leaves = jax.tree_util.tree_leaves(params)
    if not leaves:
        return 0.0
    return sum([jnp.sum(jnp.square(p)) for p in leaves])


def l2_error(tensor1, tensor2=0.0, std=1.0):
    return jnp.sum(jnp.square((tensor1 - tensor2) / std))


def _decoder_l2(model: ODEModel):
    decoders = [model.fatigue_fn, model.activity_fn]
    if model.weather_fn is not None:
        decoders.append(model.weather_fn)
    params = [eqx.filter(decoder, eqx.is_array) for decoder in decoders]
    return sum([l2_reg(p) for p in params])


def _encoder_l2(model: ODEModel):
    if model.embedding_store.encoder is None:
        return 0.0
    params = eqx.filter(model.embedding_store.encoder, eqx.is_array)
    return l2_reg(params)


def _batch_for_jax(batch):
    return {
        "activity": batch["activity"],
        "time": batch["time"],
        "subject_index": batch["subject_index"],
        "history": batch["history"],
        "history_length": batch["history_length"],
        "weather": batch["weather"],
        "heart_rate": batch["heart_rate"],
    }


def _loss_fn(model: ODEModel, batch, ode_config):
    predictions = model.forecast_batch(
        activity=batch["activity"],
        times=batch["time"],
        workout_id=batch["subject_index"],
        subject_id=batch["subject_index"],
        subject_index=batch["subject_index"],
        history=batch["history"],
        history_length=batch["history_length"],
        weather=batch["weather"],
        step_size=ode_config.ode_step_size,
    )
    predictions_hr = predictions["heart_rate"]
    heart_rate_reconstruction_l2 = l2_error(
        predictions_hr, batch["heart_rate"], std=STD_HR
    )

    embedding_l2 = jnp.sum(jnp.square(predictions["workout_embedding"] / STD_EMBEDDING))
    embedding_l2 *= ode_config.embedding_reg_strength

    decoders_weights_l2 = _decoder_l2(model) * ode_config.decoder_reg_strength
    encoder_weights_l2 = _encoder_l2(model) * ode_config.encoder_reg_strength

    loss = (
        heart_rate_reconstruction_l2
        + embedding_l2
        + decoders_weights_l2
        + encoder_weights_l2
    )
    return loss


def train_ode_model(
    model: ODEModel,
    train_dataloader,
    test_dataloader,
    train_workout_ids: Optional[set] = None,
):
    ode_config = model.config
    if ode_config.clip_gradient and ode_config.clip_gradient > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(ode_config.clip_gradient),
            optax.adam(ode_config.learning_rate),
        )
    else:
        optimizer = optax.adam(ode_config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    evaluation_logs = []

    for epoch in range(ode_config.n_epochs):
        start = time.time()
        epoch_loss = 0.0

        for batch in tqdm.tqdm(train_dataloader):
            train_batch = _batch_for_jax(batch)
            loss, grads = eqx.filter_value_and_grad(_loss_fn)(
                model, train_batch, ode_config
            )
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            epoch_loss += float(loss)

        evaluation_log = evaluate(
            model, epoch, test_dataloader, start, train_workout_ids
        )
        evaluation_logs.append(evaluation_log)

    return model, evaluation_logs


def evaluate(model: ODEModel, epoch, test_dataloader, start_time, train_workout_ids):
    ode_config = model.config
    predicted_hr_all = []
    true_hr_all = []
    embeddings_all = []
    ode_parameters_all = collections.defaultdict(list)
    subject_id_all = []
    workout_id_all = []

    for batch in tqdm.tqdm(test_dataloader):
        predictions = model.forecast_batch(
            batch["activity"],
            batch["time"],
            batch["workout_id"],
            batch["subject_id"],
            batch["subject_index"],
            batch["history"],
            batch["history_length"],
            batch["weather"],
            step_size=ode_config.ode_step_size,
        )
        predictions_hr = jax.device_get(predictions["heart_rate"])
        heart_rate = jax.device_get(batch["heart_rate"])
        for ii in range(len(batch["full_workout_length"])):
            end_index = int(batch["full_workout_length"][ii].item())
            predicted_hr_all.append(predictions_hr[ii, :end_index])
            true_hr_all.append(heart_rate[ii, :end_index])
        embeddings_all.append(predictions["workout_embedding"])
        for k in predictions["ode_params"]:
            ode_parameters_all[k].append(predictions["ode_params"][k])
        subject_id_all.extend(list(batch["subject_id"]))
        workout_id_all.extend(list(batch["workout_id"]))

    embeddings_all = jnp.concatenate(embeddings_all, axis=0)
    for k in ode_parameters_all:
        ode_parameters_all[k] = jnp.concatenate(ode_parameters_all[k], axis=0)

    metrics = {
        "l2": lambda x, y: np.mean((x - y) ** 2) ** 0.5,
        "l1": lambda x, y: np.mean(np.abs(x - y)),
        "relative": lambda pred, truth: np.mean(np.abs(pred - truth) / truth),
    }

    logged_data_for_all_workouts = []
    for idx in range(len(predicted_hr_all)):
        logged_data_for_workout = {}
        for m in metrics:
            logged_data_for_workout[m] = metrics[m](
                predicted_hr_all[idx], true_hr_all[idx]
            )
            logged_data_for_workout[m + "-after2min"] = metrics[m](
                predicted_hr_all[idx][12:], true_hr_all[idx][12:]
            )

        if train_workout_ids is None:
            logged_data_for_workout["in_train"] = True
        else:
            logged_data_for_workout["in_train"] = (
                workout_id_all[idx] in train_workout_ids
            )
        logged_data_for_workout["subject_id"] = subject_id_all[idx]
        logged_data_for_workout["workout_id"] = workout_id_all[idx]
        logged_data_for_workout["subject_embeddings"] = jax.device_get(
            embeddings_all[idx, : model.config.subject_embedding_dim]
        )
        logged_data_for_workout["encoder_embeddings"] = jax.device_get(
            embeddings_all[idx, model.config.subject_embedding_dim :]
        )
        for ode_param_name in ode_parameters_all:
            logged_data_for_workout[ode_param_name] = float(
                jax.device_get(ode_parameters_all[ode_param_name][idx])
            )
        logged_data_for_all_workouts.append(logged_data_for_workout)

    logged_data_for_all_workouts = pd.DataFrame(logged_data_for_all_workouts)
    train_flag = logged_data_for_all_workouts["in_train"]

    print(
        f"Epoch {epoch} took {time.time() - start_time:.1f} seconds",
        "Train mean l1: %.3f bpm (= %.3f %%)"
        % (
            logged_data_for_all_workouts[train_flag]["l1"].mean(),
            logged_data_for_all_workouts[train_flag]["relative"].mean() * 100,
        ),
        "Test mean l1: %.3f bpm (= %.3f %%)"
        % (
            logged_data_for_all_workouts[~train_flag]["l1"].mean(),
            logged_data_for_all_workouts[~train_flag]["relative"].mean() * 100,
        ),
        "Test mean l1-after2min: %.3f bpm (= %.3f %%)"
        % (
            logged_data_for_all_workouts[~train_flag]["l1-after2min"].mean(),
            logged_data_for_all_workouts[~train_flag]["relative-after2min"].mean()
            * 100,
        ),
        sep="\n",
    )
    return logged_data_for_all_workouts
