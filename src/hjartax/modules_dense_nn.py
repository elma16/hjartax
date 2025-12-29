#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

PersonalizationType = Literal["none", "softmax", "concatenate"]


class DenseNN(eqx.Module):
    layers: list[eqx.nn.Linear]
    activation: Callable = eqx.field(static=True)
    dim_context: int = eqx.field(static=True)
    output_bounds: Optional[tuple[float, float]] = eqx.field(static=True)

    def __init__(
        self,
        *dim_layers: int,
        activation: Callable = jax.nn.softplus,
        dim_context: int = 0,
        bias: bool = True,
        output_bounds: Optional[tuple[float, float]] = None,
        key: jax.Array,
    ):
        if len(dim_layers) < 2:
            raise ValueError("DenseNN requires at least input and output dimensions")
        dims = list(dim_layers)
        dims[0] += dim_context

        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], use_bias=bias, key=keys[i])
            for i in range(len(dims) - 1)
        ]
        self.activation = activation
        self.dim_context = dim_context
        self.output_bounds = output_bounds

    def __call__(self, x: jax.Array, *context: jax.Array) -> jax.Array:
        if self.dim_context:
            x = jnp.concatenate([x, *context], axis=-1)

        def apply_linear(layer: eqx.nn.Linear, inputs: jax.Array) -> jax.Array:
            outputs = jnp.matmul(inputs, layer.weight.T)
            if layer.bias is not None:
                outputs = outputs + layer.bias
            return outputs

        h = x
        for layer in self.layers[:-1]:
            h = self.activation(apply_linear(layer, h))
        h = apply_linear(self.layers[-1], h)

        if self.output_bounds is not None:
            low, high = self.output_bounds
            h = jax.nn.sigmoid(h) * (high - low) + low

        return h


class PersonalizedScalarNN(eqx.Module):
    dense_nn: DenseNN
    personalization: PersonalizationType = eqx.field(static=True)

    def __init__(
        self,
        *dim_layers: int,
        personalization: PersonalizationType,
        dim_personalization: int = 0,
        output_bounds: Optional[tuple[float, float]] = None,
        activation: Callable = jax.nn.softplus,
        bias: bool = True,
        key: jax.Array,
    ):
        self.personalization = personalization
        if personalization == "none":
            dim_personalization = 0
            self.dense_nn = DenseNN(
                *dim_layers,
                1,
                dim_context=dim_personalization,
                output_bounds=output_bounds,
                activation=activation,
                bias=bias,
                key=key,
            )
        elif personalization == "softmax":
            self.dense_nn = DenseNN(
                *dim_layers,
                dim_personalization,
                dim_context=0,
                output_bounds=output_bounds,
                activation=activation,
                bias=bias,
                key=key,
            )
        elif personalization == "concatenate":
            self.dense_nn = DenseNN(
                *dim_layers,
                1,
                dim_context=dim_personalization,
                output_bounds=output_bounds,
                activation=activation,
                bias=bias,
                key=key,
            )
        else:
            raise ValueError(f"Unknown personalization {personalization}")

    def __call__(self, x: jax.Array, context: Optional[jax.Array] = None) -> jax.Array:
        if self.personalization == "none":
            return jnp.squeeze(self.dense_nn(x), axis=-1)
        if context is None:
            raise ValueError(
                f"Context is required for personalization '{self.personalization}'"
            )
        if self.personalization == "softmax":
            h = self.dense_nn(x)
            weights = jax.nn.softmax(context, axis=-1)
            return jnp.sum(h * weights, axis=-1)
        if self.personalization == "concatenate":
            return jnp.squeeze(self.dense_nn(x, context), axis=-1)
        raise ValueError(f"Unknown personalization {self.personalization}")
