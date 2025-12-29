#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


def _conv1d(
    x: jax.Array,
    weight: jax.Array,
    bias: Optional[jax.Array] = None,
    stride: int = 1,
    dilation: int = 1,
    padding: tuple[int, int] = (0, 0),
) -> jax.Array:
    x = jnp.pad(x, ((0, 0), (0, 0), padding))
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(stride,),
        padding="VALID",
        rhs_dilation=(dilation,),
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    if bias is not None:
        y = y + bias[None, :, None]
    return y


class Conv1d(eqx.Module):
    weight: jax.Array
    bias: Optional[jax.Array]
    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    padding: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: tuple[int, int] = (0, 0),
        bias: bool = True,
        key: jax.Array = None,
    ):
        if key is None:
            raise ValueError("Conv1d requires a PRNG key")
        k1, k2 = jax.random.split(key, 2)
        scale = 1.0 / jnp.sqrt(in_channels * kernel_size)
        self.weight = jax.random.normal(k1, (out_channels, in_channels, kernel_size))
        self.weight = self.weight * scale
        self.bias = jnp.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return _conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
        )


class WeightNormConv1d(eqx.Module):
    v: jax.Array
    g: jax.Array
    bias: Optional[jax.Array]
    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        key: jax.Array = None,
    ):
        if key is None:
            raise ValueError("WeightNormConv1d requires a PRNG key")
        k1, _ = jax.random.split(key, 2)
        scale = 1.0 / jnp.sqrt(in_channels * kernel_size)
        self.v = jax.random.normal(k1, (out_channels, in_channels, kernel_size)) * scale
        g = jnp.linalg.norm(self.v, axis=(1, 2), keepdims=True)
        self.g = g
        self.bias = jnp.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def __call__(self, x: jax.Array) -> jax.Array:
        norm = jnp.linalg.norm(self.v, axis=(1, 2), keepdims=True)
        weight = self.v * (self.g / (norm + 1e-8))
        padding = (self.kernel_size - 1) * self.dilation
        return _conv1d(
            x,
            weight,
            self.bias,
            stride=self.stride,
            dilation=self.dilation,
            padding=(padding, 0),
        )


class CausalConvolutionBlock(eqx.Module):
    conv1: WeightNormConv1d
    conv2: WeightNormConv1d
    upordownsample: Optional[Conv1d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        batch_norm: bool = False,
        key: jax.Array = None,
    ):
        if batch_norm:
            raise ValueError("batch_norm is not supported in the JAX port")
        if key is None:
            raise ValueError("CausalConvolutionBlock requires a PRNG key")
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = WeightNormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            key=k1,
        )
        self.conv2 = WeightNormConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            key=k2,
        )
        if in_channels != out_channels:
            self.upordownsample = Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                key=k3,
            )
        else:
            self.upordownsample = None

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jax.nn.leaky_relu(self.conv1(x))
        h = jax.nn.leaky_relu(self.conv2(h))
        res = x if self.upordownsample is None else self.upordownsample(x)
        return h + res


class CausalCNN(eqx.Module):
    blocks: list[CausalConvolutionBlock]

    def __init__(self, *channels: int, kernel_size: int = 3, key: jax.Array = None):
        if key is None:
            raise ValueError("CausalCNN requires a PRNG key")
        keys = jax.random.split(key, len(channels) - 1)
        dilation_size = 1
        blocks = []
        for idx, (c_in, c_out) in enumerate(zip(channels, channels[1:])):
            blocks.append(
                CausalConvolutionBlock(
                    c_in,
                    c_out,
                    kernel_size,
                    dilation_size,
                    key=keys[idx],
                )
            )
            dilation_size *= 2
        self.blocks = blocks

    def __call__(self, x: jax.Array) -> jax.Array:
        h = x
        for block in self.blocks:
            h = block(h)
        return h


class CNNEncoder(eqx.Module):
    causal_cnn: CausalCNN
    linear: eqx.nn.Linear

    def __init__(
        self,
        *channels_cnn: int,
        dim_output_embedding: int = 32,
        kernel_size: int = 6,
        key: jax.Array = None,
    ):
        if key is None:
            raise ValueError("CNNEncoder requires a PRNG key")
        k1, k2 = jax.random.split(key, 2)
        self.causal_cnn = CausalCNN(*channels_cnn, kernel_size=kernel_size, key=k1)
        self.linear = eqx.nn.Linear(channels_cnn[-1], dim_output_embedding, key=k2)

    def __call__(self, x: jax.Array, length: Optional[jax.Array] = None) -> jax.Array:
        x = jnp.swapaxes(x, -1, -2)
        h = self.causal_cnn(x)
        if length is not None:
            length = length.astype(jnp.float32)
            mask = jnp.arange(h.shape[-1])[None, None, :] < length[:, None, None]
            h = h * mask
            h = jnp.sum(h, axis=-1) / length[:, None]
        else:
            h = jnp.mean(h, axis=-1)
        return jnp.matmul(h, self.linear.weight.T) + self.linear.bias
