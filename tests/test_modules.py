import pytest
import jax
import jax.numpy as jnp

from hjartax.modules_cnn import (
    CausalCNN,
    CausalConvolutionBlock,
    CNNEncoder,
    Conv1d,
    WeightNormConv1d,
)
from hjartax.modules_dense_nn import DenseNN, PersonalizedScalarNN


def test_dense_nn_output_bounds():
    key = jax.random.PRNGKey(0)
    model = DenseNN(4, 3, 2, output_bounds=(0.5, 1.5), key=key)
    x = jax.random.normal(key, (5, 4))
    y = model(x)
    assert y.shape == (5, 2)
    assert jnp.all(y >= 0.5)
    assert jnp.all(y <= 1.5)


def test_dense_nn_requires_dims():
    with pytest.raises(ValueError):
        DenseNN(3, key=jax.random.PRNGKey(0))


def test_dense_nn_with_context():
    key = jax.random.PRNGKey(3)
    model = DenseNN(2, 4, 1, dim_context=1, key=key)
    x = jnp.ones((4, 2))
    ctx = jnp.zeros((4, 1))
    y = model(x, ctx)
    assert y.shape == (4, 1)


def test_personalized_scalar_nn_softmax():
    key = jax.random.PRNGKey(1)
    model = PersonalizedScalarNN(
        3,
        4,
        personalization="softmax",
        dim_personalization=5,
        key=key,
    )
    x = jax.random.normal(key, (7, 3))
    context = jax.random.normal(key, (7, 5))
    y = model(x, context)
    assert y.shape == (7,)


def test_personalized_scalar_nn_none():
    key = jax.random.PRNGKey(4)
    model = PersonalizedScalarNN(2, 3, personalization="none", key=key)
    x = jax.random.normal(key, (5, 2))
    y = model(x)
    assert y.shape == (5,)


def test_personalized_scalar_nn_concatenate_requires_context():
    key = jax.random.PRNGKey(5)
    model = PersonalizedScalarNN(
        2,
        3,
        personalization="concatenate",
        dim_personalization=2,
        key=key,
    )
    x = jnp.ones((3, 2))
    with pytest.raises(ValueError):
        _ = model(x)


def test_causal_cnn_shape():
    key = jax.random.PRNGKey(2)
    cnn = CausalCNN(3, 4, 5, kernel_size=2, key=key)
    x = jnp.ones((2, 3, 10))
    y = cnn(x)
    assert y.shape == (2, 5, 10)


def test_conv1d_shape():
    key = jax.random.PRNGKey(6)
    conv = Conv1d(3, 4, kernel_size=3, key=key)
    x = jnp.ones((2, 3, 8))
    y = conv(x)
    assert y.shape == (2, 4, 6)


def test_weightnorm_conv1d_preserves_length():
    key = jax.random.PRNGKey(7)
    conv = WeightNormConv1d(2, 3, kernel_size=3, dilation=1, key=key)
    x = jnp.ones((1, 2, 9))
    y = conv(x)
    assert y.shape == (1, 3, 9)


def test_causal_convolution_block_shape():
    key = jax.random.PRNGKey(8)
    block = CausalConvolutionBlock(2, 4, kernel_size=3, dilation=1, key=key)
    x = jnp.ones((2, 2, 7))
    y = block(x)
    assert y.shape == (2, 4, 7)


def test_cnn_encoder_masking_matches_full_length():
    key = jax.random.PRNGKey(9)
    encoder = CNNEncoder(3, 4, dim_output_embedding=5, kernel_size=2, key=key)
    x = jnp.ones((2, 6, 3))
    lengths = jnp.array([6, 6])
    y_masked = encoder(x, lengths)
    y_full = encoder(x)
    assert y_masked.shape == (2, 5)
    assert jnp.allclose(y_masked, y_full)
