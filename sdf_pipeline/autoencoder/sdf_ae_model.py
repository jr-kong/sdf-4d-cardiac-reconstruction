from typing import Callable, Dict

import tensorflow as tf
from tensorflow.keras import layers as KL


def masked_mse_loss(num_channels: int, invalid_weight: float = 0.05) -> Callable:
    invalid_weight = float(invalid_weight)

    def loss_fn(y_true_packed, y_pred):
        y_true = y_true_packed[..., :num_channels]
        mask = y_true_packed[..., num_channels:]
        weights = invalid_weight + (1.0 - invalid_weight) * mask
        sq_err = tf.square(y_pred - y_true)
        weighted_err = sq_err * weights
        denom = tf.reduce_sum(weights) + 1e-6
        return tf.reduce_sum(weighted_err) / denom

    return loss_fn


def valid_region_mse(num_channels: int) -> Callable:
    def metric_fn(y_true_packed, y_pred):
        y_true = y_true_packed[..., :num_channels]
        mask = y_true_packed[..., num_channels:]
        sq_err = tf.square(y_pred - y_true) * mask
        denom = tf.reduce_sum(mask) + 1e-6
        return tf.reduce_sum(sq_err) / denom

    metric_fn.__name__ = "valid_region_mse"
    return metric_fn


def full_region_mse(num_channels: int) -> Callable:
    def metric_fn(y_true_packed, y_pred):
        y_true = y_true_packed[..., :num_channels]
        return tf.reduce_mean(tf.square(y_pred - y_true))

    metric_fn.__name__ = "full_region_mse"
    return metric_fn


def build_sdf_video_autoencoder(
    target_frames: int = 24,
    grid_size: int = 32,
    channels: int = 5,
    base_filters: int = 16,
    frame_embedding_dim: int = 128,
    video_latent_dim: int = 128,
) -> tf.keras.Model:
    if grid_size != 32:
        raise ValueError("This baseline decoder currently expects grid_size=32.")

    inputs = tf.keras.Input(
        shape=(target_frames, grid_size, grid_size, grid_size, channels),
        name="sdf_video",
    )

    x = inputs
    for filters in (base_filters, base_filters * 2, base_filters * 4):
        x = KL.TimeDistributed(
            KL.Conv3D(filters, kernel_size=3, strides=2, padding="same")
        )(x)
        x = KL.TimeDistributed(KL.BatchNormalization())(x)
        x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(KL.GlobalAveragePooling3D())(x)
    x = KL.TimeDistributed(KL.Dense(frame_embedding_dim, activation="relu"))(x)

    x = KL.Bidirectional(
        KL.LSTM(video_latent_dim // 2, return_sequences=False),
        name="temporal_encoder",
    )(x)
    latent = KL.Dense(video_latent_dim, activation="relu", name="video_latent")(x)

    x = KL.RepeatVector(target_frames)(latent)
    x = KL.LSTM(video_latent_dim, return_sequences=True, name="temporal_decoder")(x)
    x = KL.TimeDistributed(KL.Dense(4 * 4 * 4 * (base_filters * 4), activation="relu"))(x)
    x = KL.TimeDistributed(KL.Reshape((4, 4, 4, base_filters * 4)))(x)

    for filters in (base_filters * 4, base_filters * 2, base_filters):
        x = KL.TimeDistributed(
            KL.Conv3DTranspose(filters, kernel_size=3, strides=2, padding="same")
        )(x)
        x = KL.TimeDistributed(KL.BatchNormalization())(x)
        x = KL.Activation("relu")(x)

    outputs = KL.TimeDistributed(
        KL.Conv3D(channels, kernel_size=3, padding="same", activation="tanh"),
        name="reconstruction",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="sdf_video_autoencoder")


def compile_sdf_video_autoencoder(
    model: tf.keras.Model,
    num_channels: int = 5,
    learning_rate: float = 1e-3,
    invalid_weight: float = 0.05,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_mse_loss(num_channels=num_channels, invalid_weight=invalid_weight),
        metrics=[
            valid_region_mse(num_channels=num_channels),
            full_region_mse(num_channels=num_channels),
        ],
    )
    return model
