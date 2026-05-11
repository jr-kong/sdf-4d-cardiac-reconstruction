from typing import Sequence

import tensorflow as tf
from tensorflow.keras import layers as KL


def build_sdf_ef_predictor(
    input_dim: int,
    architecture: str = "mlp",
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.0,
    final_activation: str = "sigmoid",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="sdf_latent")
    if architecture == "mesh_like":
        x = KL.Dense(16, use_bias=True, name="dense_0")(inputs)
        x = KL.PReLU(name="dense_0_prelu")(x)
        outputs = KL.Dense(1, activation="relu", use_bias=True, name="ef_prediction")(x)
    elif architecture == "mlp":
        x = inputs
        for units in hidden_dims:
            x = KL.Dense(int(units), activation="relu")(x)
            if dropout and dropout > 0.0:
                x = KL.Dropout(float(dropout))(x)
        outputs = KL.Dense(1, activation=final_activation, name="ef_prediction")(x)
    else:
        raise ValueError(f"Unsupported SDF EF predictor architecture: {architecture}")
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="sdf_ef_predictor")


def compile_sdf_ef_predictor(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    loss: str = "mse",
) -> tf.keras.Model:
    loss_name = str(loss).lower()
    if loss_name == "l1":
        loss_name = "mae"
    elif loss_name == "l2":
        loss_name = "mse"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_name,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model
