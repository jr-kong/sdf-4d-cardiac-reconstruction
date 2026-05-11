import json
import logging
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

from source.constants import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + "." + __name__)


def _format_elapsed(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return h, m, s


class DenseStack(Model):
    """
    Config-driven dense stack used by the SDF CycleGAN generators and discriminators.

    Unlike the mesh CycleGAN, this stack always operates on the full latent vector.
    The current SDF latent does not expose explicit frequency / phase slots.
    """

    def __init__(self, dense_layers, name="dense_stack", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_layers = []
        self.batch_norms = []
        self.activations = []

        for dense_layer in dense_layers:
            batch_norm = dense_layer.get("batch_norm", False)
            dense_layer_new = {}
            for key, value in dense_layer.items():
                if key != "batch_norm":
                    dense_layer_new[key] = value
            dense_layer = dense_layer_new

            act_cfg = dense_layer["activation"]
            act_name = act_cfg["name"].lower()
            if act_name == "leakyrelu":
                activation = LeakyReLU(alpha=act_cfg.get("alpha", 0.2))
            elif act_name == "prelu":
                activation = PReLU()
            else:
                activation = tf.keras.activations.get(act_cfg["name"])

            dense_layer["activation"] = "linear"
            dense = Dense(**dense_layer)
            self.dense_layers.append(dense)
            self.batch_norms.append(BatchNormalization() if batch_norm else None)
            self.activations.append(activation)

    def call(self, inputs, training=False):
        h = inputs
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](h)
            if self.batch_norms[i] is not None:
                h = self.batch_norms[i](h, training=training)
            h = self.activations[i](h)
        return h


class EchoToSDFGen(Model):
    def __init__(self, dense_layers, latent_space_sdf_dim, name="Gen_G_SDF", **kwargs):
        super().__init__(name=name, **kwargs)
        assert dense_layers[-1]["units"] == latent_space_sdf_dim
        self.stack = DenseStack(dense_layers=dense_layers, name="echo_to_sdf_stack")

    def call(self, inputs, training=False):
        return self.stack(inputs, training=training)


class SDFToEchoGen(Model):
    def __init__(self, dense_layers, latent_space_echo_dim, name="Gen_F_SDF", **kwargs):
        super().__init__(name=name, **kwargs)
        assert dense_layers[-1]["units"] == latent_space_echo_dim
        self.stack = DenseStack(dense_layers=dense_layers, name="sdf_to_echo_stack")

    def call(self, inputs, training=False):
        return self.stack(inputs, training=training)


class LatentDisc(Model):
    def __init__(self, dense_layers, name="LatentDisc", **kwargs):
        super().__init__(name=name, **kwargs)
        assert dense_layers[-1]["units"] == 1
        self.stack = DenseStack(dense_layers=dense_layers, name=f"{name}_stack")

    def call(self, inputs, training=False):
        return self.stack(inputs, training=training)


class CycleGanSDF(Model):
    """
    SDF-compatible latent CycleGAN.

    This mirrors the existing mesh CycleGAN at the latent-translation level while
    avoiding assumptions that the target latent has explicit frequency / phase slots.

    Expected collaborators:
    - echo_ae: frozen echo autoencoder / encoder provider
    - sdf_ae: frozen SDF autoencoder / decoder provider
    - sdf_ef_pred: frozen EF predictor operating on SDF latents
    """

    def __init__(
        self,
        echo_ae,
        sdf_ae,
        sdf_ef_pred,
        log_dir,
        training_params,
        model_params,
        save_metrics=False,
        name="cycle_gan_sdf",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.log_dir = Path(log_dir)
        self.training_params = training_params
        self.model_params = model_params

        self.echo_ae = echo_ae
        self.sdf_ae = sdf_ae
        self.sdf_ef_pred = sdf_ef_pred

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.save_metrics = save_metrics

        self.build_model(model_params)
        if save_metrics:
            self.create_metrics_writers()
        logger.info("SDF CycleGAN scaffold initialized.")

    def build_model(self, model_params):
        latent_space_echo_dim = int(model_params.get("latent_space_echo_dim", 128))
        latent_space_sdf_dim = int(model_params.get("latent_space_sdf_dim", 128))

        self.sdf_gen = EchoToSDFGen(
            dense_layers=model_params["sdf_gen"]["dense_layers"],
            latent_space_sdf_dim=latent_space_sdf_dim,
        )
        self.echo_gen = SDFToEchoGen(
            dense_layers=model_params["echo_gen"]["dense_layers"],
            latent_space_echo_dim=latent_space_echo_dim,
        )
        self.sdf_disc = LatentDisc(
            dense_layers=model_params["sdf_disc"]["dense_layers"],
            name="Disc_SDFLatent",
        )
        self.echo_disc = LatentDisc(
            dense_layers=model_params["echo_disc"]["dense_layers"],
            name="Disc_EchoLatent_SDF",
        )

        fake_sdf_latent = self.sdf_gen(tf.zeros(shape=(1, latent_space_echo_dim)))
        fake_echo_latent = self.echo_gen(tf.zeros(shape=(1, latent_space_sdf_dim)))
        self.sdf_disc(fake_sdf_latent)
        self.echo_disc(fake_echo_latent)

    def build_optimizers(self):
        optimizers_params = self.training_params["optimizers"]
        self.sdf_gen_optimizer = Adam(**optimizers_params["sdf_gen"])
        self.echo_gen_optimizer = Adam(**optimizers_params["echo_gen"])
        self.sdf_disc_optimizer = Adam(**optimizers_params["sdf_disc"])
        self.echo_disc_optimizer = Adam(**optimizers_params["echo_disc"])

    def create_metrics_writers(self):
        train_metrics = [
            "cycle_loss_lambda",
            "ef_loss_lambda",
            "sdf_disc_loss",
            "sdf_disc_real_loss",
            "sdf_disc_fake_loss",
            "echo_disc_loss",
            "echo_disc_real_loss",
            "echo_disc_fake_loss",
            "sdf_gen_loss",
            "echo_gen_loss",
            "sdf_gen_total_loss",
            "echo_gen_total_loss",
            "echo_cycle_loss",
            "sdf_cycle_loss",
            "ef_loss",
            "ef_loss_echo_to_sdf",
            "ef_loss_sdf_to_sdf",
        ]
        val_metrics = ["ef_loss", "ef_loss_echo_to_sdf", "ef_loss_sdf_to_sdf"]

        self._train_metrics = {name: tf.keras.metrics.Mean(name=name) for name in train_metrics}
        self._val_metrics = {name: tf.keras.metrics.Mean(name=name) for name in val_metrics}

        train_log_dir = self.log_dir / "metrics" / "train"
        val_log_dir = self.log_dir / "metrics" / "validation"
        self._train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
        self._val_summary_writer = tf.summary.create_file_writer(str(val_log_dir))

    @staticmethod
    def _log_metrics(losses, metrics):
        for key, value in losses.items():
            if key in metrics:
                metrics[key](value)

    def reset_metrics(self, train=False, val=False):
        if train and hasattr(self, "_train_metrics"):
            for metric in self._train_metrics.values():
                metric.reset_states()
        if val and hasattr(self, "_val_metrics"):
            for metric in self._val_metrics.values():
                metric.reset_states()

    @staticmethod
    def write_summaries(summary_writer, metrics, global_step, log_str):
        metrics_str = f"{log_str}:\n"
        with summary_writer.as_default():
            for name, metric in metrics.items():
                result = metric.result()
                tf.summary.scalar(name, result, step=global_step)
                metrics_str += f"{name}: {float(result):.6f}\n"
        logger.info(metrics_str)

    def save_me(self, tag="best"):
        trained_models_dir = self.log_dir / "trained_models"
        trained_models_dir.mkdir(parents=True, exist_ok=True)
        save_path = trained_models_dir / f"cycleGAN_sdf_{tag}"
        self.save_weights(str(save_path))
        logger.info("Saved SDF CycleGAN weights to %s", save_path)

    @tf.function
    def call(self, real_echo_latents, real_sdf_latents, training=False):
        out_echo_to_sdf = self.translate_echo_to_sdf(real_echo_latents, training=training)
        out_sdf_to_echo = self.translate_sdf_to_echo(real_sdf_latents, training=training)
        return out_echo_to_sdf, out_sdf_to_echo

    def translate_echo_to_sdf(self, real_echo_latents, training=False):
        fake_sdf_latents = self.sdf_gen(real_echo_latents, training=training)
        cycled_echo_latents = self.echo_gen(fake_sdf_latents, training=training)
        return fake_sdf_latents, cycled_echo_latents

    def translate_sdf_to_echo(self, real_sdf_latents, training=False):
        fake_echo_latents = self.echo_gen(real_sdf_latents, training=training)
        cycled_sdf_latents = self.sdf_gen(fake_echo_latents, training=training)
        return fake_echo_latents, cycled_sdf_latents

    def decode_sdf_latents(self, sdf_latents, times, row_lengths, training=False):
        """
        Decode translated SDF latents back into SDF videos.

        This delegates to the frozen SDF AE decoder and keeps the output in the
        SDF domain, rather than trying to convert immediately to mesh features.
        """

        return self.sdf_ae.decode(sdf_latents, times, row_lengths, training=training)

    def mse(self, y_true, y_pred):
        return tf.reduce_mean(losses.mean_squared_error(y_true, y_pred))

    def mae(self, y_true, y_pred):
        return tf.reduce_mean(losses.mean_absolute_error(y_true, y_pred))

    def _discriminator_loss(self, real_logits, generated_logits):
        real_loss = self.loss_obj(tf.ones_like(real_logits), real_logits)
        generated_loss = self.loss_obj(tf.zeros_like(generated_logits), generated_logits)

        true_vals = tf.concat([tf.ones_like(real_logits), tf.zeros_like(generated_logits)], axis=0)
        pred_vals = tf.concat([real_logits, generated_logits], axis=0)
        all_data_loss = self.loss_obj(true_vals, pred_vals)
        return all_data_loss, real_loss, generated_loss

    def _generator_loss(self, generated_logits):
        return self.loss_obj(tf.ones_like(generated_logits), generated_logits)

    def _cycle_loss(self, real, cycled):
        cycle_loss_str = self.training_params["cycle_loss_str"]
        if cycle_loss_str == "l1":
            return tf.reduce_mean(losses.mean_absolute_error(real, cycled))
        if cycle_loss_str == "l2":
            return tf.reduce_mean(losses.mean_squared_error(real, cycled))
        raise NotImplementedError(f"Unsupported cycle loss: {cycle_loss_str}")

    def _ef_loss(self, true_echo_efs, true_sdf_efs, fake_sdf_latents, cycled_sdf_latents):
        true_echo_efs = tf.where(tf.reduce_max(true_echo_efs) > 1.5, true_echo_efs / 100.0, true_echo_efs)
        true_sdf_efs = tf.where(tf.reduce_max(true_sdf_efs) > 1.5, true_sdf_efs / 100.0, true_sdf_efs)
        fake_sdf_pred_efs = tf.math.abs(self.sdf_ef_pred(fake_sdf_latents))
        cycled_sdf_pred_efs = tf.math.abs(self.sdf_ef_pred(cycled_sdf_latents))

        ef_loss_str = self.training_params["ef_loss_str"]
        if ef_loss_str == "l1":
            ef_loss_echo_to_sdf = self.mae(true_echo_efs, fake_sdf_pred_efs)
            ef_loss_sdf_to_sdf = self.mae(true_sdf_efs, cycled_sdf_pred_efs)
        elif ef_loss_str == "l2":
            ef_loss_echo_to_sdf = self.mse(true_echo_efs, fake_sdf_pred_efs)
            ef_loss_sdf_to_sdf = self.mse(true_sdf_efs, cycled_sdf_pred_efs)
        else:
            raise NotImplementedError(f"Unsupported EF loss: {ef_loss_str}")

        return ef_loss_echo_to_sdf, ef_loss_sdf_to_sdf

    @tf.function
    def train_step_latents(self, real_echo_latents, real_sdf_latents, true_echo_efs, true_sdf_efs):
        cycle_loss_lambda = tf.constant(float(self.training_params["cycle_loss_lambda"]), dtype=tf.float32)
        ef_loss_lambda = tf.constant(float(self.training_params["ef_loss_lambda"]), dtype=tf.float32)

        sdf_gen_vars = self.sdf_gen.trainable_variables
        echo_gen_vars = self.echo_gen.trainable_variables
        sdf_disc_vars = self.sdf_disc.trainable_variables
        echo_disc_vars = self.echo_disc.trainable_variables

        with tf.GradientTape(watch_accessed_variables=False) as sdf_gen_tape, tf.GradientTape(
            watch_accessed_variables=False
        ) as echo_gen_tape, tf.GradientTape(watch_accessed_variables=False) as sdf_disc_tape, tf.GradientTape(
            watch_accessed_variables=False
        ) as echo_disc_tape:
            sdf_gen_tape.watch(sdf_gen_vars)
            echo_gen_tape.watch(echo_gen_vars)
            sdf_disc_tape.watch(sdf_disc_vars)
            echo_disc_tape.watch(echo_disc_vars)

            (fake_sdf_latents, cycled_echo_latents), (fake_echo_latents, cycled_sdf_latents) = self.call(
                real_echo_latents, real_sdf_latents, training=True
            )

            real_sdf_logits = self.sdf_disc(real_sdf_latents, training=True)
            fake_sdf_logits = self.sdf_disc(fake_sdf_latents, training=True)
            real_echo_logits = self.echo_disc(real_echo_latents, training=True)
            fake_echo_logits = self.echo_disc(fake_echo_latents, training=True)

            sdf_disc_loss, sdf_disc_real_loss, sdf_disc_fake_loss = self._discriminator_loss(
                real_sdf_logits, fake_sdf_logits
            )
            echo_disc_loss, echo_disc_real_loss, echo_disc_fake_loss = self._discriminator_loss(
                real_echo_logits, fake_echo_logits
            )
            sdf_gen_loss = self._generator_loss(fake_sdf_logits)
            echo_gen_loss = self._generator_loss(fake_echo_logits)
            echo_cycle_loss = self._cycle_loss(real_echo_latents, cycled_echo_latents)
            sdf_cycle_loss = self._cycle_loss(real_sdf_latents, cycled_sdf_latents)
            total_cycle_loss = cycle_loss_lambda * (echo_cycle_loss + sdf_cycle_loss)

            ef_loss_echo_to_sdf, ef_loss_sdf_to_sdf = self._ef_loss(
                true_echo_efs, true_sdf_efs, fake_sdf_latents, cycled_sdf_latents
            )

            sdf_gen_total_loss = sdf_gen_loss + total_cycle_loss + ef_loss_lambda * ef_loss_echo_to_sdf
            echo_gen_total_loss = echo_gen_loss + total_cycle_loss + ef_loss_lambda * ef_loss_sdf_to_sdf

        sdf_gen_grads = sdf_gen_tape.gradient(sdf_gen_total_loss, sdf_gen_vars)
        echo_gen_grads = echo_gen_tape.gradient(echo_gen_total_loss, echo_gen_vars)
        sdf_disc_grads = sdf_disc_tape.gradient(sdf_disc_loss, sdf_disc_vars)
        echo_disc_grads = echo_disc_tape.gradient(echo_disc_loss, echo_disc_vars)

        self.sdf_gen_optimizer.apply_gradients(zip(sdf_gen_grads, sdf_gen_vars))
        self.echo_gen_optimizer.apply_gradients(zip(echo_gen_grads, echo_gen_vars))
        self.sdf_disc_optimizer.apply_gradients(zip(sdf_disc_grads, sdf_disc_vars))
        self.echo_disc_optimizer.apply_gradients(zip(echo_disc_grads, echo_disc_vars))

        return {
            "sdf_disc_loss": sdf_disc_loss,
            "sdf_disc_real_loss": sdf_disc_real_loss,
            "sdf_disc_fake_loss": sdf_disc_fake_loss,
            "echo_disc_loss": echo_disc_loss,
            "echo_disc_real_loss": echo_disc_real_loss,
            "echo_disc_fake_loss": echo_disc_fake_loss,
            "sdf_gen_loss": sdf_gen_loss,
            "echo_gen_loss": echo_gen_loss,
            "sdf_gen_total_loss": sdf_gen_total_loss,
            "echo_gen_total_loss": echo_gen_total_loss,
            "echo_cycle_loss": echo_cycle_loss,
            "sdf_cycle_loss": sdf_cycle_loss,
            "ef_loss": ef_loss_echo_to_sdf,
            "ef_loss_echo_to_sdf": ef_loss_echo_to_sdf,
            "ef_loss_sdf_to_sdf": ef_loss_sdf_to_sdf,
        }

    @tf.function
    def val_step_latents(self, real_echo_latents, real_sdf_latents, true_echo_efs, true_sdf_efs):
        (fake_sdf_latents, _), (_, cycled_sdf_latents) = self.call(real_echo_latents, real_sdf_latents, training=False)
        ef_loss_echo_to_sdf, ef_loss_sdf_to_sdf = self._ef_loss(
            true_echo_efs, true_sdf_efs, fake_sdf_latents, cycled_sdf_latents
        )
        return {
            "ef_loss": ef_loss_echo_to_sdf,
            "ef_loss_echo_to_sdf": ef_loss_echo_to_sdf,
            "ef_loss_sdf_to_sdf": ef_loss_sdf_to_sdf,
        }

    def fit(self, sdf_datasets_enc, echo_datasets_enc):
        if not hasattr(self, "_train_metrics"):
            self.create_metrics_writers()

        opt_early_stopping_metric = np.inf
        max_patience = self.training_params["max_patience"]
        count = 0
        global_step = 0
        early_stopping = False

        self.build_optimizers()
        logger.info("\nOptimizers params:")
        logger.info(json.dumps(self.training_params["optimizers"], sort_keys=False, indent=4))
        logger.info("\nStart SDF CycleGAN training...\n")

        num_steps_summaries = self.training_params["num_steps_summaries"]
        num_steps_val = self.training_params["num_steps_val"]
        max_steps = self.training_params["max_steps"]
        cycle_loss_lambda = float(self.training_params["cycle_loss_lambda"])
        ef_loss_lambda = float(self.training_params["ef_loss_lambda"])

        sdf_train_enc, sdf_val_enc = sdf_datasets_enc["train"], sdf_datasets_enc["val"]
        echo_train_enc, echo_val_enc = echo_datasets_enc["train"], echo_datasets_enc["val"]

        t1_steps = time.time()
        for echo_batch, sdf_batch in zip(echo_train_enc, sdf_train_enc):
            if global_step == max_steps or early_stopping:
                break

            echo_latents, echo_efs = echo_batch
            sdf_latents, sdf_efs = sdf_batch
            losses = self.train_step_latents(echo_latents, sdf_latents, echo_efs, sdf_efs)
            self._log_metrics(losses, self._train_metrics)

            if global_step % num_steps_summaries == 0:
                self._train_metrics["cycle_loss_lambda"](cycle_loss_lambda)
                self._train_metrics["ef_loss_lambda"](ef_loss_lambda)
                self.write_summaries(self._train_summary_writer, self._train_metrics, global_step, "Train")
                h, m, s = _format_elapsed(time.time() - t1_steps)
                logger.info("%s steps done in %s:%s:%s\n", num_steps_summaries, h, m, s)
                self.reset_metrics(train=True)
                t1_steps = time.time()

            if global_step % num_steps_val == 0:
                logger.info("Computing SDF CycleGAN validation error...")
                t1_val = time.time()
                for echo_val_batch, sdf_val_batch in zip(echo_val_enc, sdf_val_enc):
                    echo_latents_val, echo_efs_val = echo_val_batch
                    sdf_latents_val, sdf_efs_val = sdf_val_batch
                    val_losses = self.val_step_latents(
                        echo_latents_val, sdf_latents_val, echo_efs_val, sdf_efs_val
                    )
                    self._log_metrics(val_losses, self._val_metrics)

                self.write_summaries(self._val_summary_writer, self._val_metrics, global_step, "Validation")
                h, m, s = _format_elapsed(time.time() - t1_val)
                logger.info("Validation done in %s:%s:%s\n", h, m, s)

                early_stopping_loss = float(self._val_metrics["ef_loss"].result())
                if early_stopping_loss >= opt_early_stopping_metric:
                    count += 1
                    logger.info(
                        "Validation EF loss did not improve from %s. Counter: %s",
                        opt_early_stopping_metric,
                        count,
                    )
                    if count == max_patience:
                        logger.info("Early Stopping")
                        early_stopping = True
                else:
                    logger.info("Validation loss (%s) improved, saving model.", early_stopping_loss)
                    opt_early_stopping_metric = early_stopping_loss
                    count = 0
                    self.save_me("best")

                self.save_me("last")
                self.reset_metrics(val=True)

            global_step += 1
