#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sdf_ae_data import discover_split_files
from sdf_ae_model import build_sdf_video_autoencoder
from sdf_ef_data import encode_sdf_files_to_latents, summarize_latent_split
from sdf_ef_model import build_sdf_ef_predictor, compile_sdf_ef_predictor


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def fit_standardizer(X_train: np.ndarray):
    mean = np.asarray(X_train.mean(axis=0), dtype=np.float32)
    std = np.asarray(X_train.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float32) - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train an SDF EF predictor on frozen SDF AE latents.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/svu/e0957732/4DHeartModel/sdf_outputs/SDF_GLOBAL_FULL"),
        help="Root folder containing TRAIN/VALIDATION/TEST global SDF NPZ files.",
    )
    parser.add_argument(
        "--ae-run-dir",
        type=Path,
        required=True,
        help="SDF AE run directory containing config.json and best.weights.h5.",
    )
    parser.add_argument(
        "--mesh-generated-root",
        type=Path,
        default=Path("/home/svu/e0957732/4DHeartModel/experiments/CONRADData_DHB/generated/rkrpth"),
        help="Root folder of generated mesh-video NPZs used to recover EF labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save checkpoints and metrics. Defaults to Tools/SDF_AE/ef_runs/<timestamp>.",
    )
    parser.add_argument("--ef-key", type=str, default="EF_Biplane", choices=["EF_Biplane", "EF_Vol"])
    parser.add_argument("--target-frames", type=int, default=None, help="Override AE target frames if needed.")
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "l1", "l2"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--predictor-architecture", type=str, default="mlp", choices=["mlp", "mesh_like"])
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--standardize-latents", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    with (args.ae_run_dir / "config.json").open() as f:
        ae_cfg = json.load(f)
    ae_weights = args.ae_run_dir / "best.weights.h5"
    if not ae_weights.exists():
        raise FileNotFoundError(f"Missing AE checkpoint: {ae_weights}")

    target_frames = args.target_frames if args.target_frames is not None else int(ae_cfg["target_frames"])

    effective_learning_rate = args.learning_rate
    effective_loss = args.loss
    effective_patience = args.patience
    if args.predictor_architecture == "mesh_like":
        if args.learning_rate == 1e-3:
            effective_learning_rate = 1e-4
        if args.loss == "mse":
            effective_loss = "l1"
        if args.patience == 8:
            effective_patience = 20

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = THIS_DIR / "ef_runs" / timestamp
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ae_model = build_sdf_video_autoencoder(
        target_frames=target_frames,
        grid_size=int(ae_cfg["grid_size"]),
        channels=int(ae_cfg["channels"]),
        base_filters=int(ae_cfg["base_filters"]),
        frame_embedding_dim=int(ae_cfg["frame_embedding_dim"]),
        video_latent_dim=int(ae_cfg["video_latent_dim"]),
    )
    ae_model.load_weights(str(ae_weights))
    encoder = tf.keras.Model(
        inputs=ae_model.input,
        outputs=ae_model.get_layer("video_latent").output,
        name="sdf_video_encoder",
    )

    train_files = discover_split_files(args.data_root, "TRAIN", limit=args.train_limit)
    val_files = discover_split_files(args.data_root, "VALIDATION", limit=args.val_limit)
    test_files = discover_split_files(args.data_root, "TEST", limit=args.test_limit)

    if not train_files or not val_files:
        raise ValueError("Need at least TRAIN and VALIDATION files to train the SDF EF predictor.")

    print(
        "Discovered global SDF files: "
        f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        ,
        flush=True,
    )
    print(
        "Using AE checkpoint: "
        f"{ae_weights} | target_frames={target_frames} | ef_key={args.ef_key}"
        ,
        flush=True,
    )
    print(
        "Predictor settings: "
        f"architecture={args.predictor_architecture}, loss={effective_loss}, "
        f"learning_rate={effective_learning_rate}, patience={effective_patience}",
        flush=True,
    )

    print("Encoding TRAIN latents from global SDF videos...", flush=True)
    X_train, y_train, meta_train = encode_sdf_files_to_latents(
        encoder,
        train_files,
        target_frames=target_frames,
        mesh_generated_root=args.mesh_generated_root,
        ef_key=args.ef_key,
        batch_size=args.encode_batch_size,
    )
    print(f"Encoded TRAIN latents: shape={X_train.shape}, labels={y_train.shape}", flush=True)

    print("Encoding VALIDATION latents from global SDF videos...", flush=True)
    X_val, y_val, meta_val = encode_sdf_files_to_latents(
        encoder,
        val_files,
        target_frames=target_frames,
        mesh_generated_root=args.mesh_generated_root,
        ef_key=args.ef_key,
        batch_size=args.encode_batch_size,
    )
    print(f"Encoded VALIDATION latents: shape={X_val.shape}, labels={y_val.shape}", flush=True)

    X_test = y_test = meta_test = None
    if test_files:
        print("Encoding TEST latents from global SDF videos...", flush=True)
        X_test, y_test, meta_test = encode_sdf_files_to_latents(
            encoder,
            test_files,
            target_frames=target_frames,
            mesh_generated_root=args.mesh_generated_root,
            ef_key=args.ef_key,
            batch_size=args.encode_batch_size,
        )
        print(f"Encoded TEST latents: shape={X_test.shape}, labels={y_test.shape}", flush=True)

    latent_standardizer = None
    if args.standardize_latents:
        print("Standardizing latent vectors using TRAIN split statistics...", flush=True)
        latent_mean, latent_std = fit_standardizer(X_train)
        X_train = apply_standardizer(X_train, latent_mean, latent_std)
        X_val = apply_standardizer(X_val, latent_mean, latent_std)
        if X_test is not None:
            X_test = apply_standardizer(X_test, latent_mean, latent_std)
        latent_standardizer = {
            "enabled": True,
            "feature_dim": int(latent_mean.shape[0]),
            "mean_min": float(latent_mean.min()),
            "mean_max": float(latent_mean.max()),
            "std_min": float(latent_std.min()),
            "std_max": float(latent_std.max()),
        }
        print(
            "Latent standardization applied: "
            f"feature_dim={latent_standardizer['feature_dim']} "
            f"std_range=[{latent_standardizer['std_min']:.6f}, {latent_standardizer['std_max']:.6f}]",
            flush=True,
        )

    np.savez_compressed(output_dir / "train_latents.npz", X=X_train, y=y_train)
    np.savez_compressed(output_dir / "val_latents.npz", X=X_val, y=y_val)
    if X_test is not None:
        np.savez_compressed(output_dir / "test_latents.npz", X=X_test, y=y_test)
    print(f"Saved latent caches to {output_dir}", flush=True)

    predictor = build_sdf_ef_predictor(
        input_dim=int(X_train.shape[1]),
        architecture=args.predictor_architecture,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        final_activation="sigmoid",
    )
    compile_sdf_ef_predictor(
        predictor,
        learning_rate=effective_learning_rate,
        loss=effective_loss,
    )

    config = {
        "data_root": str(args.data_root),
        "ae_run_dir": str(args.ae_run_dir),
        "mesh_generated_root": str(args.mesh_generated_root),
        "output_dir": str(output_dir),
        "ef_key": args.ef_key,
        "target_frames": target_frames,
        "encode_batch_size": args.encode_batch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "predictor_architecture": args.predictor_architecture,
        "loss": effective_loss,
        "learning_rate": effective_learning_rate,
        "patience": effective_patience,
        "standardize_latents": bool(args.standardize_latents),
        "latent_standardizer": latent_standardizer,
        "dropout": args.dropout,
        "hidden_dims": args.hidden_dims,
        "train_summary": summarize_latent_split(X_train, y_train, meta_train),
        "val_summary": summarize_latent_split(X_val, y_val, meta_val),
        "test_summary": summarize_latent_split(X_test, y_test, meta_test) if X_test is not None else None,
    }
    save_json(output_dir / "config.json", config)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "history.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=effective_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(
        "Starting SDF EF predictor training with "
        f"train_size={X_train.shape[0]}, val_size={X_val.shape[0]}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}"
        ,
        flush=True,
    )
    history = predictor.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    save_json(output_dir / "history.json", history.history)

    best_weights_path = output_dir / "best.weights.h5"
    if best_weights_path.exists():
        predictor.load_weights(str(best_weights_path))
        print(f"Reloaded best checkpoint from {best_weights_path} for final evaluation")

    val_pred = predictor.predict(X_val, batch_size=args.batch_size, verbose=0)
    val_metrics = regression_metrics(y_val, val_pred)

    metrics = {
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]) if X_test is not None else 0,
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "best_val_loss_from_history": float(np.min(history.history["val_loss"])),
        "val_metrics_best_checkpoint": val_metrics,
    }

    np.savez_compressed(
        output_dir / "val_predictions.npz",
        y_true=y_val.astype(np.float32),
        y_pred=np.asarray(val_pred, dtype=np.float32),
    )

    if X_test is not None:
        test_pred = predictor.predict(X_test, batch_size=args.batch_size, verbose=0)
        test_metrics = regression_metrics(y_test, test_pred)
        metrics["test_metrics_best_checkpoint"] = test_metrics
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            y_true=y_test.astype(np.float32),
            y_pred=np.asarray(test_pred, dtype=np.float32),
        )

    save_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved SDF EF predictor outputs to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
