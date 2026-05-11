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

from sdf_ae_data import build_dataset, discover_split_files, load_global_sdf_sample, summarize_files
from sdf_ae_model import build_sdf_video_autoencoder, compile_sdf_video_autoencoder


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_preview_reconstructions(
    model: tf.keras.Model,
    files,
    output_dir: Path,
    target_frames: int,
    limit: int = 3,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in files[:limit]:
        video_id = path.stem.replace("_global_sdf", "")
        x, y = load_global_sdf_sample(path, target_frames=target_frames)
        pred = model.predict(x[None, ...], verbose=0)[0]
        np.savez_compressed(
            output_dir / f"{video_id}_reconstruction_preview.npz",
            input_sdf=x.astype(np.float32),
            recon_sdf=pred.astype(np.float32),
            valid_mask=y[..., x.shape[-1]:].astype(np.float32),
            source_file=str(path),
        )


def main():
    parser = argparse.ArgumentParser(description="Train a minimal SDF video autoencoder baseline.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/svu/e0957732/4DHeartModel/sdf_outputs/SDF_GLOBAL_PILOT_100_20_20"),
        help="Root folder containing TRAIN/VALIDATION/TEST global SDF NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save checkpoints, logs and previews. Defaults to Tools/SDF_AE/runs/<timestamp>.",
    )
    parser.add_argument("--target-frames", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--invalid-weight", type=float, default=0.05)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--frame-embedding-dim", type=int, default=128)
    parser.add_argument("--video-latent-dim", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--preview-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = THIS_DIR / "runs" / timestamp
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = discover_split_files(args.data_root, "TRAIN", limit=args.train_limit)
    val_files = discover_split_files(args.data_root, "VALIDATION", limit=args.val_limit)
    test_files = discover_split_files(args.data_root, "TEST", limit=args.test_limit)

    if not train_files:
        raise ValueError(f"No TRAIN files found under {args.data_root}")
    if not val_files:
        raise ValueError(f"No VALIDATION files found under {args.data_root}")

    train_ds = build_dataset(
        train_files,
        target_frames=args.target_frames,
        grid_size=args.grid_size,
        channels=args.channels,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_ds = build_dataset(
        val_files,
        target_frames=args.target_frames,
        grid_size=args.grid_size,
        channels=args.channels,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = None
    if test_files:
        test_ds = build_dataset(
            test_files,
            target_frames=args.target_frames,
            grid_size=args.grid_size,
            channels=args.channels,
            batch_size=args.batch_size,
            shuffle=False,
        )

    model = build_sdf_video_autoencoder(
        target_frames=args.target_frames,
        grid_size=args.grid_size,
        channels=args.channels,
        base_filters=args.base_filters,
        frame_embedding_dim=args.frame_embedding_dim,
        video_latent_dim=args.video_latent_dim,
    )
    compile_sdf_video_autoencoder(
        model,
        num_channels=args.channels,
        learning_rate=args.learning_rate,
        invalid_weight=args.invalid_weight,
    )

    config = {
        "data_root": str(args.data_root),
        "output_dir": str(output_dir),
        "target_frames": args.target_frames,
        "grid_size": args.grid_size,
        "channels": args.channels,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "invalid_weight": args.invalid_weight,
        "base_filters": args.base_filters,
        "frame_embedding_dim": args.frame_embedding_dim,
        "video_latent_dim": args.video_latent_dim,
        "seed": args.seed,
        "train_files": summarize_files(train_files),
        "val_files": summarize_files(val_files),
        "test_files": summarize_files(test_files),
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
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    save_json(output_dir / "history.json", history.history)

    metrics = {"train_size": len(train_files), "val_size": len(val_files), "test_size": len(test_files)}
    metrics.update({f"final_{k}": float(v[-1]) for k, v in history.history.items()})

    best_weights_path = output_dir / "best.weights.h5"
    if best_weights_path.exists():
        model.load_weights(str(best_weights_path))
        print(f"Reloaded best checkpoint from {best_weights_path} for final evaluation")

    best_epoch = None
    if "val_loss" in history.history and history.history["val_loss"]:
        best_epoch = int(np.argmin(history.history["val_loss"]))
        metrics["best_epoch"] = best_epoch + 1
        metrics["best_val_loss_from_history"] = float(history.history["val_loss"][best_epoch])

    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=1)
    metrics["best_checkpoint_val_metrics"] = {k: float(v) for k, v in val_metrics.items()}

    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
        metrics["best_checkpoint_test_metrics"] = {k: float(v) for k, v in test_metrics.items()}

    save_json(output_dir / "metrics.json", metrics)
    save_preview_reconstructions(
        model=model,
        files=val_files,
        output_dir=output_dir / "reconstruction_previews" / "validation",
        target_frames=args.target_frames,
        limit=args.preview_count,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved run outputs to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
