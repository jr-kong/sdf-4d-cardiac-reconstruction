from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sdf_ae_data import discover_split_files
from sdf_ae_model import build_sdf_video_autoencoder
from sdf_ef_model import build_sdf_ef_predictor
from sdf_ef_data import encode_sdf_files_to_latents, video_id_from_global_file


LOGGER = logging.getLogger(__name__)


def _format_elapsed(seconds: float) -> str:
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_sdf_ae_model_from_run(
    ae_run_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[tf.keras.Model, Dict[str, object]]:
    logger = logger or LOGGER
    ae_run_dir = Path(ae_run_dir)
    config_path = ae_run_dir / "config.json"
    weights_path = ae_run_dir / "best.weights.h5"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing SDF AE config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing SDF AE checkpoint: {weights_path}")

    with config_path.open() as f:
        ae_cfg = json.load(f)

    ae_model = build_sdf_video_autoencoder(
        target_frames=int(ae_cfg["target_frames"]),
        grid_size=int(ae_cfg["grid_size"]),
        channels=int(ae_cfg["channels"]),
        base_filters=int(ae_cfg["base_filters"]),
        frame_embedding_dim=int(ae_cfg["frame_embedding_dim"]),
        video_latent_dim=int(ae_cfg["video_latent_dim"]),
    )
    ae_model.load_weights(str(weights_path))
    logger.info("Loaded SDF AE model from %s", weights_path)
    return ae_model, ae_cfg


def load_sdf_ae_encoder_from_run(
    ae_run_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[tf.keras.Model, Dict[str, object]]:
    logger = logger or LOGGER
    ae_model, ae_cfg = load_sdf_ae_model_from_run(ae_run_dir=ae_run_dir, logger=logger)
    encoder = tf.keras.Model(
        inputs=ae_model.input,
        outputs=ae_model.get_layer("video_latent").output,
        name="sdf_video_encoder",
    )
    logger.info("Built SDF AE encoder view from run %s", ae_run_dir)
    return encoder, ae_cfg


def load_sdf_ef_predictor_from_run(
    ef_run_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[tf.keras.Model, Dict[str, object]]:
    logger = logger or LOGGER
    ef_run_dir = Path(ef_run_dir)
    config_path = ef_run_dir / "config.json"
    weights_path = ef_run_dir / "best.weights.h5"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing SDF EF config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing SDF EF checkpoint: {weights_path}")

    with config_path.open() as f:
        ef_cfg = json.load(f)

    latent_dim = None
    train_summary = ef_cfg.get("train_summary")
    if isinstance(train_summary, dict):
        latent_dim = train_summary.get("latent_dim")
    if latent_dim is None:
        train_latents_path = ef_run_dir / "train_latents.npz"
        if train_latents_path.exists():
            with np.load(train_latents_path, allow_pickle=False) as data:
                latent_dim = int(data["X"].shape[1])
    if latent_dim is None:
        raise ValueError(f"Could not infer SDF EF predictor input_dim from {ef_run_dir}")

    predictor = build_sdf_ef_predictor(
        input_dim=int(latent_dim),
        architecture=str(ef_cfg.get("predictor_architecture", "mlp")),
        hidden_dims=ef_cfg.get("hidden_dims", [128, 64]),
        dropout=float(ef_cfg.get("dropout", 0.0)),
        final_activation="sigmoid",
    )
    predictor.load_weights(str(weights_path))
    logger.info("Loaded SDF EF predictor from %s", weights_path)
    return predictor, ef_cfg


def _load_cached_encodings(
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    with np.load(cache_path, allow_pickle=False) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        efs = np.asarray(data["efs"], dtype=np.float32).reshape(-1, 1)
        filenames = [str(x) for x in data["filenames"]]
        source_files = [str(x) for x in data["source_files"]]
        mesh_npz_files = [str(x) for x in data["mesh_npz_files"]]
    return latents, efs, filenames, source_files, mesh_npz_files


def get_encoded_sdf_dataset(
    filepaths: Sequence[Path],
    sdf_encoder: tf.keras.Model,
    logger: Optional[logging.Logger] = None,
    cache_path: Optional[Path] = None,
    mesh_generated_root: Optional[Path] = None,
    ef_key: str = "EF_Biplane",
    target_frames: int = 24,
    encode_batch_size: int = 8,
    force_recompute: bool = False,
    return_filenames: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[str]]:
    logger = logger or LOGGER
    t_start = time.time()

    requested_files = [Path(path) for path in filepaths]
    requested_filenames = [video_id_from_global_file(path) for path in requested_files]
    filename_data: Dict[str, Tuple[np.ndarray, np.ndarray, str, str]] = {}

    files_to_encode = requested_files

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            logger.info("Loading cached SDF encodings from %s", cache_path)
            latents, efs, filenames, source_files, mesh_npz_files = _load_cached_encodings(cache_path)
            cached_lookup = {name: idx for idx, name in enumerate(filenames)}

            missing_indices: List[int] = []
            for i, filename in enumerate(requested_filenames):
                if filename in cached_lookup:
                    idx = cached_lookup[filename]
                    filename_data[filename] = (
                        latents[idx : idx + 1],
                        efs[idx : idx + 1],
                        source_files[idx],
                        mesh_npz_files[idx],
                    )
                else:
                    missing_indices.append(i)

            if missing_indices:
                files_to_encode = [requested_files[i] for i in missing_indices]
                logger.info(
                    "Found %d/%d cached SDF encodings. Encoding %d missing files.",
                    len(requested_filenames) - len(missing_indices),
                    len(requested_filenames),
                    len(missing_indices),
                )
            else:
                files_to_encode = []
                logger.info("Found all requested SDF encodings in cache.")

    if files_to_encode:
        logger.info("Encoding %d SDF files into frozen AE latents...", len(files_to_encode))
        latents_new, efs_new, metadata = encode_sdf_files_to_latents(
            encoder=sdf_encoder,
            files=files_to_encode,
            target_frames=target_frames,
            mesh_generated_root=mesh_generated_root,
            ef_key=ef_key,
            batch_size=encode_batch_size,
        )
        for latent, ef, meta in zip(latents_new, efs_new, metadata):
            filename = meta["video_id"]
            filename_data[filename] = (
                latent[None, ...].astype(np.float32),
                np.asarray(ef, dtype=np.float32).reshape(1, 1),
                str(meta["global_sdf_file"]),
                str(meta["mesh_npz_file"]),
            )

    missing_filenames = [filename for filename in requested_filenames if filename not in filename_data]
    if missing_filenames:
        raise KeyError(
            "Some requested SDF files are still missing after cache lookup and encoding: "
            f"{missing_filenames[:5]}"
        )

    latents = np.concatenate([filename_data[f][0] for f in requested_filenames], axis=0).astype(np.float32)
    efs = np.concatenate([filename_data[f][1] for f in requested_filenames], axis=0).astype(np.float32)
    source_files = [filename_data[f][2] for f in requested_filenames]
    mesh_npz_files = [filename_data[f][3] for f in requested_filenames]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            latents=latents,
            efs=efs,
            filenames=np.asarray(requested_filenames),
            source_files=np.asarray(source_files),
            mesh_npz_files=np.asarray(mesh_npz_files),
        )
        logger.info("Saved encoded SDF cache to %s", cache_path)

    elapsed = _format_elapsed(time.time() - t_start)
    logger.info("Encoded SDF dataset in %s", elapsed)
    logger.info("SDF latents shape: %s", latents.shape)
    logger.info("SDF efs shape: %s", efs.shape)

    if return_filenames:
        return latents, efs, requested_filenames
    return latents, efs


def get_encoded_sdf_split_dataset(
    data_root: Path,
    split: str,
    ae_run_dir: Path,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[Path] = None,
    mesh_generated_root: Optional[Path] = None,
    ef_key: str = "EF_Biplane",
    target_frames: Optional[int] = None,
    encode_batch_size: int = 8,
    limit: Optional[int] = None,
    force_recompute: bool = False,
    return_filenames: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[str]]:
    logger = logger or LOGGER
    encoder, ae_cfg = load_sdf_ae_encoder_from_run(ae_run_dir, logger=logger)
    effective_target_frames = (
        int(target_frames) if target_frames is not None else int(ae_cfg["target_frames"])
    )
    files = discover_split_files(data_root, split, limit=limit)
    if not files:
        raise ValueError(f"No {split} files found under {data_root}")

    cache_path = None
    if output_dir is not None:
        cache_path = Path(output_dir) / f"{split.lower()}.npz"

    return get_encoded_sdf_dataset(
        filepaths=files,
        sdf_encoder=encoder,
        logger=logger,
        cache_path=cache_path,
        mesh_generated_root=mesh_generated_root,
        ef_key=ef_key,
        target_frames=effective_target_frames,
        encode_batch_size=encode_batch_size,
        force_recompute=force_recompute,
        return_filenames=return_filenames,
    )


def get_encoded_sdf_splits(
    data_root: Path,
    splits: Iterable[str],
    ae_run_dir: Path,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[Path] = None,
    mesh_generated_root: Optional[Path] = None,
    ef_key: str = "EF_Biplane",
    target_frames: Optional[int] = None,
    encode_batch_size: int = 8,
    split_limits: Optional[Dict[str, int]] = None,
    force_recompute: bool = False,
    return_filenames: bool = False,
):
    logger = logger or LOGGER
    encoder, ae_cfg = load_sdf_ae_encoder_from_run(ae_run_dir, logger=logger)
    effective_target_frames = (
        int(target_frames) if target_frames is not None else int(ae_cfg["target_frames"])
    )

    encoded: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    filenames_out: Dict[str, List[str]] = {}
    split_limits = split_limits or {}

    for split in splits:
        files = discover_split_files(data_root, split, limit=split_limits.get(split))
        if not files:
            raise ValueError(f"No {split} files found under {data_root}")

        cache_path = None
        if output_dir is not None:
            cache_path = Path(output_dir) / f"{split.lower()}.npz"

        result = get_encoded_sdf_dataset(
            filepaths=files,
            sdf_encoder=encoder,
            logger=logger,
            cache_path=cache_path,
            mesh_generated_root=mesh_generated_root,
            ef_key=ef_key,
            target_frames=effective_target_frames,
            encode_batch_size=encode_batch_size,
            force_recompute=force_recompute,
            return_filenames=return_filenames,
        )
        if return_filenames:
            latents, efs, filenames = result
            filenames_out[split] = filenames
        else:
            latents, efs = result
        encoded[split] = (latents, efs)

    if return_filenames:
        return encoded, filenames_out
    return encoded
