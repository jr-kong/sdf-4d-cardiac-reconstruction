from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from sdf_ae_data import load_global_sdf_sample


CONRAD_DATA_PARAMS = (
    "nb_cycles",
    "time_per_cycle",
    "shapes_per_cycle",
    "cycle_shift",
    "time_shift",
    "frequency",
    "EF_Vol",
    "EF_Biplane",
)


def video_id_from_global_file(path: Path) -> str:
    stem = Path(path).stem
    return stem.replace("_global_sdf", "")


def split_name_from_global_file(path: Path) -> str:
    return Path(path).parent.name.upper()


def split_name_to_mesh_dir(split_name: str) -> str:
    split_name = str(split_name).upper()
    mapping = {
        "TRAIN": "train",
        "VALIDATION": "val",
        "TEST": "test",
    }
    return mapping.get(split_name, split_name.lower())


def resolve_mesh_npz_path(
    global_sdf_path: Path,
    mesh_generated_root: Optional[Path] = None,
) -> Path:
    global_sdf_path = Path(global_sdf_path)
    split = split_name_to_mesh_dir(split_name_from_global_file(global_sdf_path))
    video_id = video_id_from_global_file(global_sdf_path)

    if mesh_generated_root is not None:
        candidate = Path(mesh_generated_root) / split / f"{video_id}.npz"
        if candidate.exists():
            return candidate

    with np.load(global_sdf_path, allow_pickle=False) as data:
        raw_sdf_path = Path(str(data["source_file"]))

    if not raw_sdf_path.exists():
        raise FileNotFoundError(
            f"Global SDF file points to missing raw SDF file: {raw_sdf_path}"
        )

    with np.load(raw_sdf_path, allow_pickle=True) as raw_sdf_data:
        source_mesh_path = Path(str(raw_sdf_data["source_npz"]))

    if source_mesh_path.exists():
        return source_mesh_path

    if mesh_generated_root is not None:
        fallback = Path(mesh_generated_root) / split / source_mesh_path.name
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        "Could not resolve the source mesh-video NPZ for "
        f"{global_sdf_path}. Missing original path {source_mesh_path}."
    )


def load_ef_label(
    global_sdf_path: Path,
    ef_key: str = "EF_Biplane",
    mesh_generated_root: Optional[Path] = None,
    normalize_to_unit: bool = True,
) -> float:
    if ef_key not in CONRAD_DATA_PARAMS:
        raise KeyError(f"Unsupported EF key: {ef_key}")

    mesh_npz_path = resolve_mesh_npz_path(
        global_sdf_path=global_sdf_path,
        mesh_generated_root=mesh_generated_root,
    )
    with np.load(mesh_npz_path, allow_pickle=True) as mesh_npz:
        params = np.asarray(mesh_npz["params"], dtype=np.float32)
    ef_idx = CONRAD_DATA_PARAMS.index(ef_key)
    label = float(params[ef_idx])
    if normalize_to_unit:
        label /= 100.0
    return label


def encode_sdf_files_to_latents(
    encoder: tf.keras.Model,
    files: Sequence[Path],
    target_frames: int,
    mesh_generated_root: Optional[Path] = None,
    ef_key: str = "EF_Biplane",
    normalize_ef_to_unit: bool = True,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    files = [Path(path) for path in files]

    all_latents: List[np.ndarray] = []
    all_labels: List[float] = []
    all_metadata: List[Dict[str, str]] = []

    pending_x: List[np.ndarray] = []
    pending_meta: List[Dict[str, str]] = []

    def flush_pending():
        nonlocal pending_x, pending_meta
        if not pending_x:
            return
        batch_x = np.stack(pending_x, axis=0).astype(np.float32)
        batch_latents = encoder(batch_x, training=False)
        batch_latents = np.asarray(batch_latents, dtype=np.float32)
        for latent, meta in zip(batch_latents, pending_meta):
            all_latents.append(latent)
            all_labels.append(meta["ef"])
            all_metadata.append(
                {
                    "global_sdf_file": meta["global_sdf_file"],
                    "mesh_npz_file": meta["mesh_npz_file"],
                    "video_id": meta["video_id"],
                    "split": meta["split"],
                }
            )
        pending_x = []
        pending_meta = []

    for file_path in files:
        x, _ = load_global_sdf_sample(file_path, target_frames=target_frames)
        mesh_npz_path = resolve_mesh_npz_path(file_path, mesh_generated_root=mesh_generated_root)
        ef = load_ef_label(
            file_path,
            ef_key=ef_key,
            mesh_generated_root=mesh_generated_root,
            normalize_to_unit=normalize_ef_to_unit,
        )
        pending_x.append(x.astype(np.float32))
        pending_meta.append(
            {
                "global_sdf_file": str(file_path),
                "mesh_npz_file": str(mesh_npz_path),
                "video_id": video_id_from_global_file(file_path),
                "split": split_name_from_global_file(file_path),
                "ef": ef,
            }
        )
        if len(pending_x) >= batch_size:
            flush_pending()

    flush_pending()
    X = np.stack(all_latents, axis=0).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.float32).reshape(-1, 1)
    return X, y, all_metadata


def summarize_latent_split(
    latents: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Dict[str, str]],
) -> Dict[str, object]:
    return {
        "count": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "ef_min": float(labels.min()),
        "ef_max": float(labels.max()),
        "ef_mean": float(labels.mean()),
        "first_video": metadata[0]["video_id"] if metadata else None,
        "last_video": metadata[-1]["video_id"] if metadata else None,
    }
