from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


def discover_split_files(root: Path, split: str, limit: Optional[int] = None) -> List[Path]:
    split_root = Path(root) / split
    files = sorted(split_root.glob("*_global_sdf.npz"))
    if limit is not None and limit > 0:
        files = files[:limit]
    return files


def uniform_frame_indices(n_frames: int, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be a positive integer.")
    if n_frames == target_frames:
        return np.arange(n_frames, dtype=np.int64)
    if n_frames == 1:
        return np.zeros((target_frames,), dtype=np.int64)
    return np.round(np.linspace(0, n_frames - 1, target_frames)).astype(np.int64)


def load_global_sdf_sample(npz_path: Path, target_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        sdf = np.asarray(data["sdf_tensor"], dtype=np.float32)
        mask = np.asarray(data["component_valid_mask"], dtype=np.float32)

    frame_indices = uniform_frame_indices(sdf.shape[0], target_frames)
    sdf = sdf[frame_indices]
    mask = mask[frame_indices]

    # Convert from [T, C, D, H, W] to [T, D, H, W, C] for channels-last 3D convs.
    sdf = np.transpose(sdf, (0, 2, 3, 4, 1))
    mask = np.transpose(mask, (0, 2, 3, 4, 1))

    # Pack the target with the mask so the loss can access both.
    target = np.concatenate([sdf, mask], axis=-1).astype(np.float32)
    return sdf.astype(np.float32), target


def dataset_output_signature(
    target_frames: int,
    grid_size: int,
    channels: int,
) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
    x_shape = (target_frames, grid_size, grid_size, grid_size, channels)
    y_shape = (target_frames, grid_size, grid_size, grid_size, channels * 2)
    return (
        tf.TensorSpec(shape=x_shape, dtype=tf.float32),
        tf.TensorSpec(shape=y_shape, dtype=tf.float32),
    )


def build_dataset(
    files: Sequence[Path],
    target_frames: int,
    grid_size: int,
    channels: int,
    batch_size: int,
    shuffle: bool = False,
    repeat: bool = False,
    prefetch_buffer: int = 1,
    private_threadpool_size: int = 1,
) -> tf.data.Dataset:
    files = [Path(f) for f in files]

    def generator():
        while True:
            order = np.arange(len(files))
            if shuffle:
                np.random.shuffle(order)
            for idx in order:
                yield load_global_sdf_sample(files[idx], target_frames=target_frames)
            if not repeat:
                break

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=dataset_output_signature(
            target_frames=target_frames,
            grid_size=grid_size,
            channels=channels,
        ),
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = int(private_threadpool_size)
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(int(prefetch_buffer))


def summarize_files(files: Sequence[Path]) -> dict:
    return {
        "count": len(files),
        "first": str(files[0]) if files else None,
        "last": str(files[-1]) if files else None,
    }
