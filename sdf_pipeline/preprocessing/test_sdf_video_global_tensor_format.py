#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def load_sdf_video(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return data, data["frames"]


def infer_component_count(frames, expected_components: Optional[int]) -> int:
    if expected_components is not None and expected_components > 0:
        return expected_components

    max_component_idx = -1
    for frame in frames:
        for comp in frame["components"]:
            max_component_idx = max(max_component_idx, int(comp["component_idx"]))
    return max_component_idx + 1


def component_world_bounds(component: dict) -> Tuple[np.ndarray, np.ndarray]:
    sdf = np.asarray(component["sdf"], dtype=np.float32)
    origin = np.asarray(component["origin"], dtype=np.float32)
    voxel_size = np.asarray(component["voxel_size"], dtype=np.float32)
    max_corner = origin + voxel_size * (np.asarray(sdf.shape, dtype=np.float32) - 1.0)
    return origin, max_corner


def compute_global_bounds(frames, padding: float, box_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for frame in frames:
        for comp in frame["components"]:
            comp_min, comp_max = component_world_bounds(comp)
            mins.append(comp_min)
            maxs.append(comp_max)

    if not mins:
        raise ValueError("No components found while computing global bounds.")

    mins = np.vstack(mins)
    maxs = np.vstack(maxs)
    bbox_min = mins.min(axis=0)
    bbox_max = maxs.max(axis=0)
    span = bbox_max - bbox_min
    pad = span * float(padding)
    global_min = bbox_min - pad
    global_max = bbox_max + pad

    if box_mode == "cube":
        center = 0.5 * (global_min + global_max)
        cube_size = float(np.max(global_max - global_min))
        global_min = center - cube_size / 2.0
        global_max = center + cube_size / 2.0

    return global_min.astype(np.float32), global_max.astype(np.float32)


def make_global_grid(global_min: np.ndarray, global_max: np.ndarray, grid_size: int):
    axes = [
        np.linspace(float(global_min[i]), float(global_max[i]), int(grid_size), dtype=np.float32)
        for i in range(3)
    ]
    grid = np.stack(np.meshgrid(axes[0], axes[1], axes[2], indexing="ij"), axis=-1)
    return axes, grid.reshape(-1, 3)


def resample_component_to_global(component: dict, global_points: np.ndarray, fill_value: float, grid_size: int):
    sdf = np.asarray(component["sdf"], dtype=np.float32)
    origin = np.asarray(component["origin"], dtype=np.float32)
    voxel_size = np.asarray(component["voxel_size"], dtype=np.float32)
    local_axes = [
        origin[i] + voxel_size[i] * np.arange(sdf.shape[i], dtype=np.float32)
        for i in range(3)
    ]
    interpolator = RegularGridInterpolator(
        points=local_axes,
        values=sdf,
        method="linear",
        bounds_error=False,
        fill_value=float(fill_value),
    )
    sampled = interpolator(global_points).astype(np.float32)
    return sampled.reshape(grid_size, grid_size, grid_size)


def tensorize_to_global_sdf(
    frames,
    num_components: int,
    grid_size: int,
    fill_value: float,
    padding: float,
    box_mode: str,
):
    global_min, global_max = compute_global_bounds(frames, padding=padding, box_mode=box_mode)
    axes, global_points = make_global_grid(global_min, global_max, grid_size=grid_size)

    frame_count = len(frames)
    sdf_tensor = np.full((frame_count, num_components, grid_size, grid_size, grid_size), fill_value, dtype=np.float32)
    component_mask = np.zeros((frame_count, num_components), dtype=bool)
    frame_indices = np.zeros((frame_count,), dtype=np.int64)

    for frame_pos, frame in enumerate(frames):
        frame_indices[frame_pos] = int(frame["frame_idx"])
        for comp in frame["components"]:
            comp_idx = int(comp["component_idx"])
            if comp_idx >= num_components:
                raise ValueError(
                    f"Component index {comp_idx} exceeds configured component count {num_components} "
                    f"for frame {frame['frame_idx']}"
                )
            sdf_tensor[frame_pos, comp_idx] = resample_component_to_global(
                component=comp,
                global_points=global_points,
                fill_value=fill_value,
                grid_size=grid_size,
            )
            component_mask[frame_pos, comp_idx] = True

    merged_sdf = np.min(sdf_tensor, axis=1)
    voxel_size = np.array(
        [
            axes[0][1] - axes[0][0] if grid_size > 1 else 0.0,
            axes[1][1] - axes[1][0] if grid_size > 1 else 0.0,
            axes[2][1] - axes[2][0] if grid_size > 1 else 0.0,
        ],
        dtype=np.float32,
    )

    return {
        "sdf_tensor_raw": sdf_tensor,
        "merged_sdf_raw": merged_sdf,
        "component_mask": component_mask,
        "frame_indices": frame_indices,
        "grid_origin": global_min,
        "grid_max": global_max,
        "grid_voxel_size": voxel_size,
    }


def make_autoencoder_ready_tensors(tensorized, fill_value: float, clip_distance: float, normalize: bool):
    sdf_tensor_raw = tensorized["sdf_tensor_raw"]
    component_valid_mask = sdf_tensor_raw != float(fill_value)

    sdf_tensor = sdf_tensor_raw.copy()
    sdf_tensor[~component_valid_mask] = float(clip_distance)
    sdf_tensor = np.clip(sdf_tensor, -float(clip_distance), float(clip_distance))

    merged_sdf = np.min(sdf_tensor, axis=1)
    merged_valid_mask = np.any(component_valid_mask, axis=1)

    if normalize:
        sdf_tensor = sdf_tensor / np.float32(clip_distance)
        merged_sdf = merged_sdf / np.float32(clip_distance)

    return {
        "sdf_tensor": sdf_tensor.astype(np.float32),
        "merged_sdf": merged_sdf.astype(np.float32),
        "component_valid_mask": component_valid_mask,
        "merged_valid_mask": merged_valid_mask,
    }


def build_summary(input_path: Path, metadata, tensorized, processed, grid_size: int, fill_value: float, clip_distance: float, normalize: bool, box_mode: str):
    sdf_tensor_raw = tensorized["sdf_tensor_raw"]
    merged_sdf_raw = tensorized["merged_sdf_raw"]
    sdf_tensor = processed["sdf_tensor"]
    component_mask = tensorized["component_mask"]
    merged_sdf = processed["merged_sdf"]
    component_valid_mask = processed["component_valid_mask"]
    merged_valid_mask = processed["merged_valid_mask"]
    nonfill = sdf_tensor_raw[sdf_tensor_raw != fill_value]

    return {
        "input_file": str(input_path),
        "source_npz": str(metadata["source_npz"]) if "source_npz" in metadata else None,
        "frame_count": int(sdf_tensor.shape[0]),
        "channel_count": int(sdf_tensor.shape[1]),
        "global_tensor_shape": list(sdf_tensor.shape),
        "merged_tensor_shape": list(merged_sdf.shape),
        "grid_size": int(grid_size),
        "box_mode": str(box_mode),
        "fill_value": float(fill_value),
        "clip_distance": float(clip_distance),
        "normalized_to_unit_range": bool(normalize),
        "component_counts_per_frame": component_mask.sum(axis=1).astype(int).tolist(),
        "component_mask_all_true": bool(component_mask.all()),
        "component_valid_fraction": float(np.mean(component_valid_mask)),
        "merged_valid_fraction": float(np.mean(merged_valid_mask)),
        "frame_indices_start": int(tensorized["frame_indices"][0]),
        "frame_indices_end": int(tensorized["frame_indices"][-1]),
        "grid_origin": tensorized["grid_origin"].astype(float).tolist(),
        "grid_max": tensorized["grid_max"].astype(float).tolist(),
        "grid_voxel_size": tensorized["grid_voxel_size"].astype(float).tolist(),
        "raw_fill_fraction_channels": float(np.mean(sdf_tensor_raw == fill_value)),
        "raw_fill_fraction_merged": float(np.mean(merged_sdf_raw == fill_value)),
        "raw_nonfill_min": float(np.min(nonfill)),
        "raw_nonfill_max": float(np.max(nonfill)),
        "raw_nonfill_mean": float(np.mean(nonfill)),
        "processed_sdf_min": float(np.min(sdf_tensor)),
        "processed_sdf_max": float(np.max(sdf_tensor)),
        "processed_sdf_mean": float(np.mean(sdf_tensor)),
        "processed_merged_min": float(np.min(merged_sdf)),
        "processed_merged_max": float(np.max(merged_sdf)),
        "processed_merged_mean": float(np.mean(merged_sdf)),
        "negative_fraction_channels": float(np.mean(sdf_tensor < 0)),
        "negative_fraction_merged": float(np.mean(merged_sdf < 0)),
        "surface_point_method": str(metadata["surface_point_method"]) if "surface_point_method" in metadata else None,
        "sign_method": str(metadata["sign_method"]) if "sign_method" in metadata else None,
        "notes": [
            "This output is a shared global multichannel SDF tensor: [T, C, D, H, W].",
            "Channels are globally aligned in one coordinate system, unlike the original local component cubes.",
            "sdf_tensor is clipped and optionally normalized for autoencoder prototyping.",
            "component_valid_mask indicates where resampled values come from a real component grid rather than outside-fill padding.",
            "merged_sdf is the voxelwise min over component channels and is provided only as a convenience baseline.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test-convert a component-wise SDF video into a global multichannel tensor for autoencoder prototyping."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to *_sdf_video.npz")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional JSON summary output")
    parser.add_argument("--expected-components", type=int, default=5, help="Number of component channels")
    parser.add_argument("--grid-size", type=int, default=48, help="Global cubic grid resolution")
    parser.add_argument("--fill-value", type=float, default=1e3, help="Outside-grid fill value")
    parser.add_argument("--padding", type=float, default=0.02, help="Extra padding applied to the global bounds")
    parser.add_argument("--box-mode", type=str, default="tight", choices=["tight", "cube"], help="Whether to use a tight bounding box or expand to a cube")
    parser.add_argument("--clip-distance", type=float, default=40.0, help="Clip SDF magnitudes to this absolute distance")
    parser.add_argument("--no-normalize", action="store_true", help="Keep clipped SDF values in world units instead of scaling to [-1, 1]")
    args = parser.parse_args()

    metadata, frames = load_sdf_video(args.input)
    num_components = infer_component_count(frames, args.expected_components)
    tensorized = tensorize_to_global_sdf(
        frames=frames,
        num_components=num_components,
        grid_size=int(args.grid_size),
        fill_value=float(args.fill_value),
        padding=float(args.padding),
        box_mode=str(args.box_mode),
    )
    processed = make_autoencoder_ready_tensors(
        tensorized=tensorized,
        fill_value=float(args.fill_value),
        clip_distance=float(args.clip_distance),
        normalize=not bool(args.no_normalize),
    )
    summary = build_summary(
        input_path=args.input,
        metadata=metadata,
        tensorized=tensorized,
        processed=processed,
        grid_size=int(args.grid_size),
        fill_value=float(args.fill_value),
        clip_distance=float(args.clip_distance),
        normalize=not bool(args.no_normalize),
        box_mode=str(args.box_mode),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sdf_tensor=processed["sdf_tensor"],
        merged_sdf=processed["merged_sdf"],
        sdf_tensor_raw=tensorized["sdf_tensor_raw"],
        merged_sdf_raw=tensorized["merged_sdf_raw"],
        component_mask=tensorized["component_mask"],
        component_valid_mask=processed["component_valid_mask"],
        merged_valid_mask=processed["merged_valid_mask"],
        frame_indices=tensorized["frame_indices"],
        grid_origin=tensorized["grid_origin"],
        grid_max=tensorized["grid_max"],
        grid_voxel_size=tensorized["grid_voxel_size"],
        source_file=str(args.input),
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved global tensorized output to {args.output}")

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {args.summary_path}")


if __name__ == "__main__":
    main()
