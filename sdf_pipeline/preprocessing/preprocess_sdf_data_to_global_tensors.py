#!/usr/bin/env python3
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, Iterable, List
import sys

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from test_sdf_video_global_tensor_format import (
    build_summary,
    infer_component_count,
    load_sdf_video,
    make_autoencoder_ready_tensors,
    tensorize_to_global_sdf,
)


DEFAULT_SPLITS = ("TRAIN", "VALIDATION", "TEST")


def discover_input_files(
    input_root: Path,
    splits: Iterable[str],
    filename_glob: str,
) -> Dict[str, List[Path]]:
    discovered: Dict[str, List[Path]] = {}
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Input split directory does not exist: {split_dir}")
        files = sorted(split_dir.rglob(filename_glob))
        discovered[split] = [path for path in files if path.is_file()]
    return discovered


def shard_paths(paths: List[Path], shard_index: int, shard_count: int) -> List[Path]:
    if shard_count <= 1:
        return paths
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard spec: shard_index={shard_index}, shard_count={shard_count}")
    return [path for idx, path in enumerate(paths) if idx % shard_count == shard_index]


def video_id_from_input(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_sdf_video"):
        return stem[: -len("_sdf_video")]
    return stem


def make_output_paths(output_root: Path, split: str, video_id: str):
    split_dir = output_root / split
    npz_path = split_dir / f"{video_id}_global_sdf.npz"
    summary_path = split_dir / f"{video_id}_global_sdf.json"
    return split_dir, npz_path, summary_path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def summarize_report(
    args,
    discovered_counts: Dict[str, int],
    selected_counts: Dict[str, int],
    processed_summaries: List[dict],
    failures: List[dict],
    skipped_existing: List[str],
):
    report = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "splits": list(args.splits),
        "discovered_counts": discovered_counts,
        "selected_counts": selected_counts,
        "processed_count": len(processed_summaries),
        "skipped_existing_count": len(skipped_existing),
        "failure_count": len(failures),
        "skipped_existing": skipped_existing,
        "failures": failures,
        "config": {
            "expected_components": int(args.expected_components),
            "grid_size": int(args.grid_size),
            "fill_value": float(args.fill_value),
            "padding": float(args.padding),
            "box_mode": str(args.box_mode),
            "clip_distance": float(args.clip_distance),
            "normalize": not bool(args.no_normalize),
            "filename_glob": str(args.filename_glob),
            "limit_per_split": int(args.limit_per_split) if args.limit_per_split is not None else None,
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "force": bool(args.force),
        },
    }

    if processed_summaries:
        overall = {
            "frame_count_min": int(min(s["frame_count"] for s in processed_summaries)),
            "frame_count_max": int(max(s["frame_count"] for s in processed_summaries)),
            "frame_count_mean": float(np.mean([s["frame_count"] for s in processed_summaries])),
            "component_valid_fraction_mean": float(
                np.mean([s["component_valid_fraction"] for s in processed_summaries])
            ),
            "raw_fill_fraction_channels_mean": float(
                np.mean([s["raw_fill_fraction_channels"] for s in processed_summaries])
            ),
            "negative_fraction_channels_mean": float(
                np.mean([s["negative_fraction_channels"] for s in processed_summaries])
            ),
        }
        report["overall_stats"] = overall

        per_split = {}
        for split in args.splits:
            split_rows = [s for s in processed_summaries if s.get("split") == split]
            if not split_rows:
                continue
            per_split[split] = {
                "count": len(split_rows),
                "frame_count_min": int(min(s["frame_count"] for s in split_rows)),
                "frame_count_max": int(max(s["frame_count"] for s in split_rows)),
                "frame_count_mean": float(np.mean([s["frame_count"] for s in split_rows])),
                "component_valid_fraction_mean": float(
                    np.mean([s["component_valid_fraction"] for s in split_rows])
                ),
                "raw_fill_fraction_channels_mean": float(
                    np.mean([s["raw_fill_fraction_channels"] for s in split_rows])
                ),
            }
        report["per_split_stats"] = per_split

    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-convert component-wise SDF_DATA videos into global multichannel tensor NPZs "
            "for SDF autoencoder training."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/svu/e0957732/4DHeartModel/sdf_outputs/SDF_DATA"),
        help="Root directory containing split folders such as TRAIN, VALIDATION, TEST.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/svu/e0957732/4DHeartModel/sdf_outputs/SDF_GLOBAL_FULL"),
        help="Output root for global tensor NPZ files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split folder names to process.",
    )
    parser.add_argument(
        "--filename-glob",
        type=str,
        default="*_sdf_video.npz",
        help="Filename pattern used to discover input videos inside each split.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument("--expected-components", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--fill-value", type=float, default=1e3)
    parser.add_argument("--padding", type=float, default=0.02)
    parser.add_argument("--box-mode", choices=["tight", "cube"], default="tight")
    parser.add_argument("--clip-distance", type=float, default=40.0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if NPZ already exists.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path for the aggregate JSON report. Defaults to <output-root>/preprocess_report.json",
    )
    args = parser.parse_args()

    args.splits = [str(split).upper() for split in args.splits]
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.report_path is None:
        args.report_path = args.output_root / "preprocess_report.json"

    discovered = discover_input_files(
        input_root=args.input_root,
        splits=args.splits,
        filename_glob=args.filename_glob,
    )
    discovered_counts = {split: len(paths) for split, paths in discovered.items()}

    selected_counts: Dict[str, int] = {}
    processed_summaries: List[dict] = []
    failures: List[dict] = []
    skipped_existing: List[str] = []

    for split in args.splits:
        split_paths = discovered[split]
        split_paths = shard_paths(split_paths, args.shard_index, args.shard_count)
        if args.limit_per_split is not None:
            split_paths = split_paths[: args.limit_per_split]
        selected_counts[split] = len(split_paths)

        print(f"[{split}] selected {len(split_paths)} files")

        for index, input_path in enumerate(split_paths, start=1):
            video_id = video_id_from_input(input_path)
            split_dir, output_npz, output_summary = make_output_paths(args.output_root, split, video_id)

            if output_npz.exists() and not args.force:
                skipped_existing.append(f"{split}/{video_id}")
                print(f"[SKIP] {split} {index}/{len(split_paths)} {video_id} (output exists)")
                continue

            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RUN ] {split} {index}/{len(split_paths)} {video_id}")

            try:
                metadata, frames = load_sdf_video(input_path)
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
                    input_path=input_path,
                    metadata=metadata,
                    tensorized=tensorized,
                    processed=processed,
                    grid_size=int(args.grid_size),
                    fill_value=float(args.fill_value),
                    clip_distance=float(args.clip_distance),
                    normalize=not bool(args.no_normalize),
                    box_mode=str(args.box_mode),
                )
                summary["split"] = split
                summary["video_id"] = video_id
                summary["output_file"] = str(output_npz)

                np.savez_compressed(
                    output_npz,
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
                    source_file=str(input_path),
                )
                output_summary.write_text(json.dumps(summary, indent=2))
                processed_summaries.append(summary)
            except Exception as exc:
                failures.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "input_file": str(input_path),
                        "output_file": str(output_npz),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[FAIL] {split} {video_id}: {type(exc).__name__}: {exc}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = summarize_report(
        args=args,
        discovered_counts=discovered_counts,
        selected_counts=selected_counts,
        processed_summaries=processed_summaries,
        failures=failures,
        skipped_existing=skipped_existing,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    print("")
    print(json.dumps(report, indent=2))
    print(f"Saved aggregate report to {args.report_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
