#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh
import vtk
from mesh_to_sdf import mesh_to_voxels
from vtk.util.numpy_support import vtk_to_numpy


def load_reference_faces(mesh_path: Path) -> np.ndarray:
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(mesh_path))
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetPolys() is None or poly.GetPolys().GetNumberOfCells() == 0:
            raise ValueError(f"Reference VTP has no polygon cells: {mesh_path}")
        faces_raw = vtk_to_numpy(poly.GetPolys().GetData()).astype(np.int64)
        return faces_raw.reshape((-1, 4))[:, 1:]

    mesh = trimesh.load(mesh_path, process=False)
    if not hasattr(mesh, "faces"):
        raise ValueError(f"Reference mesh has no faces: {mesh_path}")
    return np.asarray(mesh.faces, dtype=np.int64)


def convert_frame_to_sdf_record(
    frame_idx: int,
    verts_frame: np.ndarray,
    faces_ref: np.ndarray,
    resolution: int,
    padding: float,
    surface_point_method: str,
    sign_method: str,
    scan_count: int,
    scan_resolution: int,
    sample_point_count: int,
    normal_sample_count: int,
    min_component_faces: int,
    expected_components: Optional[int],
) -> dict:
    mesh_in = trimesh.Trimesh(vertices=np.asarray(verts_frame), faces=np.asarray(faces_ref), process=False)
    components_original = mesh_in.split(only_watertight=False)
    components = [c for c in components_original if len(c.faces) >= int(min_component_faces)]
    components = sorted(components, key=lambda m: m.faces.shape[0], reverse=True)
    if expected_components is not None:
        components = components[: int(expected_components)]

    comp_records = []
    for comp_idx, comp in enumerate(components):
        bbox_min = comp.bounds[0]
        bbox_max = comp.bounds[1]
        span = bbox_max - bbox_min
        bbox_min = bbox_min - float(padding) * span
        bbox_max = bbox_max + float(padding) * span

        center = 0.5 * (bbox_min + bbox_max)
        cube_size = float(np.max(bbox_max - bbox_min))
        cube_min = center - cube_size / 2.0

        v_unit = (comp.vertices - center) / cube_size
        comp_unit = trimesh.Trimesh(vertices=v_unit, faces=comp.faces, process=False)

        sdf_unit = mesh_to_voxels(
            comp_unit,
            voxel_resolution=int(resolution),
            surface_point_method=str(surface_point_method),
            sign_method=str(sign_method),
            scan_count=int(scan_count),
            scan_resolution=int(scan_resolution),
            sample_point_count=int(sample_point_count),
            normal_sample_count=int(normal_sample_count),
        )
        sdf_world = sdf_unit.astype(np.float32) * np.float32(cube_size)
        origin = cube_min.astype(np.float32)
        voxel_size = (
            np.array([cube_size, cube_size, cube_size], dtype=np.float32)
            / np.float32(int(resolution) - 1)
        )

        comp_records.append(
            {
                "component_idx": int(comp_idx),
                "input_faces": int(comp.faces.shape[0]),
                "sdf": sdf_world,
                "origin": origin,
                "voxel_size": voxel_size,
            }
        )

    return {
        "frame_idx": int(frame_idx),
        "n_components_original": int(len(components_original)),
        "n_components_kept": int(len(comp_records)),
        "components": comp_records,
    }


def convert_video(
    mesh_video_npz: Path,
    out_video_dir: Path,
    faces_ref: np.ndarray,
    resolution: int,
    padding: float,
    surface_point_method: str,
    sign_method: str,
    scan_count: int,
    scan_resolution: int,
    sample_point_count: int,
    normal_sample_count: int,
    min_component_faces: int,
    expected_components: Optional[int],
    frame_stride: int,
    force: bool,
) -> None:
    video_id = mesh_video_npz.stem
    out_video_dir.mkdir(parents=True, exist_ok=True)
    out_sdf = out_video_dir / f"{video_id}_sdf_video.npz"
    out_settings = out_video_dir / f"{video_id}_settings.json"
    out_log = out_video_dir / "logs.txt"

    if out_sdf.exists() and not force:
        print(f"[SKIP] {video_id}: output exists")
        return

    t0 = time.perf_counter()
    data = np.load(mesh_video_npz)
    if "feat_matrix" not in data:
        raise ValueError(f"Missing feat_matrix in {mesh_video_npz}")
    feat_matrix = data["feat_matrix"].astype(np.float32)

    frame_ids = list(range(0, int(feat_matrix.shape[0]), int(frame_stride)))
    records = []

    with out_log.open("a") as lf:
        lf.write(f"Processing {video_id} with {len(frame_ids)} frames\n")
        for k, frame_idx in enumerate(frame_ids, start=1):
            rec = convert_frame_to_sdf_record(
                frame_idx=frame_idx,
                verts_frame=feat_matrix[frame_idx],
                faces_ref=faces_ref,
                resolution=resolution,
                padding=padding,
                surface_point_method=surface_point_method,
                sign_method=sign_method,
                scan_count=scan_count,
                scan_resolution=scan_resolution,
                sample_point_count=sample_point_count,
                normal_sample_count=normal_sample_count,
                min_component_faces=min_component_faces,
                expected_components=expected_components,
            )
            records.append(rec)
            msg = (
                f"  frame {k}/{len(frame_ids)} done "
                f"(idx={rec['frame_idx']}, comps={rec['n_components_kept']}/{rec['n_components_original']})\n"
            )
            lf.write(msg)

    np.savez_compressed(
        out_sdf,
        frames=np.array(records, dtype=object),
        frame_indices=np.array(frame_ids, dtype=np.int64),
        source_npz=str(mesh_video_npz),
        resolution=np.int64(resolution),
        padding=np.float32(padding),
        surface_point_method=np.array(surface_point_method),
        sign_method=np.array(sign_method),
        scan_count=np.int64(scan_count),
        scan_resolution=np.int64(scan_resolution),
        sample_point_count=np.int64(sample_point_count),
        normal_sample_count=np.int64(normal_sample_count),
        min_component_faces=np.int64(min_component_faces),
        expected_components=(-1 if expected_components is None else np.int64(expected_components)),
    )

    settings = {
        "video_id": video_id,
        "source_npz": str(mesh_video_npz),
        "n_frames_total": int(feat_matrix.shape[0]),
        "n_frames_processed": int(len(frame_ids)),
        "resolution": int(resolution),
        "padding": float(padding),
        "surface_point_method": str(surface_point_method),
        "sign_method": str(sign_method),
        "scan_count": int(scan_count),
        "scan_resolution": int(scan_resolution),
        "sample_point_count": int(sample_point_count),
        "normal_sample_count": int(normal_sample_count),
        "min_component_faces": int(min_component_faces),
        "expected_components": (None if expected_components is None else int(expected_components)),
        "frame_stride": int(frame_stride),
        "timing_seconds_total": float(time.perf_counter() - t0),
    }
    out_settings.write_text(json.dumps(settings, indent=2))
    print(f"[DONE] {video_id}: {settings['timing_seconds_total']:.1f}s")


def shard_paths(paths: List[Path], shard_index: int, shard_count: int) -> List[Path]:
    if shard_count <= 1:
        return paths
    return [p for i, p in enumerate(paths) if i % shard_count == shard_index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch convert mesh videos to explicit SDF NPZs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/kjr/4DHeartModel/experiments/CONRADData_DHB/generated/qdmqtr/val"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/kjr/4DHeartModel/sdf_outputs/SDF_DATA/VALIDATION"),
    )
    parser.add_argument(
        "--reference-mesh",
        type=Path,
        default=Path("/home/kjr/4DHeartModel/experiments/CONRADData_DHB/transform matrices/iulciz/ds_polys/M_0.vtp"),
    )
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--padding", type=float, default=0.01)
    parser.add_argument("--surface-point-method", type=str, default="sample", choices=["scan", "sample"])
    parser.add_argument("--sign-method", type=str, default="normal", choices=["depth", "normal"])
    parser.add_argument("--scan-count", type=int, default=100)
    parser.add_argument("--scan-resolution", type=int, default=400)
    parser.add_argument("--sample-point-count", type=int, default=10000000)
    parser.add_argument("--normal-sample-count", type=int, default=11)
    parser.add_argument("--min-component-faces", type=int, default=100)
    parser.add_argument("--expected-components", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based shard index")
    parser.add_argument("--shard-count", type=int, default=1, help="number of shards")
    args = parser.parse_args()

    if args.surface_point_method == "scan":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYGLET_HEADLESS", "1")

    if args.surface_point_method == "sample" and args.sign_method == "depth":
        print("[WARN] sample + depth is unsupported in mesh_to_sdf; switching sign-method to normal")
        args.sign_method = "normal"

    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    files = sorted(args.input_dir.glob("*.npz"))
    files = shard_paths(files, args.shard_index, args.shard_count)
    if args.limit > 0:
        files = files[: args.limit]

    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Videos to process: {len(files)}")
    print(f"Shard: {args.shard_index + 1}/{args.shard_count}")

    faces_ref = load_reference_faces(args.reference_mesh)
    failed = []
    t_all = time.perf_counter()
    for i, p in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {p.name}")
        out_video_dir = args.output_dir / p.stem
        try:
            convert_video(
                mesh_video_npz=p,
                out_video_dir=out_video_dir,
                faces_ref=faces_ref,
                resolution=args.resolution,
                padding=args.padding,
                surface_point_method=args.surface_point_method,
                sign_method=args.sign_method,
                scan_count=args.scan_count,
                scan_resolution=args.scan_resolution,
                sample_point_count=args.sample_point_count,
                normal_sample_count=args.normal_sample_count,
                min_component_faces=args.min_component_faces,
                expected_components=args.expected_components,
                frame_stride=args.frame_stride,
                force=args.force,
            )
        except Exception as exc:
            failed.append((p.name, str(exc)))
            print(f"[FAIL] {p.name}: {exc}", file=sys.stderr)

    print(f"Finished in {time.perf_counter() - t_all:.1f}s")
    print(f"Failed: {len(failed)}")
    if failed:
        fail_path = args.output_dir / f"failed_shard{args.shard_index:02d}.txt"
        fail_path.write_text("\n".join([f"{n}\t{e}" for n, e in failed]))
        print(f"Failure list: {fail_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
