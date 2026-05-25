import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from PIL import Image


WINDOW_MIN = -1000.0
WINDOW_MAX = 400.0
DEFAULT_STRIP_COLUMNS = 0
DEFAULT_MAX_IMAGE_SIDE = 1536
DEFAULT_VOLUME_ROOT = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data"
DEFAULT_INPUT_PATH = DEFAULT_VOLUME_ROOT
DEFAULT_OUTPUT_IMAGE = "/mnt/huali/ct_dataset_10000/pseudo_video_montage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NIfTI volume to pseudo-video montage image (same logic as test_real_data_v6.py)."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to a .nii/.nii.gz file, or a directory that contains NIfTI files.",
    )
    parser.add_argument(
        "--output-image",
        default=DEFAULT_OUTPUT_IMAGE,
        help="Output montage image path (e.g. ./montage.png).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Batch output directory. Used when input is recursively processed into multiple cases.",
    )
    parser.add_argument(
        "--strip-columns",
        type=int,
        default=DEFAULT_STRIP_COLUMNS,
        help="Number of columns in montage grid. Use 0 for auto grid.",
    )
    parser.add_argument(
        "--window-min",
        type=float,
        default=WINDOW_MIN,
        help="Window minimum (HU).",
    )
    parser.add_argument(
        "--window-max",
        type=float,
        default=WINDOW_MAX,
        help="Window maximum (HU).",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIDE,
        help="Resize limit for output image long side. Use 0 to disable resizing.",
    )
    parser.add_argument(
        "--save-slices-dir",
        default="",
        help="Optional directory to save slice frames as PNG (0000.png, 0001.png, ...).",
    )
    return parser.parse_args()


def find_nifti_in_dir(dir_path: str) -> List[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    candidates: List[str] = []
    for ext in (".nii.gz", ".nii"):
        candidates.extend(str(path) for path in sorted(Path(dir_path).glob(f"*{ext}")))
    return candidates


def pick_largest_nifti(nifti_paths: Sequence[str]) -> str:
    if not nifti_paths:
        raise ValueError("No NIfTI files were provided.")
    return max(nifti_paths, key=lambda path: os.path.getsize(path))


def is_nifti_file(path: Path) -> bool:
    text = str(path).lower()
    return text.endswith(".nii") or text.endswith(".nii.gz")


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def collect_nifti_jobs(input_path: str) -> Tuple[List[Dict[str, str]], bool]:
    path = Path(input_path)
    if path.is_file() and is_nifti_file(path):
        return [{"nifti_path": str(path), "rel_dir": ".", "case_name": nifti_stem(path)}], False
    if not path.is_dir():
        raise FileNotFoundError(f"Invalid input path: {input_path}")

    direct_nifti_paths = find_nifti_in_dir(str(path))
    if direct_nifti_paths:
        selected = pick_largest_nifti(direct_nifti_paths)
        selected_path = Path(selected)
        return [{"nifti_path": selected, "rel_dir": ".", "case_name": nifti_stem(selected_path)}], False

    grouped: Dict[Path, List[str]] = {}
    for nifti_file in path.rglob("*.nii"):
        grouped.setdefault(nifti_file.parent, []).append(str(nifti_file))
    for nifti_file in path.rglob("*.nii.gz"):
        grouped.setdefault(nifti_file.parent, []).append(str(nifti_file))

    if not grouped:
        raise FileNotFoundError(f"No .nii/.nii.gz found under directory: {input_path}")

    jobs: List[Dict[str, str]] = []
    for parent_dir in sorted(grouped.keys(), key=lambda p: str(p).lower()):
        selected = pick_largest_nifti(grouped[parent_dir])
        selected_path = Path(selected)
        rel_dir = str(parent_dir.relative_to(path)).replace("\\", "/")
        jobs.append(
            {
                "nifti_path": selected,
                "rel_dir": rel_dir,
                "case_name": nifti_stem(selected_path),
            }
        )
    return jobs, True


def preprocess_volume(img_data: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    if img_data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape={img_data.shape}")
    img_data = np.nan_to_num(img_data, nan=-1024.0, posinf=window_max, neginf=window_min)
    img_data = np.clip(img_data, window_min, window_max)
    return img_data.astype(np.float32)


def process_nifti_volume(nifti_path: str, window_min: float, window_max: float) -> np.ndarray:
    nii = nib.load(nifti_path)
    img_data = np.asarray(nii.get_fdata(), dtype=np.float32)
    if img_data.ndim < 3:
        raise ValueError(f"NIfTI volume must be 3D, got shape={img_data.shape}")
    return preprocess_volume(img_data, window_min, window_max)


def compute_auto_grid(depth: int) -> Tuple[int, int]:
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    rows = max(int(math.sqrt(depth)), 1)
    cols = rows + 1
    while rows * cols < depth:
        cols += 1
    return rows, cols


def volume_to_uint8(volume_hwd: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    normalized = (volume_hwd - window_min) / (window_max - window_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).round().astype(np.uint8)


def volume_to_montage_image(volume_hwd: np.ndarray, cols: int, window_min: float, window_max: float) -> Image.Image:
    depth = volume_hwd.shape[2]
    if cols > 0:
        rows = (depth + cols - 1) // cols
    else:
        rows, cols = compute_auto_grid(depth)

    normalized = volume_to_uint8(volume_hwd, window_min, window_max)

    tiles = []
    index = 0
    for _ in range(rows):
        row_tiles = []
        for _ in range(cols):
            if index < depth:
                row_tiles.append(normalized[:, :, index])
                index += 1
            else:
                row_tiles.append(np.zeros_like(normalized[:, :, 0]))
        tiles.append(np.concatenate(row_tiles, axis=1))

    montage = np.concatenate(tiles, axis=0)
    rgb = np.repeat(montage[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def maybe_resize_image(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image

    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image

    scale = max_side / float(longest_side)
    resized_width = max(int(round(width * scale)), 1)
    resized_height = max(int(round(height * scale)), 1)
    return image.resize((resized_width, resized_height), Image.BILINEAR)


def save_slice_frames(volume_hwd: np.ndarray, output_dir: str, window_min: float, window_max: float) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    normalized = volume_to_uint8(volume_hwd, window_min, window_max)
    depth = normalized.shape[2]
    for i in range(depth):
        frame = normalized[:, :, i]
        rgb = np.repeat(frame[:, :, None], 3, axis=2)
        Image.fromarray(rgb, mode="RGB").save(Path(output_dir) / f"{i:04d}.png")


def main() -> None:
    args = parse_args()
    if args.window_min >= args.window_max:
        raise ValueError("--window-min must be smaller than --window-max")
    if args.input == DEFAULT_INPUT_PATH and not Path(args.input).exists():
        raise FileNotFoundError(
            f"Default --input path does not exist in current environment: {args.input}. "
            "Please pass --input explicitly."
        )

    jobs, is_batch = collect_nifti_jobs(args.input)

    # 单文件处理逻辑
    if not is_batch:
        nifti_path = jobs[0]["nifti_path"]
        try:
            volume = process_nifti_volume(nifti_path, args.window_min, args.window_max)
            montage = volume_to_montage_image(volume, args.strip_columns, args.window_min, args.window_max)
            montage = maybe_resize_image(montage, args.max_image_side)

            output_image = Path(args.output_image)
            output_image.parent.mkdir(parents=True, exist_ok=True)
            montage.save(output_image)

            if args.save_slices_dir:
                save_slice_frames(volume, args.save_slices_dir, args.window_min, args.window_max)

            print(f"[OK] selected_nifti: {nifti_path}")
            print(f"[OK] montage saved: {output_image}")
            if args.save_slices_dir:
                print(f"[OK] slice frames saved: {args.save_slices_dir}")
        except Exception as e:
            print(f"[FAILED] Failed to process {nifti_path}: {e}")
        return

    # 批量处理逻辑
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_image = Path(args.output_image)
        output_root = output_image.parent / f"{output_image.stem}_batch"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[*] batch cases: {len(jobs)}")
    print(f"[*] batch output dir: {output_root}")

    for idx, job in enumerate(jobs, start=1):
        nifti_path = job["nifti_path"]
        rel_dir = job["rel_dir"]
        case_name = job["case_name"]

        # 增加 try...except 捕获异常，防止报错导致进程中断
        try:
            volume = process_nifti_volume(nifti_path, args.window_min, args.window_max)
            montage = volume_to_montage_image(volume, args.strip_columns, args.window_min, args.window_max)
            montage = maybe_resize_image(montage, args.max_image_side)

            case_dir = output_root if rel_dir in {"", "."} else (output_root / rel_dir)
            case_dir.mkdir(parents=True, exist_ok=True)
            output_image = case_dir / f"{case_name}.png"
            montage.save(output_image)

            if args.save_slices_dir:
                slices_root = Path(args.save_slices_dir)
                slice_dir = slices_root / rel_dir / case_name if rel_dir not in {"", "."} else slices_root / case_name
                save_slice_frames(volume, str(slice_dir), args.window_min, args.window_max)

            print(f"[{idx}/{len(jobs)}] [OK] {nifti_path} -> {output_image}")

        except Exception as e:
            # 捕获错误并输出警告，随后跳过当前文件继续处理下一个
            print(f"[{idx}/{len(jobs)}] [SKIPPED] {nifti_path} | Error: {e}")


if __name__ == "__main__":
    main()
