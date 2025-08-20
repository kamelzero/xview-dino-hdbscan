import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


"""
Convert SARFish GRD labels (CSV) into the JSON schema expected by this repo:

Expected JSON per split (train/val):
{
  "images": [
    {
      "file_name": "/abs/path/to/image.tif",
      "boxes": [
        {"x1": int, "y1": int, "x2": int, "y2": int},
        ...
      ]
    },
    ...
  ]
}

Assumptions/Notes:
- We operate on GRD products. Images live inside .SAFE/measurement/*_SARFish.tiff.
- CSV schema varies by release. We try to detect common column names for:
  - image identifier (e.g., product SAFE name, file path, or image basename)
  - bounding boxes: any of {x1,y1,x2,y2} or {xmin,ymin,xmax,ymax} (case-insensitive)
- If multiple polarizations are present (VV/VH), we default to VV unless overridden.
- If we cannot find CSVs or required columns, we print a helpful message and exit non-zero.
"""


COMMON_BBOX_SETS: List[Tuple[str, str, str, str]] = [
    ("x1", "y1", "x2", "y2"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("x_min", "y_min", "x_max", "y_max"),
    ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"),
]

POSSIBLE_IMAGE_KEYS: List[str] = [
    "image_path",
    "img",
    "image",
    "file_name",
    "filename",
    "scene",
    "product",
    "safe_name",
    "image_id",
]


@dataclass
class ConverterConfig:
    dataset_root: Path
    out_root: Path
    polarization: str = "VV"  # or "VH"


def find_csvs(root: Path) -> Dict[str, List[Path]]:
    """Find CSV label files under GRD/{train,validation} (or val)."""
    split_to_csvs: Dict[str, List[Path]] = {"train": [], "val": []}

    # Common subpaths
    candidates: List[Tuple[str, Path]] = []
    for split_key, split_dirname in [("train", "train"), ("val", "validation"), ("val", "val")]:
        candidate_dir = root / split_dirname
        if candidate_dir.exists():
            candidates.append((split_key, candidate_dir))

    for split_key, base in candidates:
        for csv_path in base.rglob("*.csv"):
            split_to_csvs[split_key].append(csv_path)

    return split_to_csvs


def choose_bbox_columns(columns: Iterable[str]) -> Optional[Tuple[str, str, str, str]]:
    lower_cols = {c.lower(): c for c in columns}
    for x1, y1, x2, y2 in COMMON_BBOX_SETS:
        if x1 in lower_cols and y1 in lower_cols and x2 in lower_cols and y2 in lower_cols:
            return (
                lower_cols[x1],
                lower_cols[y1],
                lower_cols[x2],
                lower_cols[y2],
            )
    return None


def choose_image_key(columns: Iterable[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in columns}
    for key in POSSIBLE_IMAGE_KEYS:
        if key in lower_cols:
            return lower_cols[key]
    return None


def build_safe_name_index(root: Path, polarization: str) -> Dict[str, Path]:
    """
    Index SAFE products by a key that likely appears in CSV rows.
    We map several possible keys to the absolute .tiff path we want to use.

    Returns dict from keys like:
      - SAFE basename (e.g., S1B_IW_GRDH_1SDV_..._033A.SAFE)
      - Measurement VV/VH tiff basename (e.g., s1b-iw-grd-vv-..._SARFish.tiff)
    to the absolute tiff path.
    """
    pol = polarization.lower()
    index: Dict[str, Path] = {}
    for safe_dir in root.rglob("*.SAFE"):
        measurement_dir = safe_dir / "measurement"
        if not measurement_dir.exists():
            continue
        # Prefer polarization-specific TIFF
        tiffs = sorted(measurement_dir.glob(f"*grd-{pol}-*_SARFish.tif*"))
        if not tiffs:
            # Fallback: any SARFish.tif(f)
            tiffs = sorted(measurement_dir.glob("*_SARFish.tif*"))
        if not tiffs:
            continue
        chosen = tiffs[0]

        # Keys to map
        index[safe_dir.name] = chosen
        index[safe_dir.stem] = chosen
        index[chosen.name] = chosen
        index[chosen.stem] = chosen
    return index


def row_to_bbox(row: pd.Series, bbox_cols: Tuple[str, str, str, str]) -> Optional[Dict[str, int]]:
    try:
        x1 = int(round(float(row[bbox_cols[0]])))
        y1 = int(round(float(row[bbox_cols[1]])))
        x2 = int(round(float(row[bbox_cols[2]])))
        y2 = int(round(float(row[bbox_cols[3]])))
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def resolve_image_path(
    row: pd.Series,
    image_key: Optional[str],
    safe_index: Dict[str, Path],
) -> Optional[Path]:
    # 1) If row has a path-like value, try it as-is or relative to root
    if image_key is not None:
        raw_value = str(row[image_key]).strip()
        if raw_value:
            candidate = Path(raw_value)
            if candidate.is_file():
                return candidate.resolve()
            # Try mapping via index (by basename or tokens)
            base = candidate.name
            if base in safe_index:
                return safe_index[base]
            stem = candidate.stem
            if stem in safe_index:
                return safe_index[stem]

    # 2) Try to match via any overlapping key in safe_index
    for key in [
        "product",
        "safe_name",
        "scene",
        "image_id",
        "filename",
        "file_name",
    ]:
        if key in row.index:
            value = str(row[key]).strip()
            if value in safe_index:
                return safe_index[value]
            base = Path(value).name
            if base in safe_index:
                return safe_index[base]
            stem = Path(value).stem
            if stem in safe_index:
                return safe_index[stem]
    return None


def convert_split(
    csv_files: List[Path],
    safe_index: Dict[str, Path],
    polarization: str,
) -> Dict[str, List[Dict[str, int]]]:
    image_to_boxes: Dict[str, List[Dict[str, int]]] = defaultdict(list)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        bbox_cols = choose_bbox_columns(df.columns)
        image_key = choose_image_key(df.columns)
        if bbox_cols is None:
            print(
                f"[WARN] {csv_path} has columns {list(df.columns)}, but no bbox columns found. Skipping.",
                file=sys.stderr,
            )
            continue

        for _, row in df.iterrows():
            img_path = resolve_image_path(row, image_key, safe_index)
            if img_path is None:
                # Try last-resort: if CSV has a token that matches any SAFE key substring
                joined = " ".join(str(v) for v in row.values)
                match: Optional[Path] = None
                for key, path in safe_index.items():
                    if key in joined:
                        match = path
                        break
                if match is None:
                    continue
                img_path = match

            bbox = row_to_bbox(row, bbox_cols)
            if bbox is None:
                continue
            image_to_boxes[str(img_path)].append(bbox)

    return image_to_boxes


def write_json(out_path: Path, image_to_boxes: Dict[str, List[Dict[str, int]]]) -> None:
    images = []
    for file_name, boxes in image_to_boxes.items():
        if not boxes:
            continue
        images.append({"file_name": file_name, "boxes": boxes})
    payload = {"images": images}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} with {len(images)} images.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SARFish GRD CSV labels to xView3-style JSON.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path("data/raw/SARFishSample/GRD").resolve()),
        help="Path to SARFishSample/GRD directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data/raw").resolve()),
        help="Output directory for train_annotations.json and val_annotations.json",
    )
    parser.add_argument(
        "--pol",
        type=str,
        default="VV",
        choices=["VV", "VH", "vv", "vh"],
        help="Polarization to use when multiple TIFFs exist",
    )
    args = parser.parse_args()

    cfg = ConverterConfig(dataset_root=Path(args.root), out_root=Path(args.out), polarization=args.pol.upper())
    if not cfg.dataset_root.exists():
        print(f"[ERROR] Dataset root not found: {cfg.dataset_root}")
        sys.exit(1)

    csvs = find_csvs(cfg.dataset_root)
    num_csvs = sum(len(v) for v in csvs.values())
    if num_csvs == 0:
        print(
            "[ERROR] No CSV label files found under "
            f"{cfg.dataset_root}. Expected e.g., GRD/train/GRD_train.csv and GRD/validation/GRD_validation.csv.\n"
            "Please obtain SARFish GRD labels (per xView3 access) and place them under those directories, then re-run.",
            file=sys.stderr,
        )
        sys.exit(2)

    safe_index = build_safe_name_index(cfg.dataset_root, cfg.polarization)
    if not safe_index:
        print(
            f"[ERROR] Could not index any GRD SAFE measurement TIFFs under {cfg.dataset_root}.\n"
            "Ensure the .SAFE archives are extracted.",
            file=sys.stderr,
        )
        sys.exit(3)

    cfg.out_root.mkdir(parents=True, exist_ok=True)

    # Convert validation to val
    val_boxes = convert_split(csvs.get("val", []), safe_index, cfg.polarization)
    if val_boxes:
        write_json(cfg.out_root / "val_annotations.json", val_boxes)
    else:
        print("[WARN] No records converted for val split.", file=sys.stderr)

    # Convert train
    train_boxes = convert_split(csvs.get("train", []), safe_index, cfg.polarization)
    if train_boxes:
        write_json(cfg.out_root / "train_annotations.json", train_boxes)
    else:
        print("[WARN] No records converted for train split.", file=sys.stderr)


if __name__ == "__main__":
    main()



