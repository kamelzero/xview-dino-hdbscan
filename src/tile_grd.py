import argparse
import pathlib as P
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def list_measurement_tiffs(root: P.Path, polarization: str) -> List[Tuple[str, P.Path]]:
    pol = polarization.lower()
    tiffs: List[Tuple[str, P.Path]] = []
    for safe_dir in root.rglob("*.SAFE"):
        split = "train" if "train" in str(safe_dir).lower() else "val"
        meas = safe_dir / "measurement"
        if not meas.exists():
            continue
        # Prefer polarization-specific
        candidates = sorted(meas.glob(f"*grd-{pol}-*_SARFish.tif*"))
        if not candidates:
            candidates = sorted(meas.glob("*_SARFish.tif*"))
        if not candidates:
            continue
        tiffs.append((split, candidates[0]))
    return tiffs


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        channel = image.astype(np.float32)
        p2, p98 = np.percentile(channel, [2, 98])
        if p98 <= p2:
            p2, p98 = float(channel.min()), float(channel.max() or 1.0)
        scaled = np.clip((channel - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
        u8 = (scaled * 255.0).astype(np.uint8)
        return cv2.merge([u8, u8, u8])
    else:
        # For multi-channel, scale per channel
        channels = []
        for c in range(image.shape[2]):
            ch = image[:, :, c].astype(np.float32)
            p2, p98 = np.percentile(ch, [2, 98])
            if p98 <= p2:
                p2, p98 = float(ch.min()), float(ch.max() or 1.0)
            scaled = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
            channels.append((scaled * 255.0).astype(np.uint8))
        return cv2.merge(channels[:3])


def tile_image(image: np.ndarray, tile_size: int, stride: int) -> List[Tuple[int, int, np.ndarray]]:
    tiles: List[Tuple[int, int, np.ndarray]] = []
    H, W = image.shape[:2]
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            crop = image[y : y + tile_size, x : x + tile_size]
            if crop.size == 0:
                continue
            if crop.ndim == 2:
                crop = cv2.merge([crop, crop, crop])
            # Skip near-uniform tiles
            if np.std(crop) < 4.0:
                continue
            tiles.append((y, x, crop))
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile SARFish GRD measurement TIFFs into images for unsupervised pipeline.")
    parser.add_argument("--root", type=str, default=str(P.Path("data/raw/SARFishSample/GRD").resolve()))
    parser.add_argument("--out", type=str, default=str(P.Path("data/tiles").resolve()))
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--pol", type=str, default="VV", choices=["VV", "VH", "vv", "vh"])
    args = parser.parse_args()

    root = P.Path(args.root)
    out_root = P.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    tiffs = list_measurement_tiffs(root, args.pol)
    if not tiffs:
        print(f"No measurement TIFFs found under {root}. Ensure SAFE archives are extracted.")
        return

    rows = []
    for split, tiff_path in tqdm(tiffs, desc="Tilings"):
        image = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[WARN] Could not read {tiff_path}")
            continue
        image_u8 = normalize_to_uint8(image)
        tiles = tile_image(image_u8, args.tile_size, args.stride)
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        base = tiff_path.stem.replace(".tiff", "").replace(".tif", "")
        for idx, (y, x, crop) in enumerate(tiles):
            out_path = split_dir / f"{base}_y{y}_x{x}.jpg"
            cv2.imwrite(str(out_path), crop)
            rows.append({"split": split, "path": str(out_path), "is_vessel": 0})

    if rows:
        df = pd.DataFrame(rows)
        P.Path("features").mkdir(parents=True, exist_ok=True)
        df.to_parquet("features/meta_tiles.parquet", index=False)
        print(f"Wrote features/meta_tiles.parquet with {len(rows)} tiles.")
    else:
        print("No tiles written.")


if __name__ == "__main__":
    main()



