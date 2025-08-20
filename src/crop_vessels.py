import pathlib as P
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


META = P.Path("features/meta_boxes.parquet")
OUT_DIR = P.Path("data/crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARGIN = 0.5  # 50% margin around bbox
OUT_SIZE = 224


def square_crop(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    margin: float = MARGIN,
    out_size: int = OUT_SIZE,
) -> np.ndarray:
    height, width = image.shape[:2]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_width = (x2 - x1)
    box_height = (y2 - y1)
    side = int(max(box_width, box_height) * (1 + 2 * margin))

    x_min = max(0, int(center_x - side / 2))
    y_min = max(0, int(center_y - side / 2))
    x_max = min(width, x_min + side)
    y_max = min(height, y_min + side)

    crop = image[y_min:y_max, x_min:x_max]

    # pad to square
    crop_h, crop_w = crop.shape[:2]
    pad_side = max(crop_h, crop_w)
    canvas = np.zeros((pad_side, pad_side, 3), dtype=crop.dtype)
    offset_y = (pad_side - crop_h) // 2
    offset_x = (pad_side - crop_w) // 2
    canvas[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = crop

    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)


def main() -> None:
    df = pd.read_parquet(META)
    out_rows: List[dict] = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img = cv2.imread(row.img, cv2.IMREAD_COLOR)
        if img is None:
            continue
        crop = square_crop(img, row.x1, row.y1, row.x2, row.y2)
        out_path = OUT_DIR / f"{row.split}_{index:08d}.jpg"
        cv2.imwrite(str(out_path), crop)
        out_rows.append({"split": row.split, "path": str(out_path), "is_vessel": 1})

    pd.DataFrame(out_rows).to_parquet("features/meta_crops.parquet", index=False)


if __name__ == "__main__":
    main()


