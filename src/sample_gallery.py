import pathlib as P

import cv2
import numpy as np
import pandas as pd
from math import ceil


def montage(image_paths, num_columns=20, tile_size=112):
    count = len(image_paths)
    num_rows = ceil(count / num_columns)
    canvas = np.zeros((num_rows * tile_size, num_columns * tile_size, 3), dtype=np.uint8)
    for idx, path in enumerate(image_paths[: num_rows * num_columns]):
        row = idx // num_columns
        col = idx % num_columns
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, (tile_size, tile_size))
        canvas[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size] = image
    return canvas


def main() -> None:
    meta = pd.read_parquet("features/clustered.parquet")
    out_dir = P.Path("features/galleries")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Cluster galleries
    for cluster_id in sorted(meta.cluster.unique()):
        if cluster_id == -1:
            continue
        in_cluster = meta[meta.cluster == cluster_id]
        sample = in_cluster.sample(min(200, len(in_cluster)), random_state=0)
        canvas = montage(sample.path.tolist(), num_columns=20, tile_size=112)
        cv2.imwrite(str(out_dir / f"cluster_{int(cluster_id):03d}.jpg"), canvas)

    # Outliers gallery
    outliers = meta.sort_values("outlier_score", ascending=False).head(400)
    canvas = montage(outliers.path.tolist(), num_columns=20, tile_size=112)
    cv2.imwrite(str(out_dir / "outliers_top400.jpg"), canvas)


if __name__ == "__main__":
    main()


