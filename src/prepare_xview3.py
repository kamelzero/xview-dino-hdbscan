import json
import pathlib as P
import pandas as pd


ROOT = P.Path("data/raw")
OUT = P.Path("features")
OUT.mkdir(parents=True, exist_ok=True)


def load_annots() -> None:
    """
    Load xView3-style annotations and produce a unified parquet index.

    Expected schema (adjust as needed):
      - For each split in {train, val}, a JSON file at data/raw/{split}_annotations.json
      - JSON contains an "images" list where each element has:
          { "file_name": str, "boxes": [ {"x1": int, "y1": int, "x2": int, "y2": int}, ... ] }

    Output: features/meta_boxes.parquet
      Columns: split, img, x1, y1, x2, y2, is_vessel
    """
    records = []
    for split in ["train", "val"]:
        annot_path = ROOT / f"{split}_annotations.json"
        if not annot_path.exists():
            # Skip missing splits silently to allow incremental setup
            continue
        data = json.loads(annot_path.read_text())
        for image_rec in data.get("images", []):
            img_path = str(ROOT / image_rec.get("file_name", ""))
            for bbox in image_rec.get("boxes", []):
                records.append(
                    {
                        "split": split,
                        "img": img_path,
                        "x1": bbox.get("x1"),
                        "y1": bbox.get("y1"),
                        "x2": bbox.get("x2"),
                        "y2": bbox.get("y2"),
                        "is_vessel": 1,
                    }
                )

    if not records:
        raise FileNotFoundError(
            "No annotations found. Place {train,val}_annotations.json under data/raw/ and adjust schema if needed."
        )

    df = pd.DataFrame.from_records(records)
    df.to_parquet(OUT / "meta_boxes.parquet", index=False)


if __name__ == "__main__":
    load_annots()


