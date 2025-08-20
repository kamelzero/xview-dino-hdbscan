import pathlib as P

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# Quick-start default: use timm DINOv2. Set to False if you want to switch to DINOv3.
USE_TIMM = True


def build_model():
    if USE_TIMM:
        import timm

        model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
        feature_dim = model.num_features
        model.reset_classifier(0)
        return model, feature_dim
    else:
        import dinov3
        from dinov3.dinov3 import build_model_from_cfg

        cfg = dinov3.get_cfg("dinov3_vitl14")
        model = build_model_from_cfg(cfg)
        checkpoint = torch.load("dinov3/weights/dinov3_vitl14.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        # Depending on the implementation, adapt attribute name for feature dim
        feature_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
        return model, feature_dim


def iterate_in_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    # Prefer crops if available; otherwise fall back to tiles
    crops_path = P.Path("features/meta_crops.parquet")
    tiles_path = P.Path("features/meta_tiles.parquet")
    if crops_path.exists():
        df = pd.read_parquet(crops_path)
    elif tiles_path.exists():
        df = pd.read_parquet(tiles_path)
    else:
        raise FileNotFoundError("Neither features/meta_crops.parquet nor features/meta_tiles.parquet found.")

    model, feature_dim = build_model()
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # Determine target input size from model config (e.g., (3, 518, 518) for DINOv2 L/14)
    input_size = getattr(getattr(model, "default_cfg", {}), "get", lambda k, d: (3, 224, 224))("input_size", (3, 224, 224))
    if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
        target_hw = (int(input_size[1]), int(input_size[2]))
    else:
        target_hw = (518, 518)

    transform = transforms.Compose(
        [
            transforms.Resize(target_hw, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image_paths = df["path"].tolist()
    # Larger inputs (e.g., 518x518) require smaller batches
    batch_size = 16 if use_cuda else 8
    all_features = []

    with torch.no_grad():
        for batch_paths in tqdm(list(iterate_in_batches(image_paths, batch_size))):
            tensors = []
            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                tensors.append(transform(image))
            batch = torch.stack(tensors, dim=0)
            if torch.cuda.is_available():
                batch = batch.cuda(non_blocking=True)

            outputs = model(batch)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            features = outputs.detach().cpu().float().numpy()
            all_features.append(features)

    embeddings = np.concatenate(all_features, axis=0)
    np.save("features/dinov3_train.npy", embeddings)
    df.to_parquet("features/meta_train.parquet", index=False)


if __name__ == "__main__":
    main()


