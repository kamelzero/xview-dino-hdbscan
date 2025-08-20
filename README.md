# xview-dino-hdbscan (DINOv3/DINOv2 + UMAP/HDBSCAN on xView3 Maritime)

Unsupervised visual clustering + anomaly surfacing for maritime PoL:
- Extract **DINOv3** (or **DINOv2**) embeddings
- Reduce with **UMAP**
- Cluster with **HDBSCAN**
- Review clusters/outliers and derive PoL signals

---

## 1) What to run this on

**Instance types (AWS)**

- **Best value**: `g6.xlarge` (**L4, 24 GB VRAM**)  
- **Alternative**: `g5.xlarge` (**A10G, 24 GB VRAM**)  
- **Faster / more headroom**: `g6.2xlarge` / `g5.2xlarge` (more CPU/RAM), or `g6e.xlarge` (L40S, 48 GB VRAM) if you’ll push large tiles.

**Disk / storage**

- Start with **500 GB – 1 TB gp3 EBS** (raw imagery, crops, embeddings, galleries).  
- Keep raw on **S3** and sync subsets locally.  
- For heavy tiling, consider instances with **NVMe instance store** for faster scratch I/O.

**AMI & drivers**

- **Ubuntu 22.04** or an **NVIDIA/PyTorch DLAMI**.

---

## 2) Setup (uv-based, no conda)

### Install uv and create a virtual environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv /home/ubuntu/xview-dino-hdbscan/.venv
source /home/ubuntu/xview-dino-hdbscan/.venv/bin/activate
```

### Install dependencies

```bash
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install pillow opencv-python pyarrow pandas tqdm hdbscan umap-learn matplotlib seaborn scikit-learn timm
```

### Verify PyTorch

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

If `cuda` is `False` on a GPU instance, ensure NVIDIA drivers are installed and compatible with the CUDA 12.1 runtime used by the PyTorch wheels.

---

## 3) Data acquisition (xView3) and layout

This repo expects the xView3 maritime dataset under `data/raw/` and annotations in a simple JSON format. You need permission/access to xView3 from its official source.

### Expected layout

```
data/
  raw/
    images/                 # or other subfolders; match your annotation paths
    train_annotations.json  # optional if you only run val
    val_annotations.json
```

### Expected annotation schema

Each `*_annotations.json` should look like:

```json
{
  "images": [
    {
      "file_name": "images/scene_0001.jpg",
      "boxes": [
        {"x1": 100, "y1": 200, "x2": 180, "y2": 260},
        {"x1": 320, "y1": 400, "x2": 380, "y2": 460}
      ]
    }
  ]
}
```

Notes:
- `file_name` can be absolute or relative; if relative, it is resolved under `data/raw/`.
- If your xView3 release uses a different structure (CSV or different keys), adapt `src/prepare_xview3.py` accordingly.

### Ways to put data under `data/raw/`

- Existing S3 bucket (example):
  ```bash
  aws s3 sync s3://<your-bucket>/<path-to-xview3>/ /home/ubuntu/xview-dino-hdbscan/data/raw/ \
    --no-progress
  ```
- From a mounted volume or shared path:
  ```bash
  rsync -avh /mnt/xview3/ /home/ubuntu/xview-dino-hdbscan/data/raw/
  ```
- If you have packaged archives:
  ```bash
  mkdir -p /home/ubuntu/xview-dino-hdbscan/data/raw
  tar -xf xview3_images.tar -C /home/ubuntu/xview-dino-hdbscan/data/raw
  cp train_annotations.json val_annotations.json /home/ubuntu/xview-dino-hdbscan/data/raw/
  ```

Once placed, ensure that image paths in the JSON resolve correctly from `data/raw/`.

---

## 4) Run the pipeline

```bash
source /home/ubuntu/xview-dino-hdbscan/.venv/bin/activate
python /home/ubuntu/xview-dino-hdbscan/src/prepare_xview3.py
python /home/ubuntu/xview-dino-hdbscan/src/crop_vessels.py
python /home/ubuntu/xview-dino-hdbscan/src/embed_dinov3.py
python /home/ubuntu/xview-dino-hdbscan/src/cluster.py
python /home/ubuntu/xview-dino-hdbscan/src/sample_gallery.py
```

Outputs:
- Metadata: `features/meta_boxes.parquet`, `features/meta_crops.parquet`, `features/meta_train.parquet`
- Embeddings: `features/dinov3_train.npy`
- Clustering: `features/clustered.parquet`, `features/umap_hdbscan.png`
- Galleries: `features/galleries/cluster_XXX.jpg`, `features/galleries/outliers_top400.jpg`

