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
- We run everything inside **Docker** with **NVIDIA Container Runtime**; user-data bootstraps it.
