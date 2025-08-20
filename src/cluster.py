import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def main() -> None:
    X = np.load("features/dinov3_train.npy")
    meta = pd.read_parquet("features/meta_train.parquet")

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    Z = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(Z)

    meta["cluster"] = labels
    meta["outlier_score"] = clusterer.outlier_scores_
    meta.to_parquet("features/clustered.parquet", index=False)

    # Quick visualization
    plt.figure(figsize=(8, 7))
    is_cluster = labels != -1
    plt.scatter(Z[~is_cluster, 0], Z[~is_cluster, 1], s=3, alpha=0.3, label="noise/outliers")
    plt.scatter(
        Z[is_cluster, 0],
        Z[is_cluster, 1],
        s=4,
        alpha=0.8,
        c=labels[is_cluster],
        cmap="tab20",
        label="clusters",
    )
    plt.legend()
    plt.title("UMAP + HDBSCAN on DINO embeddings")
    plt.savefig("features/umap_hdbscan.png", dpi=150)

    valid_mask = is_cluster
    if valid_mask.sum() > 1 and len(np.unique(labels[valid_mask])) > 1:
        print("Silhouette:", silhouette_score(X[valid_mask], labels[valid_mask], metric="cosine"))


if __name__ == "__main__":
    main()


