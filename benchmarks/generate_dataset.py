import numpy as np


def generate_clustered_vectors(
    n_vectors: int = 50000, dim: int = 384, n_clusters: int = 50, seed: int = 42
) -> np.ndarray:
    """
    Synthetic but realistic: vectors scattered around n_clusters centroids, mimicking how real text embeddings cluster around semantic topics. Not real embeddings - that's deliberate, algorithm-structure benchmarking doesn't need real semantic content.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(n_clusters, dim)) * 5
    per_cluster = n_vectors // n_clusters

    vectors = np.vstack(
        [
            centers[i] + rng.normal(0, 1, size=(per_cluster, dim))
            for i in range(n_clusters)
        ]
    ).astype("float32")

    rng.shuffle(vectors)
    return vectors


def generate_queries(
    n_queries: int = 1000, dim: int = 384, seed: int = 43
) -> np.ndarray:
    """
    Held-out query vectors, same distribution, generated independently.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, 3, size=(n_queries, dim)).astype("float32")
