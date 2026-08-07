import time
import numpy as np
import faiss
import matplotlib.pyplot as plt


from generate_dataset import generate_clustered_vectors, generate_queries

DIM = 384
N_VECTORS = 50000
N_QUERIES = 1000
K = 10


def compute_recall_at_k(
    approx_indices: np.ndarray, exact_indices: np.ndarray, k: int = K
) -> float:
    """
    Standard ANN recall definition: for each query, what fraction of the TRUE top-k neighbors (from exact brute-force search) appear anywhere in the approximate method's top-k results.
    """
    recalls = []
    for approx, exact in zip(approx_indices, exact_indices):
        overlap = len(set(approx[:k]) & set(exact[:k]))
        recalls.append(overlap / k)
    return float(np.mean(recalls))


def time_search(index, queries: np.ndarray, k: int = K) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    _, indices = index.search(queries, k)
    elapsed_ms = (time.perf_counter() - start) / len(queries) * 1000
    return indices, elapsed_ms


def main():
    print(f"Generating {N_VECTORS} clustered vectors, dim={DIM} ...")
    vectors = generate_clustered_vectors(N_VECTORS, DIM)
    queries = generate_queries(N_QUERIES, DIM)

    # -- Ground truth: exact brute-force search --
    print("Building exact (Flat) index for ground truth ...")
    t0 = time.perf_counter()
    flat_index = faiss.IndexFlatL2(DIM)
    flat_index.add(vectors)
    flat_build_s = time.perf_counter() - t0
    exact_indices, flat_latency_ms = time_search(flat_index, queries)
    print(
        f" Flat build: {flat_build_s:.1f}s | search: {flat_latency_ms:.2f}ms/query | recall: 1.0 (this IS ground truth)"
    )

    results = [
        {
            "method": "Flat (exact)",
            "param": "-",
            "recall": 1.0,
            "latency_ms": flat_latency_ms,
            "build_s": flat_build_s,
        }
    ]

    # -- HNSW: sweep efSearch --
    print("\nBuilding HNSW index ...")
    t0 = time.perf_counter()
    hnsw_index = faiss.IndexHNSWFlat(DIM, 32)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.add(vectors)
    hnsw_build_s = time.perf_counter() - t0
    print(f" HNSW build: {hnsw_build_s:.1f}s")

    for ef in [16, 32, 64, 128, 256]:
        hnsw_index.hnsw.efSearch = ef
        approx_indices, latency_ms = time_search(hnsw_index, queries)
        recall = compute_recall_at_k(approx_indices, exact_indices)
        print(
            f" HNSW efSearch={ef:>4}: recall={recall:.4f} | latency={latency_ms:.3f}ms/query"
        )
        results.append(
            {
                "method": "HNSW",
                "param": f"ef={ef}",
                "recall": recall,
                "latency_ms": latency_ms,
                "build_s": hnsw_build_s,
            }
        )

    # -- IVF: sweep nprobe --
    print("\n Building IVF index ...")
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(DIM)
    nlist = 200
    ivf_index = faiss.IndexIVFFlat(quantizer, DIM, nlist)
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    ivf_build_s = time.perf_counter() - t0
    print(f" IVF build: {ivf_build_s:.1f}s (nlist={nlist})")

    for nprobe in [1, 4, 16, 64]:
        ivf_index.nprobe = nprobe
        approx_indices, latency_ms = time_search(ivf_index, queries)
        recall = compute_recall_at_k(approx_indices, exact_indices)
        print(
            f" IVF nprobe={nprobe:>3}: recall={recall:.4f} | latency={latency_ms:.3f}ms/query"
        )
        results.append(
            {
                "method": "IVF",
                "param": f"nprobe={nprobe}",
                "recall": recall,
                "latency_ms": latency_ms,
                "build_s": ivf_build_s,
            }
        )

    # -- Plot: recall vs latency, the actual tradeoff curve --
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, marker in [("HNSW", "o"), ("IVF", "s"), ("Flat (exact)", "*")]:
        pts = [r for r in results if r["method"] == method]
        ax.plot(
            [r["latency_ms"] for r in pts],
            [r["recall"] for r in pts],
            marker=marker,
            label=method,
        )
    ax.set_xlabel("Query latency (ms/query)")
    ax.set_ylabel("Recall@10")
    ax.set_title("HNSW vs IVF: Recall/Latency Tradeoff (50k vectors)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig("recall_vs_latency.png", dpi=120)
    print("\n Saved plot to benchmarks/recall_vs_latency.png")


if __name__ == "__main__":
    main()
