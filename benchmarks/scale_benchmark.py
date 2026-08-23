# benchmarks/scale_benchmark.py
import time
import numpy as np
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=120)
DIM = 768


def build_collection(name: str, on_disk: bool = False, shard_number: int = 1):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        shard_number=shard_number,
        vectors_config=models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
            on_disk=on_disk,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=32,
            ef_construct=200,
            on_disk=on_disk,
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )


def upload_and_time(name: str, n: int, seed: int = 42, batch_size: int = 2000) -> float:
    """
    Streams vectors and uploads in fixed-size chunks so peak memory stays
    bounded by batch_size, not n. Avoids materializing the full n x DIM
    array or the full points list at once.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    uploaded = 0
    while uploaded < n:
        this_batch = min(batch_size, n - uploaded)
        batch_vectors = rng.normal(size=(this_batch, DIM)).astype(np.float32)
        batch_points = [
            models.PointStruct(id=uploaded + i, vector=batch_vectors[i].tolist())
            for i in range(this_batch)
        ]
        client.upsert(collection_name=name, points=batch_points, wait=False)
        uploaded += this_batch
        del batch_vectors, batch_points  # release before next chunk

    client.update_collection(
        collection_name=name,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1, max_optimization_threads=2
        ),
    )
    max_wait_s = 6000
    while True:
        info = client.get_collection(name)
        indexed = info.indexed_vectors_count
        points_count = info.points_count
        elapsed = time.perf_counter() - t0
        # print(
        #     f"[{elapsed:.0f}s] points={points_count} indexed={indexed}/{n} "
        #     f"status={info.status} optimizer_status={info.optimizer_status}"
        # )
        if indexed is not None and indexed >= n and info.optimizer_status == "ok":
            break
        if elapsed > max_wait_s:
            raise TimeoutError(
                f"Indexing did not complete within {max_wait_s}s (indexed={indexed}/{n}) , status = {info.optimizer_status}"
            )
        time.sleep(2)
    return time.perf_counter() - t0


def measure_query_latency(name: str, n_queries: int = 200, seed: int = 99) -> float:
    rng = np.random.default_rng(seed)
    queries = rng.normal(size=(n_queries, DIM)).astype(np.float32)
    t0 = time.perf_counter()
    for q in queries:
        client.query_points(collection_name=name, query=q.tolist(), limit=10)
    return (time.perf_counter() - t0) / n_queries * 1000  # ms


def run(
    name: str,
    n: int,
    on_disk: bool = False,
    shard_number: int = 1,
    batch_size: int = 2000,
) -> dict:
    print(f"{name}: building {n} vectors, on_disk={on_disk}, shards={shard_number}...")
    build_collection(name, on_disk=on_disk, shard_number=shard_number)
    upload_s = upload_and_time(name, n, batch_size=batch_size)
    latency_ms = measure_query_latency(name)
    result = {
        "name": name,
        "n": n,
        "on_disk": on_disk,
        "shards": shard_number,
        "upload_s": round(upload_s, 1),
        "avg_query_ms": round(latency_ms, 2),
    }
    print(result)
    return result


if __name__ == "__main__":
    run("scale_10k", 10_000)
    run("scale_100k", 100_000)
    # run("scale_500k", 500_000, on_disk=True, batch_size=1000)
    # smaller batches for the big run
# python -m benchmarks.scale_benchmark_1
# scale_10k: building 10000 vectors, on_disk=False, shards=1...
# {'name': 'scale_10k', 'n': 10000, 'on_disk': False, 'shards': 1, 'upload_s': 25.2, 'avg_query_ms': 13.24}
# scale_100k: building 100000 vectors, on_disk=False, shards=1...
# {'name': 'scale_100k', 'n': 100000, 'on_disk': False, 'shards': 1, 'upload_s': 516.6, 'avg_query_ms': 18.82}
# scale_500k: building 500000 vectors, on_disk=True, shards=1...
# {'name': 'scale_500k', 'n': 500000, 'on_disk': True, 'shards': 1, 'upload_s': 4184.3, 'avg_query_ms': 609.19}
# python -m benchmarks.scale_benchmark_1
# scale_10k: building 10000 vectors, on_disk=False, shards=1...
# {'name': 'scale_10k', 'n': 10000, 'on_disk': False, 'shards': 1, 'upload_s': 30.0, 'avg_query_ms': 12.35}
# scale_100k: building 100000 vectors, on_disk=False, shards=1...
# {'name': 'scale_100k', 'n': 100000, 'on_disk': False, 'shards': 1, 'upload_s': 628.3, 'avg_query_ms': 23.83}
# scale_500k: building 500000 vectors, on_disk=True, shards=1...
# {'name': 'scale_500k', 'n': 500000, 'on_disk': True, 'shards': 1, 'upload_s': 4159.2, 'avg_query_ms': 571.88}
