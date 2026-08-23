import time
from qdrant_client import QdrantClient
from benchmarks.scale_benchmark import generate_vectors

q_client = QdrantClient(url="http://localhost:6333", timeout=120, prefer_grpc=True)
query = generate_vectors(1, seed=7)[0].tolist()


def timed_query() -> float:
    t0 = time.perf_counter()
    q_client.query_points(collection_name="scale_500k", query=query, limit=10)
    return (time.perf_counter() - t0) * 1000


print("cold (first touch): ", round(timed_query(), 2), " ms")
for i in range(5):
    print(f"warm run {i + 1}:", round(timed_query(), 2), " ms")
# python -m benchmarks.cold_vs_warm
# cold (first touch):  142.87  ms
# warm run 1: 30.49  ms
# warm run 2: 27.6  ms
# warm run 3: 22.84  ms
# warm run 4: 26.02  ms
# warm run 5: 27.45  ms
# python -m benchmarks.cold_vs_warm
# cold (first touch):  9133.57  ms
# warm run 1: 57.86  ms
# warm run 2: 57.87  ms
# warm run 3: 63.44  ms
# warm run 4: 33.96  ms
# warm run 5: 33.74  ms
# python -m benchmarks.cold_vs_warm
# cold (first touch):  128.62  ms
# warm run 1: 36.24  ms
# warm run 2: 32.71  ms
# warm run 3: 26.65  ms
# warm run 4: 27.76  ms
# warm run 5: 31.59  ms
# python -m benchmarks.cold_vs_warm
# cold (first touch):  130.84  ms
# warm run 1: 37.18  ms
# warm run 2: 36.48  ms
# warm run 3: 31.74  ms
# warm run 4: 34.34  ms
# warm run 5: 41.45  ms
