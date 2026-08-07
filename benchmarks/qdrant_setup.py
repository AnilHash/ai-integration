from qdrant_client import QdrantClient, models
from generate_dataset import generate_clustered_vectors

client = QdrantClient(url="http://localhost:6333")

COLLECTION = "benchmark_vectors"
DIM = 384

if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)


client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(
        size=DIM,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),
    ),
)
print(f"Created collection '{COLLECTION}' with explicit HNSW config ")

vectors = generate_clustered_vectors(n_vectors=4500)
client.upsert(
    collection_name=COLLECTION,
    points=[
        models.PointStruct(id=i, vector=vectors[i].tolist())
        for i in range(len(vectors))
    ],
)
print(f"Upserted {len(vectors)} points")

result = client.query_points(
    collection_name=COLLECTION, query=vectors[0].tolist(), limit=5
)
print(
    f"Query returned {len(result.points)} points, top score: {result.points[0].score:.4f}"
)
