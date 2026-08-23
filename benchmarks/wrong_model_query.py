from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


q_client = QdrantClient(url="http://localhost:6333")
query = "why doesn't Qdrant offer an IVF index option?"

v1 = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
v2 = SentenceTransformer("google/embeddinggemma-300m", trust_remote_code=True)


wrong = q_client.query_points(
    collection_name="docs_256_v2", query=v1.encode(query).tolist(), limit=3
)
correct = q_client.query_points(
    collection_name="docs_256_v2", query=v2.encode(query).tolist(), limit=3
)

print("WRONG model embedding -- top score:0", round(wrong.points[0].score, 4))
print("CORRECT model embedding -- top score:", round(correct.points[0].score, 4))
