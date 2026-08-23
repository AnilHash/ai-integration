"""
Run this BEFORE building anything else today. It's the reason today's session exists -- not a  hypothetical, a number from your own corpus.
"""

from app.embeddings import get_embedder
from qdrant_client import QdrantClient
import os

qdrant_client = QdrantClient(url="http://localhost:6333")
CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "256"))

query = "what mem_limit value did the docker-compose qdrant service use"

vector = get_embedder().encode(f"search_query:{query}").tolist()

result = qdrant_client.query_points(
    collection_name=f"docs_{CHUNK_SIZE}",
    query=vector,
    limit=5,
)

print("DENSE-ONLY top 5:")
for i, point in enumerate(result.points, start=1):
    contains_exact = "mem_limit: 512m" in point.payload["text"]  # type: ignore
    print(f" {i}. score={point.score:.3f} contains '512m': {contains_exact}")
