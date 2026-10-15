from qdrant_client import QdrantClient
from app.embeddings import get_embedder


client = QdrantClient(url="http://localhost:6333")

K = 5


def find_rank(hits, expected_snippet: str, expected_doc: str) -> int | None:
    """
    Unchanged from 10.2 - 1-indexed rank of the first hit matching both the expected source doc and containing the expected snippet, else None.
    """
    for i, hit in enumerate(hits, start=1):
        payload = hit.payload
        if (
            payload["source_doc"] == expected_doc
            and expected_snippet in payload["text"]
        ):
            return i
    return None


def score_retrieval(query: str, expected_snippet: str, expected_doc: str) -> int | None:
    vector = get_embedder().encode(f"search_query: {query}").tolist()
    result = client.query_points(collection_name="docs_active", query=vector, limit=K)
    return find_rank(result.points, expected_snippet, expected_doc)
