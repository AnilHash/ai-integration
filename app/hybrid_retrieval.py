import os
from typing import TypedDict
import bm25s
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from app.embeddings import get_embedder


qdrant_client = QdrantClient(url="http://localhost:6333")
CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", 256))
COLLECTION = f"docs_{CHUNK_SIZE}"


def load_chunks_from_qdrant(
    collection: str,
) -> tuple[list[int], list[str], dict[int, dict]]:
    """
    Single source of truth - BM25's chunk IDs and Qdrant's point IDs can never drift apart because they're pulled from the same collection.
    """
    ids, texts, payload_lookup = [], [], {}
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection, limit=256, offset=offset, with_payload=True
        )
        for p in points:
            ids.append(p.id)
            texts.append(p.payload["text"])  # type: ignore
            payload_lookup[p.id] = p.payload  # type: ignore
        if offset is None:
            break
    return ids, texts, payload_lookup


chunk_ids, chunk_texts, payload_lookup = load_chunks_from_qdrant(COLLECTION)
chunk_lookup = dict(zip(chunk_ids, chunk_texts))

bm25_index = bm25s.BM25()
bm25_index.index(bm25s.tokenize(chunk_texts, stopwords="en"))


def dense_search(query: str, k: int = 20) -> list[tuple[int, float]]:
    vector = get_embedder().encode(f"search_query: {query}").tolist()
    result = qdrant_client.query_points(
        collection_name=COLLECTION, query=vector, limit=k
    )
    return [(int(point.id), point.score) for point in result.points]


def bm25_search(query: str, k: int = 20) -> list[tuple[int, float]]:
    query_tokens = bm25s.tokenize(query, stopwords="en")
    results, scores = bm25_index.retrieve(query_tokens, corpus=chunk_ids, k=k)
    return list(zip(results[0], scores[0]))


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[int, float]], k: int = 60
) -> list[tuple[int, float]]:
    """
    Fuses on RANK, not raw score -- this is what lets a BM25 term-weighted sum and a Qdrant cosine similarity combine meaningfully despite living on completely different scales. k=60 is the constant from the original RRF paper; it dampens the impact of any single ranker's rank-1 pick.
    """
    fused: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked_list, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


class RankedChunk(TypedDict):
    id: int
    score: float


def rerank(query: str, candidate_ids: list[int], top_k: int = 5) -> list[RankedChunk]:
    """
    Cross-encoders take (query, document) PAIRS -- they never embed a single text alone. Only call this on the bounded candidate pool from RRF, never the whole corpus.
    """
    pairs = [(query, chunk_lookup.get(cid, "")) for cid in candidate_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [{"id": cid, "score": float(score)} for cid, score in ranked[:top_k]]


def hybrid_search(query: str, fuse_k: int = 20, final_k: int = 5) -> list[RankedChunk]:
    dense_hits = dense_search(query, k=fuse_k)
    bm25_hits = bm25_search(query, k=fuse_k)
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
    candidate_ids = [cid for cid, _ in fused[:fuse_k]]
    result = rerank(query, candidate_ids, top_k=final_k)
    return result
