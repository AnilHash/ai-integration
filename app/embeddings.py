from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """
    Loaded once per process, on first call - same pattern as a singleton DB connection pool.
    """
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
