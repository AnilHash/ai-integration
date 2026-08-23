import os
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from chunking.chunker import chunk_text
from chunking.stable_ids import stable_chunk_id


CORPUS_DIR = Path("corpus")
CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", 256))
NEW_COLLECTION = "docs_256_v2"
NEW_MODEL = "google/embeddinggemma-300m"

q_client = QdrantClient(url="http://localhost:6333")
new_embedder = SentenceTransformer(NEW_MODEL, trust_remote_code=True)


def build_new_collection():
    if q_client.collection_exists(NEW_COLLECTION):
        q_client.delete_collection(NEW_COLLECTION)

    q_client.create_collection(
        collection_name=NEW_COLLECTION,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),
    )

    points = []

    for path in sorted(CORPUS_DIR.glob("*.md")):
        doc_id = path.stem
        for i, chunk in enumerate(
            chunk_text(path.read_text(encoding="utf-8"), CHUNK_SIZE)
        ):
            points.append(
                models.PointStruct(
                    id=stable_chunk_id(doc_id, i),
                    vector=new_embedder.encode(chunk).tolist(),
                    payload={
                        "source_doc": doc_id,
                        "text": chunk,
                        "chunk_size": CHUNK_SIZE,
                    },
                )
            )
    q_client.upsert(collection_name=NEW_COLLECTION, points=points)
    print(f"{NEW_COLLECTION}: {len(points)} chunks embedded with {NEW_MODEL}")


if __name__ == "__main__":
    build_new_collection()
