import sys
from pathlib import Path
from qdrant_client import QdrantClient, models
from app.embeddings import get_embedder
from chunking.chunker import chunk_text
from chunking.stable_ids import stable_chunk_id


CHUNK_SIZE = 256
q_client = QdrantClient(
    url="http://localhost:6333",
)


def add_document(path: Path):
    doc_id = path.stem
    text = path.read_text(encoding="utf-8")
    embedder = get_embedder()

    points = []

    for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
        points.append(
            models.PointStruct(
                id=stable_chunk_id(doc_id, i),
                vector=embedder.encode(chunk).tolist(),
                payload={"source_doc": doc_id, "text": chunk, "chunk_size": CHUNK_SIZE},
            )
        )
    q_client.upsert(collection_name="docs_active", points=points)
    print(f"Added {len(points)} chunks for '{doc_id}' -- no existing points touched")


if __name__ == "__main__":
    add_document(Path(sys.argv[1]))
