"""
CI-only ingestion - builds a small, fast collection from the same corpus files local dev uses (via actions/checkout), so CI's Qdrant has real, matching content instead of starting empty. Adjust CORPUS_DIR below to whatever contant your actual ingestion script already uses.
"""

from pathlib import Path
from qdrant_client import QdrantClient, models
from app.embeddings import get_embedder
from chunking.stable_ids import stable_chunk_id
from eval.retrieval_scorer import find_rank
from chunking.chunker import chunk_text


CORPUS_DIR = Path("corpus")
CHUNK_SIZE = 256
DIM = 768

client = QdrantClient(url="http://localhost:6333")


def main():
    docs = [
        (p.stem, p.read_text(encoding="utf-8")) for p in sorted(CORPUS_DIR.glob("*.md"))
    ]

    client.create_collection(
        collection_name="docs_ci",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )

    points = []
    for doc_id, text in docs:
        for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
            vector = get_embedder().encode(f"search_document: {chunk}").tolist()
            points.append(
                models.PointStruct(
                    id=stable_chunk_id(doc_id, i),
                    vector=vector,
                    payload={"source_doc": doc_id, "text": chunk},
                )
            )

    client.upsert(collection_name="docs_ci", points=points)

    client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name="docs_ci", alias_name="docs_active"
                )
            )
        ]
    )
    print(
        f"CI ingest: {len(points)} chunks loaded into docs_ci, alias docs_active -> docs_ci"
    )


if __name__ == "__main__":
    main()
