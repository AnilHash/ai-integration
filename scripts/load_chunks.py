from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from chunking.chunker import chunk_text
from chunking.stable_ids import stable_chunk_id


CORPUS_DIR = Path("corpus")
CHUNK_SIZES = [256]
# CHUNK_SIZES = [256, 512, 1024]
DIM = 768

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def load_corpus() -> list[tuple[str, str]]:
    return [
        (p.stem, p.read_text(encoding="utf-8")) for p in sorted(CORPUS_DIR.glob("*.md"))
    ]


def build_collection(size: int, docs: list[tuple[str, str]]):
    collection = f"docs_active"
    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),
        ),
    )

    points = []
    for doc_id, text in docs:
        for i, chunk in enumerate(chunk_text(text, size)):
            vector = model.encode(f"search_document: {chunk}").tolist()
            points.append(
                models.PointStruct(
                    id=stable_chunk_id(doc_id, i),
                    vector=vector,
                    payload={"source_doc": doc_id, "text": chunk, "chunk_size": size},
                )
            )

    client.upsert(collection_name=collection, points=points)
    print(f"{collection}:{len(points)} chunks loaded")


def main():
    docs = load_corpus()
    print(f"Loaded {len(docs)} source documents from {CORPUS_DIR}/")
    print("docs signature---->>>", docs[0][0], len(docs[0][1]))
    build_collection(256, docs)
    # for size in CHUNK_SIZES:
    #     build_collection(size, docs)


if __name__ == "__main__":
    main()
