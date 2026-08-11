from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from chunking.chunker import chunk_text


CORPUS_DIR = Path("corpus")
CHUNK_SIZES = [256, 512, 1024]
DIM = 768

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def load_corpus() -> list[tuple[str, str]]:
    return [
        (p.stem, p.read_text(encoding="utf-8")) for p in sorted(CORPUS_DIR.glob("*.md"))
    ]


def build_collection(size: int, docs: list[tuple[str, str]]):
    collection = f"docs_{size}"
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

    points, point_id = [], 0
    for doc_id, text in docs:
        for chunk in chunk_text(text, size):
            vector = model.encode(f"search_document: {chunk}").tolist()
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"source_doc": doc_id, "text": chunk, "chunk_size": size},
                )
            )
            point_id += 1

    client.upsert(collection_name=collection, points=points)
    print(f"{collection}:{len(points)} chunks loaded")


def main():
    docs = load_corpus()
    print(f"Loaded {len(docs)} source documents from {CORPUS_DIR}/")
    build_collection(256, docs)
    # for size in CHUNK_SIZES:
    #     build_collection(size, docs)


if __name__ == "__main__":
    main()
