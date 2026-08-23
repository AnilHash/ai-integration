import sys

from qdrant_client import QdrantClient, models

q_client = QdrantClient(url="http://localhost:6333")


def delete_document(doc_id: str):
    q_client.delete(
        collection_name="docs_active",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_doc", match=models.MatchValue(value=doc_id)
                    )
                ]
            )
        ),
    )
    print(f"Deleted all chunks for '{doc_id}'")


if __name__ == "__main__":
    delete_document(sys.argv[1])
