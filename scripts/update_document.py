from pathlib import Path
import sys

from qdrant_client import QdrantClient, models

from scripts.add_document import add_document

q_client = QdrantClient(url="http://localhost:6333")


def update_document(path: Path):
    doc_id = path.stem
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
    add_document(path)
    print(f"Updated '{doc_id}': old chunks removed, new chunks added")


if __name__ == "__main__":
    update_document(Path(sys.argv[1]))
