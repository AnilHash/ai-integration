import uuid


NAMESPACE = uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")


def stable_chunk_id(source_doc: str, chunk_index: int) -> str:
    """
    Deterministic UUID from (doc, position-within-doc). Same doc, same chunker, same chunk_index -> always the same ID, regardless of what else exists in the collection. Qdrant accepts UUID strings as point IDs.
    """
    return str(uuid.uuid5(NAMESPACE, f"{source_doc}::{chunk_index}"))
