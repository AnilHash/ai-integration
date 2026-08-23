from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_payload_index(
    collection_name="docs_active",
    field_name="source_doc",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
