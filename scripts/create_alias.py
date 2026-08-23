from qdrant_client import QdrantClient, models


q_client = QdrantClient(url="http://localhost:6333")

q_client.update_collection_aliases(
    change_aliases_operations=[
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name="docs_256", alias_name="docs_active"
            )
        )
    ]
)
print("alias 'doc_active' -> 'docs_256' ")
