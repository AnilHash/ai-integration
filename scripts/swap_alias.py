from qdrant_client import QdrantClient, models

q_client = QdrantClient(url="http://localhost:6333")
q_client.update_collection_aliases(
    change_aliases_operations=[
        models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name="docs_active")
        ),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name="docs_256_v2", alias_name="docs_active"
            )
        ),
    ]
)
print("docs_active now points to doc_256_v2")
