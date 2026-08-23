from qdrant_client import QdrantClient

q_client = QdrantClient(url="http://localhost:6333")


def main():
    # points = q_client.query_points(
    #     collection_name="docs_256",
    #     limit=1,
    #     query_filter=models.Filter(
    #         must=[
    #             models.FieldCondition(
    #                 key="source_doc",
    #                 match=models.MatchValue(value="10_5_embedding_pipeline_operations"),
    #             )
    #         ]
    #     ),
    # )
    # print(f"Points: {points.model_dump_json()}")
    info = q_client.get_collection("docs_active")
    print(
        "status:",
        info.status,
        "| points:",
        info.points_count,
        "| segments:",
        info.segments_count,
        "| payload indexes:",
        info.payload_schema,
        "| indexed:",
        info.indexed_vectors_count,
    )


if __name__ == "__main__":
    main()
