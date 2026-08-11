import os
import sys
import time
from langfuse import observe, get_client, propagate_attributes
from app.llm_client import client, DEFAULT_MODEL
from qdrant_client import QdrantClient
from app.embeddings import get_embedder

qdrant_client = QdrantClient(url="http://localhost:6333")
CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "512"))
langfuse = get_client()


@observe(name="retrieval")
def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    vector = get_embedder().encode(f"search_query: {query}").tolist()
    collection = f"docs_{CHUNK_SIZE}"
    result = qdrant_client.query_points(
        collection_name=collection, query=vector, limit=top_k
    )

    chunks = [
        {
            "id": str(point.id),
            "text": point.payload.get("text", "") if point.payload else "",
            "score": point.score,
        }
        for point in result.points
    ]

    langfuse.update_current_span(
        output=chunks,
        metadata={
            "num_chunks": str(len(chunks)),
            "top_score": str(chunks[0]["score"]) if chunks else "0",
            "retrieval_backend": "qdrant",
        },
    )

    return chunks


@observe(name="context_assembly")
def assemble_context(chunks: list[dict]) -> str:
    context_parts = [f"[Source:{c['id']}]\n{c['text']}" for c in chunks]
    context = "\n\n---\n\n".join(context_parts)

    langfuse.update_current_span(
        output={"context_length_chars": len(context), "num_sources": len(chunks)}
    )
    return context


def generate_answer(query: str, context: str, prompt_version: int | None = None) -> str:
    """
    prompt_version=None  -> fetch whatever's labeled "production" (normal runtime path)
    prompt_version=1 or 2 -> force a specific version (used by the A/B test script)

    langfuse_prompt=prompt on the .create() call links this generation to the
    exact prompt version used — this is what lets Langfuse aggregate metrics
    per prompt version in the UI, and what the A/B test script queries against.
    """
    if prompt_version is not None:
        prompt = langfuse.get_prompt("rag-answer", version=prompt_version)
    else:
        prompt = langfuse.get_prompt("rag-answer", label="production")

    compiled_prompt = prompt.compile(context=context, query=query)

    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": compiled_prompt}],
        max_tokens=256,
        temperature=0.1,
        stream=True,
        stream_options={"include_usage": True},
        langfuse_prompt=prompt,
    )  # type: ignore

    chunks: list[str] = []

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    return "".join(chunks).strip()


@observe(name="rag_query")
def run_rag_pipeline(
    query: str, user_id: str = "anonymous", prompt_version: int | None = None
) -> dict:
    with propagate_attributes(
        user_id=user_id,
        tags=["rag", "v1", "mock_retrieval"],
        metadata={"pipeline_version": "2.1", "query_length": str(len(query))},
    ):
        chunks = retrieve_documents(query)

        context = assemble_context(chunks)

        answer = generate_answer(query, context, prompt_version=prompt_version)

    langfuse.update_current_span(
        output={"answer_length": len(answer), "chunks_used": len(chunks)}
    )
    return {
        "query": query,
        "answer": answer,
        "sources": [c["id"] for c in chunks],
        "chunks_retrieved": len(chunks),
        "trace_id": langfuse.get_current_trace_id(),
    }
