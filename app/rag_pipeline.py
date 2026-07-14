import time
from langfuse import observe, get_client, propagate_attributes
from app.llm_client import client, DEFAULT_MODEL

langfuse = get_client()


@observe(name="retrieval")
def retrieve_documents(query: str) -> list[dict]:
    time.sleep(0.1)

    mock_chunks = [
        {
            "id": "doc-001",
            "text": "FastAPI is a modern, high-performance Python web framework built on Starlette.",
            "score": 0.91,
        },
        {
            "id": "doc-002",
            "text": "Pydantic v2 handles data validation using Python type annotations.",
            "score": 0.87,
        },
        {
            "id": "doc-003",
            "text": "Docker containers package applications with all their dependencies.",
            "score": 0.72,
        },
    ]

    langfuse.update_current_span(
        output=mock_chunks,
        metadata={
            "num_chunks": str(len(mock_chunks)),
            "top_score": str(mock_chunks[0]["score"]),
            "retrieval_backend": "mock_v0",
        },
    )

    return mock_chunks


@observe(name="context_assembly")
def assemble_context(chunks: list[dict]) -> str:
    context_parts = [f"[Source:{c['id']}]\n{c['text']}" for c in chunks]
    context = "\n\n---\n\n".join(context_parts)

    langfuse.update_current_span(
        output={"context_length_chars": len(context), "num_sources": len(chunks)}
    )
    return context


def generate_answer(query: str, context: str) -> str:
    prompt = (
        "Answer the question using ONLY the provided context. "
        "If the context does not contain enough information, say so. "
        "Be concise - 2-3 sentences maximum.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:{query}\n\n"
        "Answer:"
    )

    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks: list[str] = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    return "".join(chunks).strip()


@observe(name="rag_query")
def run_rag_pipeline(query: str, user_id: str = "anonymous") -> dict:
    with propagate_attributes(
        user_id=user_id,
        tags=["rag", "v1", "mock_retrieval"],
        metadata={"pipeline_version": "2.1", "query_length": str(len(query))},
    ):
        chunks = retrieve_documents(query)

        context = assemble_context(chunks)

        answer = generate_answer(query, context)

    langfuse.update_current_span(
        output={"answer_length": len(answer), "chunks_used": len(chunks)}
    )
    return {
        "query": query,
        "answer": answer,
        "sources": [c["id"] for c in chunks],
        "chunks_retrieved": len(chunks),
    }
