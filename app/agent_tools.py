from app.rag_pipeline import retrieve_documents


def search_docs(query: str) -> str:
    chunks = retrieve_documents(query, top_k=3)
    return "\n".join(c["text"] if isinstance(c, dict) else c for c in chunks)


def word_count(text: str) -> int:
    return len(text.split())
