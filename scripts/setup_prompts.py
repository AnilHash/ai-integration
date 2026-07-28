from dotenv import load_dotenv


load_dotenv()

from langfuse import get_client

langfuse_client = get_client()

langfuse_client.create_prompt(
    name="rag-answer",
    type="text",
    prompt=(
        "Answer the question using ONLY the provided context."
        "If the context does not contain enough information, say so. "
        "Be concise - 2-3 sentences maximum. \n\n"
        "Context:\n{{context}}\n\n"
        "Question: {{query}}\n\n"
        "Answer:"
    ),
    labels=["production"],
)

langfuse_client.create_prompt(
    name="rag-answer",
    type="text",
    prompt=(
        "Answer the question using ONLY the provided context below. "
        "Cite the source document ID (e.g., 'doc-001')) for any fact you use."
        "If the context does not contain enough information, say so clearly. "
        "Be concise - 2-3 sentences maximum.\n\n"
        "Context:\n{{context}}\n\n"
        "Question: {{query}}\n\n"
        "Answer:"
    ),
)

print("Created rag-answer v1 (production) and v2 (citation variant)")
langfuse_client.flush()
