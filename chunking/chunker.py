import tiktoken

_enc = tiktoken.get_encoding(
    "cl100k_base"
)  # consistent ruler - doesn't need to match your embedding model's own tokenizer, just needs to be consistent across all three sizes


def chunk_text(text: str, size: int, overlap_pct: float = 0.15) -> list[str]:
    tokens = _enc.encode(text)
    step = max(1, int(size * (1 - overlap_pct)))
    chunks, start = [], 0
    while start < len(tokens):
        window = tokens[start : start + size]
        if not window:
            break
        chunks.append(_enc.decode(window))
        if start + size >= len(tokens):
            break
        start += step
    return chunks
