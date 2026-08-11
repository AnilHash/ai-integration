import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunking.chunker import chunk_text

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
GOLDEN_PATH = BASE_DIR / "eval" / "golden" / "queries.json"
CHUNK_SIZES = [256, 512, 1024]

# Punctuation variants that commonly get mangled by copy/paste, editors,
# or non-UTF-8 file writes -- normalize these before comparing so a
# dash/quote encoding drift doesn't masquerade as a chunking-boundary miss.
_PUNCT_MAP = {
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2018": "'",
    "\u2019": "'",  # curly single quotes
    "\u201c": '"',
    "\u201d": '"',  # curly double quotes
    "\u2026": "...",  # ellipsis
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text


def main():
    golden_set = json.loads(GOLDEN_PATH.read_text())
    docs = {p.stem: p.read_text(encoding="utf-8") for p in CORPUS_DIR.glob("*.md")}

    total_checks = 0
    total_misses = 0
    miss_summary = {size: 0 for size in CHUNK_SIZES}

    for item in golden_set:
        doc_text = docs.get(item["source_doc"])
        if doc_text is None:
            print(f"MISSING DOC: {item['source_doc']}")
            continue
        expected = normalize(item["expected_snippet"])
        for size in CHUNK_SIZES:
            total_checks += 1
            chunks = chunk_text(doc_text, size)
            if not any(expected in normalize(c) for c in chunks):
                total_misses += 1
                miss_summary[size] += 1
                print(f"[chunk={size:4d}] snippet NOT intact -- {item['query'][:60]!r}")
                print(
                    f"             doc={item['source_doc']}  snippet={item['expected_snippet']!r}"
                )

    print("\n--- Summary ---")
    print(f"Total (query x chunk_size) checks: {total_checks}")
    print(f"Total misses: {total_misses}")
    for size in CHUNK_SIZES:
        print(f"  chunk_size={size}: {miss_summary[size]} misses")


if __name__ == "__main__":
    main()
