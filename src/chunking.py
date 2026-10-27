"""Split cleaned document text into overlapping word chunks, keeping citation metadata.
Pure Python (no models) - tested in tests/test_pipeline.py."""
from __future__ import annotations


def chunk_text(text: str, doc_id: str, page: str = "", chunk_words: int = 300, overlap: int = 50):
    """Return a list of dicts: {chunk_id, doc_id, page, text}.
    chunk_words ~200-500 per the project plan; overlap keeps a sentence that straddles a
    boundary from being cut in half in both chunks."""
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    words = text.split()
    chunks, start, i = [], 0, 0
    step = chunk_words - overlap
    while start < len(words):
        piece = words[start:start + chunk_words]
        chunks.append({"chunk_id": f"{doc_id}::p{page or 'x'}::c{i}", "doc_id": doc_id,
                       "page": page, "text": " ".join(piece)})
        i += 1
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks
