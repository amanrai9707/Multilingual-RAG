"""Hand-off format between Person B (clean documents) and Person A (search index).

Person B saves one plain-text file per source document:   data/clean/<doc_id>.txt
Page boundaries are marked on their own line like:       <<<PAGE 12>>>
(The page number must be the page of the ORIGINAL document, because retrieval is scored by doc_id + page.)
Text before the first marker is treated as page 'x'.
"""
from __future__ import annotations
import glob, os, re
from src.chunking import chunk_text

PAGE_RE = re.compile(r"^\s*<<<PAGE\s+([^>\s]+)\s*>>>\s*$")


def load_clean_doc(path):
    """Return a list of (page, text) tuples from one cleaned file."""
    pages, cur_page, buf = [], "x", []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = PAGE_RE.match(line)
            if m:
                if "".join(buf).strip():
                    pages.append((cur_page, "".join(buf).strip()))
                cur_page, buf = m.group(1), []
            else:
                buf.append(line)
    if "".join(buf).strip():
        pages.append((cur_page, "".join(buf).strip()))
    return pages


def ingest_directory(directory, chunk_words=300, overlap=50):
    """Chunk every data/clean/*.txt file page by page. doc_id = file name without .txt."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        for page, text in load_clean_doc(path):
            chunks.extend(chunk_text(text, doc_id, page, chunk_words, overlap))
    return chunks
