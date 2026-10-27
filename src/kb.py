"""Knowledge base = chunks embedded with a multilingual sentence-transformer and stored in ChromaDB.

STATUS: written from library docs; NOT executed here (embedding model must be downloaded from
Hugging Face). Run it in Colab and check the sanity queries in the plan before trusting it.
"""
from __future__ import annotations

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class KnowledgeBase:
    def __init__(self, path: str = "kb_store", name: str = "hwc_kb"):
        import chromadb
        from sentence_transformers import SentenceTransformer
        self.embed = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        vecs = self.embed.encode(texts, show_progress_bar=False).tolist()
        self.col.add(ids=[c["chunk_id"] for c in chunks], documents=texts, embeddings=vecs,
                     metadatas=[{"doc_id": c["doc_id"], "page": str(c["page"])} for c in chunks])

    def retrieve(self, question: str, k: int = 5):
        """Return top-k chunks as dicts, best first. k must be FIXED before final evaluation."""
        qv = self.embed.encode([question]).tolist()
        res = self.col.query(query_embeddings=qv, n_results=k)
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"chunk_id": cid, "text": doc, "doc_id": meta["doc_id"], "page": meta["page"], "distance": dist})
        return out
