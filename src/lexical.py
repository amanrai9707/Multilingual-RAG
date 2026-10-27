"""A small BM25 keyword retriever. Needs no model download, so it powers --dry-run and is a useful sanity
baseline (if a keyword search beats your embedding search, something is off with the embeddings/chunks)."""
from __future__ import annotations
import math, re
from collections import Counter


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalIndex:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokens(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def retrieve(self, question, k=5):
        q = tokens(question)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
            scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return [dict(self.chunks[i], distance=-s) for s, i in scored[:k] if s > 0]
