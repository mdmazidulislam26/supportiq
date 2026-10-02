# Sentence-aware chunking + hybrid retrieval (BM25 + dense embeddings fused with RRF).
import re
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi

from config import CANDIDATES, CHUNK_WORDS, RRF_K

TOKEN_RE = re.compile(r"[a-z0-9%]+")


# Lowercase word/number tokens (keeps % so '20%' stays one token).
def tokenize(text):
    return TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


# Split each document into sentence-aware chunks with a one-sentence overlap.
def chunk_documents(docs, target_words=CHUNK_WORDS):
    # Sentence-aware chunking: facts never get cut in the middle of a sentence.
    chunks = []
    for d in docs:
        sentences = re.split(r"(?<=[.!?])\s+", d["text"].strip())
        buf, n = [], 0
        parts = []
        for s in sentences:
            buf.append(s)
            n += len(s.split())
            if n >= target_words:
                parts.append(" ".join(buf))
                buf, n = [buf[-1]], len(buf[-1].split())   # 1-sentence overlap
        if buf and (not parts or " ".join(buf) != parts[-1]):
            tail = " ".join(buf)
            if not parts or tail not in parts[-1]:
                parts.append(tail)
        for i, p in enumerate(parts):
            chunks.append(Chunk(f"{d['id']}-{i}", d["id"], d["title"], p))
    return chunks


# Offline embedder for tests/CI (same interface as the real one).
class TfidfEmbedder:
    # Offline embedder for tests/CI. Same interface as the real one.
    def __init__(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)

    # Return L2-normalised TF-IDF vectors.
    def encode(self, texts):
        return self.vec.transform(texts).toarray().astype("float32")  # rows are L2-normalised


# Real dense embedder (sentence-transformers MiniLM).
class MiniLMEmbedder:
    # Load the sentence-transformers model once.
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    # Return L2-normalised embeddings, so a dot product equals cosine similarity.
    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


# Reciprocal Rank Fusion: merge several rankings without calibrating their scores.
def rrf_fuse(rank_lists, k=RRF_K):
    # Reciprocal Rank Fusion: score = sum 1/(k + rank). Needs no score calibration between retrievers.
    scores = {}
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# BM25 (keywords) + dense (meaning) retrieval, fused with RRF.
class HybridRetriever:
    # Index all chunks for both retrievers.
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.title + " " + c.text) for c in chunks])
        self.embedder = embedder
        self.matrix = embedder.encode([c.title + ". " + c.text for c in chunks])

    # Top-n chunks by cosine similarity: [(chunk_index, score), ...].
    def dense_search(self, query, n=CANDIDATES):
        q = self.embedder.encode([query])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order]

    # Top-n chunks by BM25 keyword score.
    def bm25_search(self, query, n=CANDIDATES):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order]

    # Fuse both rankings; also return the best dense score (used by the out-of-scope gate).
    def search(self, query, k=4):
        dense = self.dense_search(query)
        sparse = self.bm25_search(query)
        fused = rrf_fuse([[i for i, _ in dense], [i for i, _ in sparse]])[:k]
        top_dense = dense[0][1] if dense else 0.0     # used for the out-of-scope decision
        return [self.chunks[i] for i, _ in fused], top_dense
