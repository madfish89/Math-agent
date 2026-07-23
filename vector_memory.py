"""Vector-based semantic search for the memory system.

Uses sentence-transformers (or a lightweight fallback) to embed text
and cosine similarity for search, replacing SQLite LIKE queries.
"""
import os
import json
import time
import math
import sqlite3
from config import MEMORY_DB

_EMBEDDER = None
_USE_SENTENCE_TRANSFORMERS = False


def get_embedder():
    """Lazy-load the embedding model. Falls back to hash-based pseudo-embeddings."""
    global _EMBEDDER, _USE_SENTENCE_TRANSFORMERS
    if _EMBEDDER is not None:
        return _EMBEDDER

    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
        _USE_SENTENCE_TRANSFORMERS = True
        print("  [vector_search] Using sentence-transformers all-MiniLM-L6-v2")
        return _EMBEDDER
    except ImportError:
        print("  [vector_search] sentence-transformers not available, using hash-based embeddings")
        _EMBEDDER = _HashEmbedder(dim=384)
        return _EMBEDDER


class _HashEmbedder:
    """Lightweight fallback embedder using character n-gram hashing.
    No external dependencies. Produces 384-dim vectors."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        results = []
        for text in texts:
            vec = [0.0] * self.dim
            text_lower = text.lower()
            # Character 3-grams
            for i in range(len(text_lower) - 2):
                gram = text_lower[i:i+3]
                h = hash(gram) % self.dim
                vec[h] += 1.0
            # Word-level features
            for word in text_lower.split():
                h = hash(word) % self.dim
                vec[h] += 2.0
            # Normalize
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            results.append(vec)

        return results[0] if single else results


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorMemory:
    """Semantic search layer over the SQLite memory database."""

    def __init__(self, db_path=None):
        self.db_path = db_path or MEMORY_DB
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_tables()
        self.embedder = get_embedder()
        self._cache = {}

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ref_type TEXT NOT NULL,
            ref_id INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            text_hash TEXT,
            timestamp REAL NOT NULL
        )""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_emb_ref ON embeddings(ref_type, ref_id)""")
        self.conn.commit()

    def _embed(self, text):
        if isinstance(self.embedder, _HashEmbedder):
            return self.embedder.encode(text)
        else:
            emb = self.embedder.encode([text])
            return emb[0].tolist() if hasattr(emb[0], 'tolist') else list(emb[0])

    def _store_embedding(self, ref_type, ref_id, text):
        emb = self._embed(text)
        emb_str = json.dumps(emb)
        text_hash = str(hash(text))
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (ref_type, ref_id, embedding, text_hash, timestamp) VALUES (?, ?, ?, ?, ?)",
            (ref_type, ref_id, emb_str, text_hash, time.time())
        )
        self.conn.commit()

    def index_problem(self, problem_id, problem_text):
        self._store_embedding("problem", problem_id, problem_text)

    def index_conversation(self, conv_id, text):
        self._store_embedding("conversation", conv_id, text)

    def search_similar(self, query, ref_type="problem", limit=5, threshold=0.3):
        """Find similar items using cosine similarity."""
        query_emb = self._embed(query)

        c = self.conn.execute(
            "SELECT ref_id, embedding FROM embeddings WHERE ref_type = ?",
            (ref_type,)
        )
        results = []
        for row in c.fetchall():
            ref_id = row[0]
            stored_emb = json.loads(row[1])
            sim = cosine_similarity(query_emb, stored_emb)
            if sim >= threshold:
                results.append((ref_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search_problems(self, query, limit=5):
        """Search for similar solved problems."""
        similar = self.search_similar(query, "problem", limit)
        if not similar:
            return []

        problem_ids = [s[0] for s in similar]
        sims = {s[0]: s[1] for s in similar}
        placeholders = ",".join("?" * len(problem_ids))
        c = self.conn.execute(
            f"SELECT id, problem, solution, verified, method FROM problems WHERE id IN ({placeholders})",
            problem_ids
        )
        results = []
        for row in c.fetchall():
            results.append({
                "problem": row[1],
                "solution": row[2],
                "verified": bool(row[3]),
                "method": row[4],
                "similarity": round(sims.get(row[0], 0), 3)
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def search_conversations(self, query, limit=5):
        """Search for similar past conversations."""
        similar = self.search_similar(query, "conversation", limit)
        if not similar:
            return []

        conv_ids = [s[0] for s in similar]
        sims = {s[0]: s[1] for s in similar}
        placeholders = ",".join("?" * len(conv_ids))
        c = self.conn.execute(
            f"SELECT id, role, content FROM conversations WHERE id IN ({placeholders})",
            conv_ids
        )
        results = []
        for row in c.fetchall():
            results.append({
                "role": row[1],
                "content": row[2],
                "similarity": round(sims.get(row[0], 0), 3)
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def close(self):
        self.conn.close()