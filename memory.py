import sqlite3
import json
import time
from config import MEMORY_DB

_VECTOR_MEMORY = None


def _get_vector_memory():
    global _VECTOR_MEMORY
    if _VECTOR_MEMORY is None:
        try:
            from vector_memory import VectorMemory
            _VECTOR_MEMORY = VectorMemory()
        except Exception:
            _VECTOR_MEMORY = False
    return _VECTOR_MEMORY


class Memory:
    def __init__(self, db_path=None):
        self.db_path = db_path or MEMORY_DB
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            timestamp REAL NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT,
            timestamp REAL NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem TEXT NOT NULL,
            solution TEXT NOT NULL,
            proof TEXT,
            verified INTEGER DEFAULT 0,
            method TEXT,
            timestamp REAL NOT NULL
        )""")
        self.conn.commit()

    def remember(self, role, content, metadata=None):
        c = self.conn.execute(
            "INSERT INTO conversations (role, content, metadata, timestamp) VALUES (?, ?, ?, ?)",
            (role, content, json.dumps(metadata) if metadata else None, time.time())
        )
        self.conn.commit()
        conv_id = c.lastrowid
        vm = _get_vector_memory()
        if vm:
            try:
                vm.index_conversation(conv_id, content)
            except Exception:
                pass

    def recall(self, limit=20):
        c = self.conn.execute(
            "SELECT role, content, metadata FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = c.fetchall()
        return [{"role": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else None} for r in reversed(rows)]

    def store_fact(self, key, value, source=None, confidence=1.0):
        self.conn.execute(
            "INSERT OR REPLACE INTO facts (key, value, confidence, source, timestamp) VALUES (?, ?, ?, ?, ?)",
            (key, value, confidence, source, time.time())
        )
        self.conn.commit()

    def get_fact(self, key):
        c = self.conn.execute("SELECT value, confidence FROM facts WHERE key = ?", (key,))
        row = c.fetchone()
        return {"value": row[0], "confidence": row[1]} if row else None

    def search_facts(self, query):
        vm = _get_vector_memory()
        if vm:
            results = vm.search_similar(query, "problem", limit=5, threshold=0.2)
            if results:
                # Fall through to text search as supplement
                pass
        c = self.conn.execute(
            "SELECT key, value, confidence FROM facts WHERE key LIKE ? OR value LIKE ?",
            (f"%{query}%", f"%{query}%")
        )
        return [{"key": r[0], "value": r[1], "confidence": r[2]} for r in c.fetchall()]

    def search_similar_problems(self, query, limit=5):
        """Semantic search for similar problems using vector embeddings."""
        vm = _get_vector_memory()
        if vm:
            return vm.search_problems(query, limit)
        return []

    def search_conversations(self, query, limit=5):
        """Semantic search for similar past conversations."""
        vm = _get_vector_memory()
        if vm:
            return vm.search_conversations(query, limit)
        return []

    def store_problem(self, problem, solution, proof=None, verified=False, method=None):
        c = self.conn.execute(
            "INSERT INTO problems (problem, solution, proof, verified, method, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (problem, solution, proof, int(verified), method, time.time())
        )
        self.conn.commit()
        problem_id = c.lastrowid
        vm = _get_vector_memory()
        if vm:
            try:
                vm.index_problem(problem_id, problem)
            except Exception:
                pass

    def get_problems(self, limit=10):
        c = self.conn.execute(
            "SELECT problem, solution, proof, verified, method FROM problems ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return [{"problem": r[0], "solution": r[1], "proof": r[2], "verified": bool(r[3]), "method": r[4]} for r in c.fetchall()]

    def close(self):
        self.conn.close()