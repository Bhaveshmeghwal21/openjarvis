"""Unit-level embedding and vector search.

Spec §7 Stage D: unit-level retrieval uses a strong *general* embedder (BGE-class), not a
scientific one — a medical retrieval study found BGE beat every domain-specific model.
SPECTER2 is for paper-level work and is not used here.

Vectors are stored as float32 BLOBs and searched by brute-force cosine in numpy. At ~100k
units that is tens of milliseconds; see the deviation note in the plan.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jarvis.context import embedding_text
from jarvis.models import Unit


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def dim(self) -> int: ...
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hash-based embedder for tests. No model, no network.

    Each token contributes to two independent buckets, so the chance two distinct tokens
    produce identical vectors is ~1/dim**2 rather than 1/dim. At the default dim that is
    ~1 in 4096, which keeps ranking assertions in tests from flaking.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fake-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                vec[digest[0] % self._dim] += 1.0
                vec[digest[1] % self._dim] += 0.5
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class BGEEmbedder:
    """Real adapter. `sentence_transformers` is imported lazily."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._load().encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


def index_units(conn: sqlite3.Connection, units: Sequence[Unit], embedder: Embedder) -> int:
    """Embed each unit's `embedding_text` and store it. Returns the number indexed."""
    if not units:
        return 0
    vectors = embedder.encode([embedding_text(u) for u in units])
    conn.executemany(
        """
        INSERT INTO embeddings (unit_id, model, dim, vector) VALUES (?,?,?,?)
        ON CONFLICT(unit_id, model) DO UPDATE SET
            dim=excluded.dim, vector=excluded.vector
        """,
        [(u.unit_id, embedder.name, len(v), _pack(v)) for u, v in zip(units, vectors)],
    )
    conn.commit()
    return len(units)


def vector_search(conn: sqlite3.Connection, query_vector: Sequence[float], model: str,
                  limit: int = 20) -> list[tuple[str, float]]:
    """Brute-force cosine over stored vectors. Returns (unit_id, similarity), best first."""
    rows = conn.execute(
        "SELECT unit_id, dim, vector FROM embeddings WHERE model = ?", (model,)
    ).fetchall()
    if not rows:
        return []

    import numpy as np

    query = np.asarray(query_vector, dtype="float32")
    qn = float(np.linalg.norm(query)) or 1.0
    ids = [r["unit_id"] for r in rows]
    matrix = np.asarray([_unpack(r["vector"], r["dim"]) for r in rows], dtype="float32")
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    scores = (matrix @ query) / (norms * qn)

    order = np.argsort(-scores)[:limit]
    return [(ids[i], float(scores[i])) for i in order]
