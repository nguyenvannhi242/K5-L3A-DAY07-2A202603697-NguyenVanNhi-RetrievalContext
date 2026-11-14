"""Retrieval methods for Day 7: Lexical (TF-IDF, BM25), Day 6 Dense (pgvector), and Hybrid RRF."""
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import psycopg
from psycopg import sql

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = REPO_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from baseline import TfidfIndex  # noqa: E402

ALLOWED_FILTER_KEYS = {"section", "document_id", "year", "document_type", "language"}

ENGLISH_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "into", "is", "it",
    "its", "may", "must", "not", "of", "on", "or", "should", "such", "that",
    "the", "their", "them", "these", "this", "those", "to", "under", "was",
    "were", "what", "when", "where", "which", "who", "will", "with",
})


class BaseRetriever(Protocol):
    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        ...


def infer_metadata_filter(
    question: str, explicit_filter: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return explicit_filter if provided; otherwise infer reliable section filter from query text."""
    if explicit_filter:
        return dict(explicit_filter)
    low = unicodedata.normalize("NFC", question).casefold()
    if "part 1" in low or "foundational information" in low:
        return {"section": "part1_foundational"}
    if "part 2" in low or "core and profiles" in low:
        return {"section": "part2_core_profiles"}
    if "appendix" in low:
        return {"section": "appendix"}
    if "executive summary" in low:
        return {"section": "executive_summary"}
    return None


def _chunk_matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for key, expected in filters.items():
        if key == "document_id":
            actual = chunk.get("source_id", meta.get("document_id"))
        else:
            actual = chunk.get(key, meta.get(key))
        if actual != expected:
            return False
    return True


def tokenize_lexical(text: str) -> list[str]:
    """Tokenize text while keeping dotted section/category codes like 1.1 or 2.3 and dropping stopwords."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    raw_tokens = re.findall(r"\d+(?:\.\d+)+|\w+", normalized)
    return [tok for tok in raw_tokens if tok not in ENGLISH_STOPWORDS]


class TfidfRetriever:
    """Wrapper around starter TF-IDF baseline complying with the Day 7 Retriever protocol."""

    def __init__(self, chunks: list[dict[str, Any]], *, auto_filter: bool = False) -> None:
        self.chunks = chunks
        self.auto_filter = auto_filter
        self.index = TfidfIndex(chunks)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        effective_filters = infer_metadata_filter(query, filters) if self.auto_filter else filters
        if not effective_filters:
            return self.index.search(query, top_k=top_k)

        from baseline import tokenize as baseline_tokenize

        query_vec = self.index._vector(Counter(baseline_tokenize(query)))
        results: list[dict[str, Any]] = []
        for chunk, vector in zip(self.chunks, self.index.vectors):
            if not _chunk_matches_filters(chunk, effective_filters):
                continue
            score = sum(val * vector.get(term, 0.0) for term, val in query_vec.items())
            if score > 0:
                results.append({**chunk, "score": score, "strategy": "tfidf"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


class BM25Retriever:
    """Okapi BM25 lexical retriever with dotted-code preservation and English stopword filtering."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        auto_filter: bool = True,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.auto_filter = auto_filter

        self.doc_tf: list[Counter[str]] = [Counter(tokenize_lexical(c["text"])) for c in chunks]
        self.doc_len: list[int] = [sum(tf.values()) for tf in self.doc_tf]
        self.avg_dl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0

        df: Counter[str] = Counter()
        for tf in self.doc_tf:
            for term in tf:
                df[term] += 1

        n_docs = len(chunks)
        self.idf: dict[str, float] = {
            term: math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        effective_filters = infer_metadata_filter(query, filters) if self.auto_filter else filters
        q_tokens = tokenize_lexical(query)
        if not q_tokens:
            return []

        results: list[dict[str, Any]] = []
        for chunk, tf, dl in zip(self.chunks, self.doc_tf, self.doc_len):
            if not _chunk_matches_filters(chunk, effective_filters):
                continue
            score = 0.0
            for term in q_tokens:
                if term not in tf or term not in self.idf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                score += self.idf[term] * ((freq * (self.k1 + 1.0)) / denom)
            if score > 0.0:
                results.append({**chunk, "score": round(score, 6), "strategy": "bm25"})

        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class Day6DenseRetriever:
    """Dense retriever querying Day 6 PostgreSQL + pgvector with BAAI/bge-small-en-v1.5."""

    def __init__(
        self,
        database_url: str,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        pg_strategy: str = "C",
        auto_filter: bool = True,
    ) -> None:
        self.database_url = database_url
        self.model_name = model_name
        self.pg_strategy = pg_strategy
        self.auto_filter = auto_filter
        self._query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge-" in model_name.lower()
            else ""
        )

        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)

        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        dim = get_dim()
        if dim is None:
            raise RuntimeError(f"Could not determine embedding dimensions for {model_name}")
        self.dimensions = int(dim)

    def embed_query(self, query: str) -> list[float]:
        text = f"{self._query_prefix}{query}" if self._query_prefix else query
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        effective_filters = infer_metadata_filter(query, filters) if self.auto_filter else filters
        qvec = self.embed_query(query)
        qvec_lit = _vector_literal(qvec)

        where_clauses = [sql.SQL("strategy = %s")]
        filter_params: list[Any] = [self.pg_strategy]

        if effective_filters:
            for key, value in effective_filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {key}")
                where_clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
                filter_params.append(value)

        stmt = sql.SQL(
            """
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1.0 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s;
            """
        ).format(where_sql=sql.SQL(" AND ").join(where_clauses))

        params = [qvec_lit, *filter_params, qvec_lit, top_k]
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(stmt, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for chunk_id, doc_id, source, page, text, metadata, score in rows:
            results.append(
                {
                    "id": chunk_id,
                    "source_id": doc_id,
                    "source": source,
                    "page": int(page),
                    "text": text,
                    "metadata": dict(metadata or {}),
                    "score": round(float(score), 6),
                    "strategy": "dense_pgvector",
                }
            )
        return results


class HybridRRFRetriever:
    """Hybrid retriever fusing Dense and BM25 rankings via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        chunks: list[dict[str, Any]],
        *,
        rrf_k: int = 20,
        w_dense: float = 1.0,
        w_bm25: float = 1.0,
        candidate_multiplier: int = 4,
        enable_neighbor_expansion: bool = False,
    ) -> None:
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.w_dense = w_dense
        self.w_bm25 = w_bm25
        self.candidate_multiplier = candidate_multiplier
        self.enable_neighbor_expansion = enable_neighbor_expansion

        self._chunk_by_id = {c["id"]: c for c in chunks}
        self._pos_by_id = {c["id"]: idx for idx, c in enumerate(chunks)}

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        cand_k = max(10, top_k * self.candidate_multiplier)
        dense_hits = self.dense.search(query, filters=filters, top_k=cand_k)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=cand_k)

        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["id"]
            chunk_map[cid] = hit
            fused_scores[cid] = fused_scores.get(cid, 0.0) + self.w_dense / (self.rrf_k + rank)

        for rank, hit in enumerate(bm25_hits, start=1):
            cid = hit["id"]
            chunk_map.setdefault(cid, hit)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + self.w_bm25 / (self.rrf_k + rank)

        ranked = sorted(fused_scores.items(), key=lambda kv: (-kv[1], kv[0]))

        if self.enable_neighbor_expansion and ranked:
            for cid, base_score in ranked[:2]:
                idx = self._pos_by_id.get(cid)
                if idx is None or idx + 1 >= len(self.chunks):
                    continue
                curr_chunk = self.chunks[idx]
                next_chunk = self.chunks[idx + 1]
                same_page = (
                    curr_chunk["source_id"] == next_chunk["source_id"]
                    and curr_chunk["page"] == next_chunk["page"]
                )
                is_lead_in = curr_chunk["text"].rstrip().endswith(":") or next_chunk["text"].lstrip().startswith("•")
                if same_page and is_lead_in:
                    nid = next_chunk["id"]
                    chunk_map.setdefault(nid, next_chunk)
                    boosted = base_score * 0.95
                    if boosted > fused_scores.get(nid, 0.0):
                        fused_scores[nid] = boosted
            ranked = sorted(fused_scores.items(), key=lambda kv: (-kv[1], kv[0]))

        strategy_name = "hybrid_rrf_neighbor" if self.enable_neighbor_expansion else "hybrid_rrf"
        results: list[dict[str, Any]] = []
        for cid, score in ranked[:top_k]:
            base = dict(chunk_map[cid])
            base["score"] = round(score, 6)
            base["strategy"] = strategy_name
            results.append(base)
        return results
