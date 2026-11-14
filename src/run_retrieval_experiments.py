"""Run retrieval benchmarks across Lexical, Dense (pgvector Day 6), Hybrid RRF, and improvements."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = REPO_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import hit, pack_context, validate  # noqa: E402
from retrievers import (  # noqa: E402
    BM25Retriever,
    Day6DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
    infer_metadata_filter,
)

DEFAULT_DATABASE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"


def evaluate_retriever(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    split: str,
    top_k: int = 3,
    budget: int = 800,
    use_auto_filter: bool = True,
    index_seconds: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        explicit_filter = q.get("filter")
        applied_filter = (
            infer_metadata_filter(q["question"], explicit_filter)
            if use_auto_filter
            else explicit_filter
        )
        started = time.perf_counter()
        results = retriever.search(q["question"], filters=explicit_filter, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        marked = [{**r, "hit": hit(r, q["gold"])} for r in results]
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "strategy": name,
                "top_k": top_k,
                "budget_chars": budget,
                "filter_applied": applied_filter,
                "gold": q["gold"],
                "results": marked,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": elapsed_ms,
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked[:3]),
                "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
            }
        )

    n = len(traces)
    h1 = sum(1 for t in traces if t["hit1"])
    h3 = sum(1 for t in traces if t["hit3"])
    ch = sum(1 for t in traces if t["context_hit"])
    summary = {
        "config": name,
        "split": split,
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "hit1": h1,
        "hit1_rate": round(h1 / n, 4) if n else 0.0,
        "hit3": h3,
        "hit3_rate": round(h3 / n, 4) if n else 0.0,
        "context_hit": ch,
        "context_hit_rate": round(ch / n, 4) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 3) if n else 0.0,
        "index_seconds": round(index_seconds, 4),
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run Day 7 retrieval experiments")
    parser.add_argument("--chunks", type=Path, default=STARTER_DIR / "data" / "converted" / "chunks.json")
    parser.add_argument("--questions", type=Path, default=STARTER_DIR / "data" / "converted" / "questions.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--database-url", default=os.getenv("DAY6_DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--results-dir", type=Path, default=STARTER_DIR / "results")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in all_questions if q.get("split") == args.split]
    validate(chunks, selected)

    t0 = time.perf_counter()
    tfidf_ret = TfidfRetriever(chunks, auto_filter=False)
    t_tfidf = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_ret = BM25Retriever(chunks, auto_filter=True)
    t_bm25 = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense_ret = Day6DenseRetriever(args.database_url, auto_filter=True)
    t_dense = time.perf_counter() - t0

    hybrid_ret = HybridRRFRetriever(dense_ret, bm25_ret, chunks, enable_neighbor_expansion=False)
    hybrid_neighbor_ret = HybridRRFRetriever(dense_ret, bm25_ret, chunks, enable_neighbor_expansion=True)

    configs = [
        (f"{args.split}-tfidf", tfidf_ret, 800, False, t_tfidf),
        (f"{args.split}-dense", dense_ret, 800, True, t_dense),
        (f"{args.split}-bm25", bm25_ret, 800, True, t_bm25),
        (f"{args.split}-hybrid-b800", hybrid_ret, 800, True, t_dense + t_bm25),
        (f"{args.split}-imp1-hybrid-b2400", hybrid_ret, 2400, True, t_dense + t_bm25),
        (f"{args.split}-imp2-hybrid-neighbor-b2400", hybrid_neighbor_ret, 2400, True, t_dense + t_bm25),
    ]

    all_summaries: list[dict[str, Any]] = []
    for folder_name, retriever, budget, auto_f, idx_s in configs:
        summary, traces = evaluate_retriever(
            folder_name,
            retriever,
            selected,
            split=args.split,
            top_k=3,
            budget=budget,
            use_auto_filter=auto_f,
            index_seconds=idx_s,
        )
        out_dir = args.results_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_summaries.append(summary)

    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
