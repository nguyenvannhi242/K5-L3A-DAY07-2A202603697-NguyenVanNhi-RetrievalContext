"""Export Day 6 chunks from pgvector and convert Day 6 benchmark to Day 7 schemas."""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = REPO_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from convert_day6 import convert_chunk, convert_question  # noqa: E402

DEFAULT_DAY6_DIR = Path(
    r"c:\Users\LAPTOP\Desktop\WORKSPACE\LAB 20K AI\K5-L3A-DAY06-NguyenVanNhi-2A202603697-DataIngestion"
)
DEFAULT_DATABASE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"

# q09 (dev, page 24) had its introductory phrase in c000 and the three benefits in c001.
# Day 7 validate() requires the evidence substring to be contained wholly inside one chunk.
Q09_DEV_EVIDENCE = (
    "• enhanced processes for governing, mapping, measuring, and managing AI risk, "
    "and clearly documenting outcomes; • improved awareness of the relationships and "
    "tradeoffs among trustworthiness characteristics, socio-technical approaches, and AI "
    "risks; • explicit processes for making go/no-go system commissioning and deployment "
    "decisions;"
)

UNANSWERABLE_CASES: list[dict[str, Any]] = [
    {
        "id": "q21",
        "split": "dev",
        "question": (
            "What are the mandatory civil penalties or fine amounts imposed by NIST "
            "on organizations that fail to comply with the AI RMF 1.0?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Checked all 42 valid pages (170 chunks) of NIST AI 100-1; the AI RMF is "
            "explicitly voluntary, rights-preserving, and non-sector-specific (pages 6-7) "
            "and contains no mandatory civil penalties, fines, or enforcement sanctions."
        ),
    },
    {
        "id": "q22",
        "split": "dev",
        "question": (
            "What is the required registration fee and annual renewal schedule for "
            "obtaining official NIST AI RMF 1.0 certification?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Checked all 42 valid pages (170 chunks) of NIST AI 100-1; NIST does not "
            "offer a paid AI RMF certification program and specifies no registration fees "
            "or annual certification renewal schedules."
        ),
    },
    {
        "id": "q23",
        "split": "test",
        "question": (
            "Under the Measure function in Part 2, what specific minimum accuracy "
            "percentage threshold must all high-risk AI models achieve before deployment?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Checked Part 2 Measure function (pages 33-36) and all 170 chunks; the "
            "framework qualitative guidance recommends context-specific metrics and TEVV "
            "practices but never prescribes a universal numeric minimum accuracy percentage "
            "threshold."
        ),
    },
    {
        "id": "q24",
        "split": "test",
        "question": (
            "How does Appendix A map the AI RMF 1.0 Core functions to specific articles "
            "of the European Union AI Act and ISO/IEC 42001?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Checked Appendices A-D (pages 40-47) and all 170 chunks; neither the EU AI "
            "Act articles nor ISO/IEC 42001 appear anywhere in the NIST AI 100-1 "
            "(January 2023) document."
        ),
    },
]


def export_chunks_from_pgvector(database_url: str, strategy: str = "C") -> list[dict[str, Any]]:
    """Fetch chunks for the selected Day 6 strategy ordered by page and chunk_id."""
    query = """
        SELECT chunk_id, document_id, source, page, text, metadata
        FROM chunks
        WHERE strategy = %s
        ORDER BY page ASC, chunk_id ASC;
    """
    rows: list[dict[str, Any]] = []
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (strategy,))
            for chunk_id, doc_id, source, page, text, metadata in cur.fetchall():
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "source": source,
                        "page": int(page),
                        "text": text,
                        "metadata": dict(metadata or {}),
                    }
                )
    return rows


def normalize_day6_questions(raw_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Day 6 question dicts before passing through convert_question."""
    normalized: list[dict[str, Any]] = []
    for item in raw_questions:
        q = dict(item)
        if q.get("id") == "q09" and q.get("split") == "dev":
            q["evidence_quote"] = Q09_DEV_EVIDENCE
        if q.get("filter") is None:
            q.pop("filter", None)
        normalized.append(q)
    return normalized


def build_agent_cases(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine converted answerable questions with unanswerable cases."""
    cases: list[dict[str, Any]] = []
    for q in questions:
        case: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
        }
        if "filter" in q:
            case["filter"] = q["filter"]
        cases.append(case)
    cases.extend(UNANSWERABLE_CASES)
    return cases


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Export and convert Day 6 data to Day 7 JSON schema")
    parser.add_argument("--day6-dir", type=Path, default=DEFAULT_DAY6_DIR)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DAY6_DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument("--strategy", default="C")
    parser.add_argument("--export-dir", type=Path, default=STARTER_DIR / "data" / "day6_export")
    parser.add_argument("--output-dir", type=Path, default=STARTER_DIR / "data" / "converted")
    args = parser.parse_args()

    raw_chunks = export_chunks_from_pgvector(args.database_url, strategy=args.strategy)
    if not raw_chunks:
        raise RuntimeError(f"No chunks found in PostgreSQL for strategy={args.strategy!r}")

    questions_path = args.day6_dir / "benchmark" / "questions.json"
    raw_payload = json.loads(questions_path.read_text(encoding="utf-8"))
    raw_questions = normalize_day6_questions(raw_payload["questions"])

    args.export_dir.mkdir(parents=True, exist_ok=True)
    (args.export_dir / "chunks_raw.json").write_text(
        json.dumps(raw_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.export_dir / "questions_raw.json").write_text(
        json.dumps(raw_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    converted_chunks = [convert_chunk(c) for c in raw_chunks]
    converted_questions = [convert_question(q) for q in raw_questions]
    cases = build_agent_cases(converted_questions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("chunks.json", converted_chunks),
        ("questions.json", converted_questions),
        ("cases.json", cases),
    ):
        (args.output_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    dev_ans = sum(1 for c in cases if c["split"] == "dev" and c["answerable"])
    dev_unans = sum(1 for c in cases if c["split"] == "dev" and not c["answerable"])
    test_ans = sum(1 for c in cases if c["split"] == "test" and c["answerable"])
    test_unans = sum(1 for c in cases if c["split"] == "test" and not c["answerable"])
    print(
        f"Exported {len(converted_chunks)} chunks (strategy={args.strategy}), "
        f"{len(converted_questions)} answerable questions, and {len(cases)} total cases "
        f"(dev={dev_ans}+{dev_unans}, test={test_ans}+{test_unans})."
    )


if __name__ == "__main__":
    main()
