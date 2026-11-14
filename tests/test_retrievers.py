"""Unit tests for Day 7 retrievers: metadata inference, BM25, Hybrid RRF, and neighbor expansion."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrievers import (  # noqa: E402
    BM25Retriever,
    HybridRRFRetriever,
    infer_metadata_filter,
)


def make_chunk(cid: str, page: int, text: str, section: str = "part1_foundational") -> dict:
    return {
        "id": cid,
        "source_id": "doc",
        "source": "doc.pdf",
        "page": page,
        "text": text,
        "metadata": {"section": section, "year": 2023},
    }


class FakeDenseRetriever:
    def __init__(self, hits_by_query: dict[str, list[dict]]) -> None:
        self.hits_by_query = hits_by_query

    def search(self, query: str, *, filters=None, top_k: int = 3) -> list[dict]:
        return self.hits_by_query.get(query, [])[:top_k]


class RetrieverTests(unittest.TestCase):
    def test_infer_metadata_filter(self) -> None:
        self.assertEqual(
            infer_metadata_filter("In Part 1 (Foundational Information), what is risk?"),
            {"section": "part1_foundational"},
        )
        self.assertEqual(
            infer_metadata_filter("In Part 2, what does MAP 1.1 state?"),
            {"section": "part2_core_profiles"},
        )
        self.assertEqual(
            infer_metadata_filter("In the Appendix, what are AI Design tasks?"),
            {"section": "appendix"},
        )
        self.assertEqual(
            infer_metadata_filter("Any question", explicit_filter={"section": "appendix"}),
            {"section": "appendix"},
        )
        self.assertIsNone(infer_metadata_filter("How is risk defined?"))

    def test_bm25_preserves_dotted_codes_and_respects_topk(self) -> None:
        chunks = [
            make_chunk("c1", 25, "Overview of the MAP function in the framework.", "part2_core_profiles"),
            make_chunk("c2", 31, "MAP 1.1: Prospective settings are documented before deployment.", "part2_core_profiles"),
            make_chunk("c3", 28, "GOVERN 1.7: Processes for decommissioning AI systems.", "part2_core_profiles"),
        ]
        bm25 = BM25Retriever(chunks, auto_filter=False)
        res = bm25.search("What does MAP 1.1 document?", top_k=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], "c2")
        self.assertEqual(len({r["id"] for r in res}), len(res))

    def test_hybrid_rrf_and_neighbor_expansion(self) -> None:
        c0 = make_chunk("p24:c000", 24, "Evaluations of AI RMF effectiveness. Framework users are expected to benefit from:")
        c1 = make_chunk("p24:c001", 24, "• enhanced processes for governing AI risk; • improved awareness of tradeoffs.")
        c2 = make_chunk("p10:c000", 10, "Risk measurement challenges and effectiveness evaluations.")
        chunks = [c0, c1, c2]

        bm25 = BM25Retriever(chunks, auto_filter=False)
        dense = FakeDenseRetriever({"expected benefit": [c0, c2]})

        hybrid_plain = HybridRRFRetriever(dense, bm25, chunks, enable_neighbor_expansion=False)
        res_plain = hybrid_plain.search("expected benefit", top_k=2)
        self.assertEqual([r["id"] for r in res_plain], ["p24:c000", "p10:c000"])

        hybrid_exp = HybridRRFRetriever(dense, bm25, chunks, enable_neighbor_expansion=True)
        res_exp = hybrid_exp.search("expected benefit", top_k=2)
        self.assertEqual([r["id"] for r in res_exp], ["p24:c000", "p24:c001"])


if __name__ == "__main__":
    unittest.main()
