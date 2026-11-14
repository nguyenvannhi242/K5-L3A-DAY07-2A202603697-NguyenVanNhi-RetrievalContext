"""Unit tests for DeepSeekController decision normalization and agent integration."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))

from agentic import run_agent  # noqa: E402
from deepseek_agent import DeepSeekController, normalize_decision, parse_json_object  # noqa: E402


class FakeRetriever:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def search(self, query: str, *, filters=None, top_k: int = 3) -> list[dict]:
        return self.rows[:top_k]


class DeepSeekAgentTests(unittest.TestCase):
    def test_normalize_decision_filters_invalid_ids_and_appends_tags(self) -> None:
        raw = {
            "action": "answer",
            "answer": "Risk has two components: likelihood and harm.",
            "citations": ["c1", "not-in-context"],
        }
        norm = normalize_decision(raw, valid_context_ids=["c1", "c2"])
        self.assertEqual(norm["action"], "answer")
        self.assertEqual(norm["citations"], ["c1"])
        self.assertIn("[c1]", norm["answer"])

    def test_normalize_decision_maps_abstain_action(self) -> None:
        norm = normalize_decision({"action": "abstain"}, valid_context_ids=["c1"])
        self.assertEqual(norm["action"], "answer")
        self.assertEqual(norm["citations"], [])

        norm2 = normalize_decision(
            {"action": "answer", "answer": "Không đủ bằng chứng để trả lời. [c1]", "citations": ["c1"]},
            valid_context_ids=["c1"],
        )
        self.assertEqual(norm2["citations"], [])

    def test_parse_json_object_extracts_first_json_block(self) -> None:
        raw = 'Here is JSON:\n{"action": "search", "query": "test"}\n{"extra": 1}'
        self.assertEqual(parse_json_object(raw), {"action": "search", "query": "test"})

    def test_missing_api_key_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            DeepSeekController(api_key="")

    def test_controller_integration_answer_and_abstain(self) -> None:
        responses = iter([
            {
                "choices": [{"message": {"content": json.dumps({"action": "search", "query": "risk factors"})}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
            {
                "choices": [{"message": {"content": json.dumps({
                    "action": "answer",
                    "answer": "Negative impact and likelihood [c1]",
                    "citations": ["c1"]
                })}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
            },
        ])
        controller = DeepSeekController(api_key="test-key", chat_fn=lambda **_: next(responses))
        retriever = FakeRetriever([
            {"id": "c1", "source_id": "doc", "source": "doc.pdf", "page": 9, "text": "Negative impact and likelihood.", "metadata": {}, "score": 0.9}
        ])
        res = run_agent("What are the two risk factors?", controller, retriever)
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["c1"])
        self.assertEqual(controller.llm_calls, 2)
        self.assertEqual(controller.total_tokens, 350)


if __name__ == "__main__":
    unittest.main()
