import asyncio
import unittest

from agent.main_agent import MainAgent
from engine.llm_judge import JudgeProfile, LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


class RetrievalEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = RetrievalEvaluator(top_k=3)

    def test_hit_rate_when_expected_id_in_top_k(self) -> None:
        self.assertEqual(1.0, self.evaluator.calculate_hit_rate(["doc_b"], ["doc_a", "doc_b", "doc_c"]))

    def test_hit_rate_when_expected_id_outside_top_k(self) -> None:
        self.assertEqual(0.0, self.evaluator.calculate_hit_rate(["doc_d"], ["doc_a", "doc_b", "doc_c", "doc_d"]))

    def test_hit_rate_with_multiple_ground_truth_ids(self) -> None:
        self.assertEqual(1.0, self.evaluator.calculate_hit_rate(["doc_x", "doc_c"], ["doc_a", "doc_b", "doc_c"]))

    def test_hit_rate_with_no_retrieved_ids(self) -> None:
        self.assertEqual(0.0, self.evaluator.calculate_hit_rate(["doc_a"], []))

    def test_mrr_rank_one(self) -> None:
        self.assertEqual(1.0, self.evaluator.calculate_mrr(["doc_a"], ["doc_a", "doc_b"]))

    def test_mrr_rank_three(self) -> None:
        self.assertAlmostEqual(1 / 3, self.evaluator.calculate_mrr(["doc_c"], ["doc_a", "doc_b", "doc_c"]))

    def test_mrr_no_match(self) -> None:
        self.assertEqual(0.0, self.evaluator.calculate_mrr(["doc_z"], ["doc_a", "doc_b", "doc_c"]))


class JudgeConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        profiles = [
            JudgeProfile(name="judge_a", provider="openai", model="gpt-4o-mini"),
            JudgeProfile(name="judge_b", provider="openai", model="gpt-4o-mini"),
        ]
        self.judge = LLMJudge(profiles=profiles, conflict_threshold=1.5, fallback_enabled=True)

    def test_consensus_for_exact_match(self) -> None:
        result = asyncio.run(
            self.judge.evaluate_multi_judge(
                question="Q",
                answer="Paris",
                ground_truth="Paris",
                contexts=["Paris is the capital of France."],
            )
        )
        self.assertGreaterEqual(result["final_score"], 4.8)
        self.assertFalse(result["conflict_detected"])

    def test_consensus_for_small_disagreement(self) -> None:
        result = asyncio.run(
            self.judge.evaluate_multi_judge(
                question="Q",
                answer="Theo ngữ cảnh truy xuất được, câu trả lời là: Paris",
                ground_truth="Paris",
                contexts=["Paris is the capital of France."],
            )
        )
        self.assertFalse(result["conflict_detected"])
        self.assertIn(result["consensus_method"], {"weighted_average", "auto_arbitration"})

    def test_arbitration_for_large_disagreement(self) -> None:
        low_threshold_judge = LLMJudge(
            profiles=[
                JudgeProfile(name="judge_a", provider="openai", model="gpt-4o-mini"),
                JudgeProfile(name="judge_b", provider="openai", model="gpt-4o-mini"),
            ],
            conflict_threshold=0.1,
            fallback_enabled=True,
        )
        result = asyncio.run(
            low_threshold_judge.evaluate_multi_judge(
                question="Q",
                answer="Paris",
                ground_truth="Paris",
                contexts=["Paris is the capital of France."],
            )
        )
        self.assertTrue(result["conflict_detected"])
        self.assertEqual("auto_arbitration", result["consensus_method"])

    def test_fallback_runs_without_provider_key(self) -> None:
        result = asyncio.run(
            self.judge.evaluate_multi_judge(
                question="Q",
                answer="Paris",
                ground_truth="Paris",
                contexts=["Paris is the capital of France."],
            )
        )
        self.assertTrue(all(judge["fallback_used"] for judge in result["judges"]))


class RunnerIntegrationTests(unittest.TestCase):
    def test_runner_returns_required_fields(self) -> None:
        runner = BenchmarkRunner(
            agent=MainAgent(profile="optimized"),
            evaluator=RetrievalEvaluator(top_k=3),
            judge=LLMJudge(
                profiles=[
                    JudgeProfile(name="judge_a", provider="openai", model="gpt-4o-mini"),
                    JudgeProfile(name="judge_b", provider="openai", model="gpt-4o-mini"),
                ]
            ),
            batch_size=2,
        )
        dataset = [
            {
                "case_id": "golden_case_001",
                "question": "Capital of France?",
                "expected_answer": "Paris",
                "expected_retrieval_ids": ["doc_paris"],
                "context": "[doc_paris] Paris is the capital of France.",
                "metadata": {"difficulty": "easy", "type": "fact-check", "unanswerable": False},
            },
            {
                "case_id": "golden_case_002",
                "question": "Capital of Germany?",
                "expected_answer": "Berlin",
                "expected_retrieval_ids": ["doc_berlin"],
                "context": "[doc_berlin] Berlin is the capital of Germany.",
                "metadata": {"difficulty": "easy", "type": "fact-check", "unanswerable": False},
            },
        ]

        results = asyncio.run(runner.run_all(dataset))
        self.assertEqual(2, len(results))
        self.assertIn("retrieval", results[0])
        self.assertIn("judge", results[0])
        self.assertIn("usage", results[0])
        self.assertIn("status", results[0])
        self.assertIn("hit_rate", results[0]["retrieval"])
        self.assertIn("agreement_rate", results[0]["judge"])


if __name__ == "__main__":
    unittest.main()
