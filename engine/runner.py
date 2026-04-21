import asyncio
import time
from typing import Any, Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, batch_size: int = 5):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.batch_size = batch_size

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        response = await self.agent.query(test_case["question"], test_case=test_case)
        latency = time.perf_counter() - start_time

        retrieval_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            question=test_case["question"],
            answer=response["answer"],
            ground_truth=test_case["expected_answer"],
            contexts=response.get("contexts", []),
            metadata=test_case.get("metadata", {}),
        )

        usage = self._merge_usage(
            agent_usage=response.get("metadata", {}),
            judge_usage=judge_result.get("usage", {}),
        )

        return {
            "case_id": test_case.get("case_id"),
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "retrieval": retrieval_scores,
            "judge": judge_result,
            "latency": latency,
            "usage": usage,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict[str, Any]], batch_size: int | None = None) -> List[Dict[str, Any]]:
        active_batch_size = batch_size or self.batch_size
        results = []
        for index in range(0, len(dataset), active_batch_size):
            batch = dataset[index : index + active_batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    def _merge_usage(self, agent_usage: Dict[str, Any], judge_usage: Dict[str, Any]) -> Dict[str, Any]:
        prompt_tokens = int(agent_usage.get("prompt_tokens", 0)) + int(judge_usage.get("prompt_tokens", 0))
        completion_tokens = int(agent_usage.get("completion_tokens", 0)) + int(
            judge_usage.get("completion_tokens", 0)
        )
        total_tokens = int(agent_usage.get("tokens_used", 0)) + int(judge_usage.get("tokens_used", 0))
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_used": total_tokens,
        }

