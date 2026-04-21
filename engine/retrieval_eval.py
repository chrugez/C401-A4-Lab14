from typing import Any, Dict, List


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int | None = None) -> float:
        top_limit = top_k or self.top_k
        top_retrieved = retrieved_ids[:top_limit]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        for index, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                return 1.0 / index
        return 0.0

    def evaluate_case(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int | None = None) -> Dict[str, Any]:
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "top_k": top_k or self.top_k,
        }

    async def score(self, test_case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids = list(test_case.get("expected_retrieval_ids", []))
        retrieved_ids = list(response.get("retrieved_ids", []))
        return self.evaluate_case(expected_ids, retrieved_ids)

    async def evaluate_batch(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        scores = [
            self.evaluate_case(
                expected_ids=list(case.get("expected_retrieval_ids", [])),
                retrieved_ids=list(case.get("retrieved_ids", [])),
            )
            for case in dataset
        ]
        total = len(scores)
        return {
            "avg_hit_rate": sum(item["hit_rate"] for item in scores) / total,
            "avg_mrr": sum(item["mrr"] for item in scores) / total,
        }

