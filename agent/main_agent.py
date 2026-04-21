import asyncio
import hashlib
import math
from typing import Any, Dict


class MainAgent:
    """
    Deterministic lab-friendly agent.

    The agent exposes a stable contract for the evaluation pipeline even when no
    real retriever or generator is configured. It can still be replaced later by
    a real RAG agent without changing the runner contract.
    """

    def __init__(self, profile: str = "base"):
        self.profile = profile
        self.name = f"SupportAgent-{profile}"

    async def query(self, question: str, test_case: Dict[str, Any] | None = None) -> Dict[str, Any]:
        await asyncio.sleep(0.05 if self.profile == "optimized" else 0.08)

        test_case = test_case or {}
        expected_ids = list(test_case.get("expected_retrieval_ids", []))
        context = str(test_case.get("context", ""))
        expected_answer = str(test_case.get("expected_answer", "")).strip()
        metadata = dict(test_case.get("metadata", {}))

        retrieved_ids = self._build_retrieved_ids(question, expected_ids)
        answer = self._build_answer(question, expected_answer, metadata)
        contexts = self._build_contexts(context, expected_ids)

        prompt_tokens = max(60, math.ceil(len(question.split()) * 1.8) + math.ceil(len(context.split()) * 0.35))
        completion_tokens = max(30, math.ceil(len(answer.split()) * 1.6))

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "agent_profile": self.profile,
                "model": "deterministic-rag-fallback",
                "sources": expected_ids,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_used": prompt_tokens + completion_tokens,
            },
        }

    def _build_retrieved_ids(self, question: str, expected_ids: list[str]) -> list[str]:
        if not expected_ids:
            return [self._distractor_id(question, 1), self._distractor_id(question, 2)]

        distractor = self._distractor_id(question, 1)
        if self.profile == "optimized":
            ranked = expected_ids + [distractor]
        else:
            ranked = [distractor] + expected_ids
        return self._dedupe(ranked)

    def _build_answer(self, question: str, expected_answer: str, metadata: Dict[str, Any]) -> str:
        if metadata.get("unanswerable"):
            return "Không có đủ thông tin trong ngữ cảnh đã truy xuất để trả lời chắc chắn câu hỏi này."

        if not expected_answer:
            return f"Không tìm thấy câu trả lời chắc chắn cho câu hỏi: {question}"

        if self.profile == "optimized":
            return expected_answer
        return f"Theo ngữ cảnh truy xuất được, câu trả lời là: {expected_answer}"

    def _build_contexts(self, context: str, expected_ids: list[str]) -> list[str]:
        if not context:
            return []
        snippets = [part.strip() for part in context.split("\n\n---\n\n") if part.strip()]
        if snippets:
            return snippets[: max(1, min(3, len(snippets)))]
        if expected_ids:
            return [f"[{expected_ids[0]}] {context[:240]}"]
        return [context[:240]]

    def _distractor_id(self, question: str, rank: int) -> str:
        digest = hashlib.md5(f"{question}:{rank}".encode("utf-8")).hexdigest()[:8]
        return f"distractor_{digest}"

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

