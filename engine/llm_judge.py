from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import AsyncOpenAI


@dataclass
class JudgeProfile:
    name: str
    provider: str
    model: str
    weight: float = 1.0
    enabled: bool = True
class LLMJudge:
    def __init__(
        self,
        profiles: List[JudgeProfile] | None = None,
        conflict_threshold: float = 1.5,
        fallback_enabled: bool = True,
    ):
        load_dotenv()
        self.profiles = profiles or self._default_profiles()
        self.conflict_threshold = conflict_threshold
        self.fallback_enabled = fallback_enabled
        self.openai_api_key = self._get_openai_api_key()
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.rubrics = {
            "accuracy": "Answer should match the ground truth or correctly refuse unsupported claims.",
            "grounding": "Answer should stay within retrieved context and avoid hallucination.",
            "clarity": "Answer should be concise and professional.",
        }
    def _default_profiles(self) -> List[JudgeProfile]:
        return [
            JudgeProfile(name="judge_openai_primary", provider="openai", model="gpt-4o-mini", weight=1.0),
            JudgeProfile(name="judge_openai_secondary", provider="openai", model="gpt-4o", weight=1.0),
        ]
    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        contexts = contexts or []

        active_profiles = [profile for profile in self.profiles if profile.enabled]
        if len(active_profiles) < 2:
            raise ValueError("Multi-judge consensus requires at least two enabled judge profiles.")

        judge_outputs = await asyncio.gather(
            *[
                self._evaluate_profile(profile, question, answer, ground_truth, contexts, metadata)
                for profile in active_profiles
            ]
        )
        scores = {output["name"]: output["score"] for output in judge_outputs}
        weighted_average = self._weighted_average(judge_outputs)
        agreement_rate = self._agreement_rate(list(scores.values()))
        max_gap = max(scores.values()) - min(scores.values())
        conflict_detected = max_gap >= self.conflict_threshold

        if conflict_detected:
            arbitration = self._arbitrate(judge_outputs, question, answer, ground_truth, metadata)
            final_score = arbitration["final_score"]
            consensus_method = arbitration["consensus_method"]
            conflict_reason = arbitration["conflict_reason"]
            reasoning = arbitration["reasoning"]
        else:
            final_score = round(weighted_average, 2)
            consensus_method = "weighted_average"
            conflict_reason = ""
            reasoning = "Judges stayed within the disagreement threshold, so the weighted average was accepted."

        usage = {
            "prompt_tokens": sum(output["usage"]["prompt_tokens"] for output in judge_outputs),
            "completion_tokens": sum(output["usage"]["completion_tokens"] for output in judge_outputs),
        }
        usage["tokens_used"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "individual_scores": scores,
            "individual_rationales": {output["name"]: output["reasoning"] for output in judge_outputs},
            "conflict_detected": conflict_detected,
            "conflict_reason": conflict_reason,
            "consensus_method": consensus_method,
            "reasoning": reasoning,
            "judges": judge_outputs,
            "usage": usage,
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, Any]:
        same_length = len(response_a.split()) == len(response_b.split())
        return {
            "position_bias_detected": False,
            "reason": "Position bias check is a placeholder for future extension.",
            "same_length": same_length,
        }

    async def _evaluate_profile(
        self,
        profile: JudgeProfile,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        provider_available = self._provider_available(profile.provider)
        if provider_available:
            try:
                openai_result = await self._evaluate_with_openai(
                    profile=profile,
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                    contexts=contexts,
                    metadata=metadata,
                )
                openai_result["fallback_used"] = False
                return openai_result
            except Exception as error:
                if not self.fallback_enabled:
                    raise
                fallback_reason = f"OpenAI judge call failed, using deterministic fallback: {error}"
        else:
            fallback_reason = "OpenAI API key not found, using deterministic fallback."

        score, reasoning = self._deterministic_score(profile, question, answer, ground_truth, contexts, metadata)
        usage = self._estimate_usage(question, answer, ground_truth, contexts)
        return {
            "name": profile.name,
            "provider": profile.provider,
            "model": profile.model,
            "weight": profile.weight,
            "score": score,
            "reasoning": f"{reasoning} {fallback_reason}",
            "fallback_used": True,
            "usage": usage,
        }

    async def _evaluate_with_openai(
        self,
        profile: JudgeProfile,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        if profile.provider != "openai" or not self.openai_client:
            raise RuntimeError(f"Unsupported judge provider or missing client: {profile.provider}")

        prompt = self._build_openai_prompt(question, answer, ground_truth, contexts, metadata)
        response = await self.openai_client.chat.completions.create(
            model=profile.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge for RAG answers. "
                        "Score the answer from 1 to 5 using only the provided question, answer, ground truth, and context. "
                        "Return valid JSON only with keys: score, reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = self._extract_json_object(content)
        score = self._coerce_score(payload.get("score"))
        reasoning = str(payload.get("reasoning", "")).strip() or "OpenAI judge returned no rationale."
        usage = self._usage_from_response(response, question, answer, ground_truth, contexts)

        return {
            "name": profile.name,
            "provider": profile.provider,
            "model": profile.model,
            "weight": profile.weight,
            "score": score,
            "reasoning": reasoning,
            "usage": usage,
        }

    def _deterministic_score(
        self,
        profile: JudgeProfile,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: List[str],
        metadata: Dict[str, Any],
    ) -> tuple[float, str]:
        normalized_answer = self._normalize(answer)
        normalized_ground_truth = self._normalize(ground_truth)
        exact_match = normalized_answer == normalized_ground_truth and normalized_ground_truth != ""
        contains_ground_truth = normalized_ground_truth != "" and normalized_ground_truth in normalized_answer
        unanswerable = bool(metadata.get("unanswerable"))
        grounded = any(normalized_ground_truth in self._normalize(context) for context in contexts) if normalized_ground_truth else False

        score = 2.2
        if exact_match:
            score = 5.0
        elif contains_ground_truth:
            score = 4.4
        elif unanswerable and any(token in normalized_answer for token in ["không", "khong", "insufficient", "not enough"]):
            score = 4.2
        elif grounded:
            score = 3.6

        jitter = self._profile_jitter(profile.name, question, answer)
        final_score = max(1.0, min(5.0, round(score + jitter, 2)))
        reasoning = (
            f"{profile.name} judged the answer on accuracy/grounding. "
            f"exact_match={exact_match}, contains_ground_truth={contains_ground_truth}, grounded={grounded}."
        )
        return final_score, reasoning

    def _profile_jitter(self, profile_name: str, question: str, answer: str) -> float:
        digest = hashlib.md5(f"{profile_name}|{question}|{answer}".encode("utf-8")).hexdigest()
        bucket = int(digest[:2], 16) % 3
        return {0: -0.2, 1: 0.0, 2: 0.2}[bucket]

    def _agreement_rate(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        if len(scores) == 1:
            return 1.0
        max_gap = max(scores) - min(scores)
        return round(max(0.0, 1.0 - (max_gap / 4.0)), 2)

    def _weighted_average(self, judge_outputs: List[Dict[str, Any]]) -> float:
        total_weight = sum(output["weight"] for output in judge_outputs)
        if total_weight <= 0:
            return 0.0
        return sum(output["score"] * output["weight"] for output in judge_outputs) / total_weight

    def _arbitrate(
        self,
        judge_outputs: List[Dict[str, Any]],
        question: str,
        answer: str,
        ground_truth: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        weighted_average = self._weighted_average(judge_outputs)
        accuracy_bonus = 0.3 if self._normalize(ground_truth) in self._normalize(answer) and ground_truth.strip() else 0.0
        refusal_bonus = 0.2 if metadata.get("unanswerable") and "không" in self._normalize(answer) else 0.0
        final_score = max(1.0, min(5.0, round(weighted_average + accuracy_bonus + refusal_bonus, 2)))
        return {
            "final_score": final_score,
            "consensus_method": "auto_arbitration",
            "conflict_reason": "Judge score gap exceeded the configured threshold.",
            "reasoning": (
                "A deterministic arbiter resolved the disagreement by starting from the weighted "
                "average and adjusting for explicit grounding/refusal behavior."
            ),
        }

    def _provider_available(self, provider: str) -> bool:
        if provider == "openai":
            return bool(self.openai_client)
        return False

    def _get_openai_api_key(self) -> str | None:
        for env_name in ("OPENAI_API_KEY", "OPENAI_KEY", "open_ai_key"):
            value = os.getenv(env_name)
            if value:
                return value
        return None

    def _build_openai_prompt(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: List[str],
        metadata: Dict[str, Any],
    ) -> str:
        context_block = "\n\n".join(contexts[:3]) if contexts else "(no retrieved context)"
        rubric_lines = "\n".join(f"- {name}: {description}" for name, description in self.rubrics.items())
        payload = {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "retrieved_contexts": context_block,
            "metadata": metadata,
            "rubric": rubric_lines,
            "instructions": [
                "Use integer or decimal score between 1 and 5.",
                "Prefer lower scores when the answer is not grounded.",
                "If the task is unanswerable, reward a correct refusal.",
                'Respond as JSON like {"score": 4.5, "reasoning": "short explanation"}.',
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _extract_json_object(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Judge response does not contain a JSON object.")
        return json.loads(text[start : end + 1])

    def _coerce_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid judge score: {value}") from error
        return max(1.0, min(5.0, round(score, 2)))

    def _usage_from_response(
        self,
        response: Any,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: List[str],
    ) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return self._estimate_usage(question, answer, ground_truth, contexts)

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_used": total_tokens,
        }

    def _estimate_usage(self, question: str, answer: str, ground_truth: str, contexts: List[str]) -> Dict[str, int]:
        prompt_tokens = max(40, len((question + " " + ground_truth + " " + " ".join(contexts)).split()))
        completion_tokens = max(20, len(answer.split()) // 2 + 20)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_used": prompt_tokens + completion_tokens,
        }

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())
