import asyncio
import json
import os
import time
from typing import Any, Dict, List, Tuple

from agent.main_agent import MainAgent
from engine.llm_judge import JudgeProfile, LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


DEFAULT_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "5"))
DEFAULT_TOP_K = int(os.getenv("EVAL_TOP_K", "3"))
DEFAULT_CONFLICT_THRESHOLD = float(os.getenv("JUDGE_CONFLICT_THRESHOLD", "1.5"))
PRICE_PER_1K_TOKENS_USD = float(os.getenv("EVAL_PRICE_PER_1K_TOKENS", "0.01"))


def load_dataset() -> List[Dict[str, Any]]:
    dataset_path = "data/golden_set.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Missing data/golden_set.jsonl. Run 'python data/synthetic_gen.py' first.")

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = [json.loads(line) for line in dataset_file if line.strip()]

    if not dataset:
        raise ValueError("data/golden_set.jsonl is empty. Generate at least one test case.")
    return dataset


def build_judge() -> LLMJudge:
    profiles = [
        JudgeProfile(name="judge_openai_primary", provider="openai", model="gpt-4o-mini", weight=1.0),
        JudgeProfile(name="judge_openai_secondary", provider="openai", model="gpt-4o", weight=1.0),
    ]
    return LLMJudge(
        profiles=profiles,
        conflict_threshold=DEFAULT_CONFLICT_THRESHOLD,
        fallback_enabled=True,
    )


def calculate_cost(total_tokens: int) -> Dict[str, float]:
    total_cost = (total_tokens / 1000) * PRICE_PER_1K_TOKENS_USD
    return {
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_per_case": 0.0,
    }


def summarize_results(agent_version: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    total_tokens = sum(int(result["usage"].get("tokens_used", 0)) for result in results)
    cost_info = calculate_cost(total_tokens)
    cost_info["cost_per_case"] = round(cost_info["estimated_cost_usd"] / total, 5) if total else 0.0

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(sum(result["judge"]["final_score"] for result in results) / total, 4),
            "hit_rate": round(sum(result["retrieval"]["hit_rate"] for result in results) / total, 4),
            "mrr": round(sum(result["retrieval"]["mrr"] for result in results) / total, 4),
            "agreement_rate": round(sum(result["judge"]["agreement_rate"] for result in results) / total, 4),
            "avg_latency": round(sum(result["latency"] for result in results) / total, 4),
            "pass_rate": round(sum(1 for result in results if result["status"] == "pass") / total, 4),
        },
        "usage_statistics": cost_info,
    }
    return summary


def apply_release_gate(v1_summary: Dict[str, Any], v2_summary: Dict[str, Any]) -> Dict[str, str]:
    min_score_gain = 0.05
    min_hit_rate = 0.8

    avg_v1 = v1_summary["metrics"]["avg_score"]
    avg_v2 = v2_summary["metrics"]["avg_score"]
    hr_v2 = v2_summary["metrics"]["hit_rate"]
    delta = avg_v2 - avg_v1

    if delta >= min_score_gain and hr_v2 >= min_hit_rate:
        decision = "RELEASE APPROVED"
        reason = f"Score improved by {delta:.2f} and Hit Rate meets threshold ({hr_v2:.2%})."
    elif hr_v2 < min_hit_rate:
        decision = "BLOCK RELEASE"
        reason = f"Retrieval issue: Hit Rate ({hr_v2:.2%}) is below the threshold."
    else:
        decision = "BLOCK RELEASE"
        reason = f"Score gain ({delta:.2f}) is not significant or regressed."

    return {"status": decision, "reason": reason}


async def run_benchmark_with_results(agent_version: str, profile: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"Starting benchmark for {agent_version}...")

    dataset = load_dataset()
    runner = BenchmarkRunner(
        agent=MainAgent(profile=profile),
        evaluator=RetrievalEvaluator(top_k=DEFAULT_TOP_K),
        judge=build_judge(),
        batch_size=DEFAULT_BATCH_SIZE,
    )
    results = await runner.run_all(dataset)
    summary = summarize_results(agent_version, results)
    return results, summary


async def run_benchmark(agent_version: str, profile: str) -> Dict[str, Any]:
    _, summary = await run_benchmark_with_results(agent_version, profile)
    return summary


def persist_reports(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=2)


async def main() -> None:
    try:
        v1_summary = await run_benchmark("Agent_V1_Base", "base")
        v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", "optimized")
    except (FileNotFoundError, ValueError) as error:
        print(f"Benchmark failed: {error}")
        return

    print("\n=== KET QUA SO SANH (REGRESSION) ===")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    gate_decision = apply_release_gate(v1_summary, v2_summary)
    v2_summary["gate_decision"] = gate_decision

    print("\n=== PHAN TICH QUYET DINH (RELEASE GATE) ===")
    print(f"Ket luan: {gate_decision['status']}")
    print(f"Ly do: {gate_decision['reason']}")
    print(
        "Chi phi uoc tinh: "
        f"{v2_summary['usage_statistics']['estimated_cost_usd']}$ "
        f"({v2_summary['usage_statistics']['total_tokens']} tokens)"
    )

    persist_reports(v2_summary, v2_results)
    print(f"\nCompleted. Final decision: {gate_decision['status']}")


if __name__ == "__main__":
    asyncio.run(main())
