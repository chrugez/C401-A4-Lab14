import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent

# Giả lập các components Expert
class ExpertEvaluator:
    async def score(self, case, resp): 
        # Giả lập tính toán Hit Rate và MRR
        return {
            "faithfulness": 0.9, 
            "relevancy": 0.8,
            "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
        }

class MultiModelJudge:
    async def evaluate_multi_judge(self, q, a, gt): 
        return {
            "final_score": 4.5, 
            "agreement_rate": 0.8,
            "reasoning": "Cả 2 model đồng ý đây là câu trả lời tốt."
        }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

def calculate_cost(total_cases, avg_tokens_per_case=1000):
    # Giả sử giá trung bình là 0.01$ cho 1000 tokens (Input + Output)
    price_per_1k_tokens = 0.01 
    total_tokens = total_cases * avg_tokens_per_case
    total_cost = (total_tokens / 1000) * price_per_1k_tokens
    
    return {
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_per_case": round(total_cost / total_cases, 5) if total_cases > 0 else 0
    }

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    MIN_SCORE_GAIN = 0.05       
    MIN_HIT_RATE = 0.8          
    
    avg_v1 = v1_summary["metrics"]["avg_score"]
    avg_v2 = v2_summary["metrics"]["avg_score"]
    hr_v2 = v2_summary["metrics"]["hit_rate"]
    # delta đã tính ở trên rồi nên không cần tính lại
    
    print("\n⚖️ --- PHÂN TÍCH QUYẾT ĐỊNH (RELEASE GATE) ---")
    
    if delta >= MIN_SCORE_GAIN and hr_v2 >= MIN_HIT_RATE:
        decision = "✅ RELEASE APPROVED"
        reason = f"Cải thiện {delta:.2f} điểm và Hit Rate đạt chuẩn ({hr_v2:.2%})"
    elif hr_v2 < MIN_HIT_RATE:
        decision = "🛑 BLOCK RELEASE"
        reason = f"Lỗi Retrieval: Hit Rate ({hr_v2:.2%}) thấp hơn ngưỡng cho phép!"
    else:
        decision = "🛑 BLOCK RELEASE"
        reason = f"Cải thiện điểm số ({delta:.2f}) không đáng kể hoặc bị lùi bước."

    print(f"Kết luận: {decision}")
    print(f"Lý do: {reason}")
    
    # Cực kỳ quan trọng: Gán quyết định vào summary trước khi lưu file
    v2_summary["gate_decision"] = {"status": decision, "reason": reason}
    # ========================================================

    # 2. Tính toán Token & Cost
    cost_info = calculate_cost(v2_summary["metadata"]["total"])
    
    # Đưa thông tin vào summary để nộp bài
    v2_summary["usage_statistics"] = cost_info

    print(f"💰 Chi phí ước tính: {cost_info['estimated_cost_usd']}$ ({cost_info['total_tokens']} tokens)")

    # --- ĐOẠN GHI FILE (VẪN GIỮ NGUYÊN) ---
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        # Giờ đây v2_summary đã có thêm trường gate_decision
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
        
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    # Thay thế đoạn print cũ bằng decision mới cho chuyên nghiệp
    print(f"\n🏁 Hoàn tất! Quyết định cuối cùng: {decision}")

if __name__ == "__main__":
    asyncio.run(main())
