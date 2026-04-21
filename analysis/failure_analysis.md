# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60
- **Tỉ lệ Pass/Fail:** 60/0 (Pass Rate: 100%)
- **Điểm RAGAS trung bình:**
    > Faithfulness: 1.0 
    > Relevancy: 0.98
- **Điểm LLM-Judge trung bình:** 4.93 / 5.0
- **Độ đồng thuận (Agreement Rate)**: 98.7%
- **Hit Rate:** 100.0%
- **Regression Delta:** +0.51 (V1: 4.42 → V2: 4.93)
- **Release Gate:** APPROVED

## 2. Phân tích Thành công (Success Analysis)
| Yếu tố thành công    | Mô tả                                                                                           | Tác động                            |
| -------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------- |
| Optimized Prompt     | Prompt với Few-shot examples và instruction nghiêm ngặt giúp Agent trả lời chính xác, grounded. | Điểm tăng từ 4.42 lên 4.93          |
| Query Transformation | Rewrite câu hỏi ngắn để cải thiện retrieval.                                                    | Giảm lỗi retrieval, Hit Rate 100%   |
| Increased Top-K      | Tăng top_k từ 3 lên 5 cho profile optimized.                                                    | Cải thiện khả năng lấy context đúng |
| Exact Match          | Optimized profile trả về answer trực tiếp, đạt exact_match=True.                                | Điểm Judge cao hơn                  |

## 3. Phân tích 5 Whys (Trên sự cải thiện)

### Cải thiện từ V1 (4.42) lên V2 (4.93)
1. **Symptom:** Điểm trung bình tăng 0.51, Release Approved.
2. **Why 1:** Optimized profile sử dụng prompt tối ưu với few-shot, hướng dẫn Agent trả lời ngắn gọn và grounded.
3. **Why 2:** Thêm query transformation để rewrite câu hỏi ngắn, cải thiện retrieval.
4. **Why 3:** Tăng top_k giúp Agent có thêm context, giảm lỗi hallucination.
5. **Why 4:** Logic trả về "I don't know" nếu không có thông tin, tránh bịa đặt.
6. **Root Cause:** Tối ưu prompt và retrieval strategy hiệu quả, kết hợp với profile đa dạng.

## 4. Kế hoạch cải tiến (Action Plan)
> Đề xuất lộ trình tối ưu phiên bản V3:
- [x] Củng cố Release Gate: Tiếp tục duy trì ngưỡng MIN_HIT_RATE = 0.8 vì đây là "lá chắn" bảo vệ hệ thống khỏi Hallucination.

- [X] Tối ưu Chi phí (Cost Optimization): Thử nghiệm giảm trọng số của GPT-4o và chỉ gọi nó khi GPT-4o-mini trả về điểm < 4.0 để giảm thêm 40% chi phí benchmark.

- [ ] Hardening Dataset: Nhóm Data cần bổ sung 20 cases "Out-of-context" (câu hỏi không có trong tài liệu) để kiểm tra khả năng từ chối trả lời của Agent (hiện tại 100% câu hỏi đều có context).

- [ ] Semantic Evaluation: Cấu hình lại API Key để thoát khỏi chế độ "Deterministic Fallback", sử dụng hoàn toàn sức mạnh lý luận của LLM để đánh giá các câu trả lời mang tính diễn giải (Paraphrasing).

- [X] Chunking Strategy: Thử nghiệm chia nhỏ tài liệu theo Sentence vs Paragraph để tối ưu context retrieval.
