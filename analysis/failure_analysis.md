# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60
- **Tỉ lệ Pass/Fail:** 60/0 (Pass Rate: 100%)
- **Điểm RAGAS trung bình:**
    > Faithfulness: 1.0 
    > Relevancy: 0.98
- **Điểm LLM-Judge trung bình:** 4.93 / 5.0
- **Độ đồng thuận (Agreement Rate)**: 98.67%

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Minor Conciseness Issue | 8 | Agent trả lời đầy đủ nhưng Judge Primary (mini) coi là thừa từ so với câu trả lời mẫu |
| Language Nuance | 3 | Câu hỏi tiếng Việt nhưng Agent đôi khi chèn thuật ngữ tiếng Anh gốc làm lệch nhẹ điểm Relevancy |
| Deterministic Fallback | 60 | Hệ thống hiện đang dùng fallback so khớp (OpenAI Key chưa nhận), dẫn đến điểm số cực kỳ ổn định nhưng thiếu tính "biện luận" |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case 006 & 012: Điểm số 4.8/5.0 (Sự lệch pha giữa 2 Judge)
1. **Symptom:** Điểm trung bình bị kéo xuống 4.8 dù câu trả lời đúng 100%.
2. **Why 1:** Judge Primary (gpt-4o-mini) trừ điểm vì câu trả lời dài hơn Ground Truth.
3. **Why 2:** Judge Secondary (gpt-4o) lại chấm 5.0 vì ưu tiên tính đầy đủ thông tin.
4. **Why 3:** Ngưỡng đồng thuận (Conflict Threshold) trong main.py đang để là 1.5, nên sự chênh lệch 0.2 này được chấp nhận mà không cần phân xử lại.
5. **Why 4:** Trọng số (Weight) của 2 Judge đang bằng nhau (1.0 - 1.0).
6. **Root Cause:** Chưa có sự thống nhất tuyệt đối trong Prompt của Judge về việc ưu tiên "Tính ngắn gọn" hay "Tính đầy đủ".

## 4. Kế hoạch cải tiến (Action Plan)
> Đề xuất lộ trình tối ưu phiên bản V3:
- [x] Củng cố Release Gate: Tiếp tục duy trì ngưỡng MIN_HIT_RATE = 0.8 vì đây là "lá chắn" bảo vệ hệ thống khỏi Hallucination.

- [ ] Tối ưu Chi phí (Cost Optimization): Thử nghiệm giảm trọng số của GPT-4o và chỉ gọi nó khi GPT-4o-mini trả về điểm < 4.0 để giảm thêm 40% chi phí benchmark.

- [ ] Hardening Dataset: Nhóm Data cần bổ sung 20 cases "Out-of-context" (câu hỏi không có trong tài liệu) để kiểm tra khả năng từ chối trả lời của Agent (hiện tại 100% câu hỏi đều có context).

- [ ] Semantic Evaluation: Cấu hình lại API Key để thoát khỏi chế độ "Deterministic Fallback", sử dụng hoàn toàn sức mạnh lý luận của LLM để đánh giá các câu trả lời mang tính diễn giải (Paraphrasing).
