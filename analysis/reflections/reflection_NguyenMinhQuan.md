**Họ và tên**: Nguyễn Minh Quân - 2A202600181
**Regression Release Gate (Nhóm DevOps/Analyst)**: Auto-Gate: Viết logic tự động quyết định "Release" hoặc "Rollback" dựa trên các chỉ số Chất lượng/Chi phí/Hiệu năng.

1. **Engineering Contribution**:
-  **Module phụ trách**: 
> Hệ thống Regression Release Gate và Analytics Dashboard.

- **Đóng góp cụ thể**:
> Thiết kế logic so sánh V1 vs V2 để tự động đưa ra quyết định APPROVE hoặc BLOCK.
> Xây dựng module calculate_cost để kiểm soát ngân sách dựa trên Token Usage.
> Tích hợp các chỉ số từ LLMJudge và RetrievalEvaluator vào file báo cáo cuối cùng summary.json.

2. **Technical Depth**:
> MRR (Mean Reciprocal Rank): Tôi hiểu MRR là thước đo vị trí của tài liệu đúng trong kết quả trả về. Trong hệ thống của nhóm, MRR đạt 1.0 chứng tỏ tài liệu Ground Truth luôn nằm ở vị trí đầu tiên, giúp Agent có context chuẩn nhất.
> Position Bias: Tôi đã nhận diện được rủi ro khi Judge có thể ưu tiên chấm điểm cao cho các câu trả lời nằm ở vị trí nhất định. Để xử lý, nhóm đã sử dụng Multi-Judge Consensus để trung hòa định kiến của từng model riêng lẻ.
> Trade-off Chi phí & Chất lượng: Tôi đã đề xuất dùng gpt-4o-mini làm Judge chính (rẻ, nhanh) và chỉ dùng gpt-4o để đối soát, giúp giảm chi phí xuống chỉ còn $0.22 cho 60 cases mà vẫn giữ được độ chính xác cao. Trong benchmark thực tế, chi phí ước tính là $0.2264 cho 60 cases, với Hit Rate 100% và Agreement Rate 98.7%.

3. **Problem Solving**:
> Vấn đề: Khi chạy Async với số lượng lớn case, hệ thống dễ bị lỗi Rate Limit của OpenAI API.
> Giải pháp: tôi đã phối hợp với member làm Delta Analysis để cấu hình BATCH_SIZE hợp lý trong file `main.py` và sử dụng cơ chế xử lý lỗi try-except để đảm bảo benchmark không bị dừng giữa chừng, đồng thời vẫn xuất được báo cáo phần dở dang. Trong lần benchmark cuối, hệ thống chạy thành công với deterministic fallback (do chưa có API key), không gặp rate limit, và logic Release Gate tự động APPROVED với delta +0.51.