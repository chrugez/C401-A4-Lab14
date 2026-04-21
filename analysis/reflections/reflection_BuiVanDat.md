**Họ và tên**: Bùi Văn Đạt - 2A202600355
**Agent Optimization (Member E)**: Tối ưu Agent để thông minh hơn qua System Prompt, Chunking Strategy, Query Transformation, và Agent Profile đa dạng.

1. **Engineering Contribution**:
-  **Module phụ trách**: 
> agent/main_agent.py (Linh hồn của Agent).

- **Đóng góp cụ thể**:
> Thêm System Prompt cho profile "base" và "optimized", với optimized dùng Few-shot Prompting để hướng dẫn Agent trả lời grounded và ngắn gọn.
> Triển khai Query Transformation để rewrite câu hỏi ngắn, cải thiện retrieval.
> Tăng top_k từ 3 lên 5 cho profile optimized để giảm lỗi retrieval.
> Tinh chỉnh logic trả lời dựa trên 5 Whys từ benchmark results, thêm "I don't know" để tránh hallucination.
> Phối hợp với nhóm Data để thử nghiệm Chunking Strategy (Sentence vs Paragraph).

2. **Technical Depth**:
> Faithfulness: Faithfulness đo lường mức độ Agent trả lời dựa trên context cung cấp. Tôi cải thiện bằng cách thêm instruction "Chỉ trả lời dựa trên thông tin" và Few-shot examples, giúp điểm tăng từ 4.42 lên 4.93.
> Chunking Strategy: Chia nhỏ tài liệu theo Sentence hay Paragraph ảnh hưởng đến context retrieval. Tôi đề xuất thử nghiệm cả hai để tìm strategy tối ưu cho Agent.
> Few-shot Prompting: Kỹ thuật đưa 2-3 ví dụ mẫu vào System Prompt để "thuần hóa" Agent theo ý Judge, giúp đạt điểm 5/5 về grounding và conciseness.
> Trade-off Retrieval & Generation: Tăng top_k giúp Agent có thêm context nhưng có thể tăng chi phí; tôi cân bằng bằng cách chỉ tăng cho optimized profile, với Hit Rate đạt 100%.

3. **Problem Solving**:
> Vấn đề: Điểm Judge thấp (4.42) do Faithfulness kém, câu trả lời dài, và retrieval sai trong một số case.
> Giải pháp: Phân tích 5 Whys từ reports/benchmark_results.json, tối ưu prompt với instruction nghiêm ngặt và examples, thêm query rewrite, tăng top_k. Kết quả: Điểm tăng lên 4.93, delta +0.51, Release APPROVED, với cost ước tính $0.2264.