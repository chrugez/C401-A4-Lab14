**Họ và tên**: Hoàng Quốc Chung - 2A202600070
**Metrics Specialist (Nhóm Data/AI Evaluation)**: Code logic tính Hit Rate và MRR. Đảm bảo phần Retrieval Eval được tách biệt khỏi Generation Eval.

1. **Engineering Contribution**:
- **Module phụ trách**:
> Module `engine/retrieval_eval.py` và phần tích hợp Retrieval Eval trong pipeline benchmark.

- **Đóng góp cụ thể**:
> Tôi triển khai hàm `calculate_hit_rate()` để kiểm tra xem ít nhất một `expected_retrieval_id` có xuất hiện trong top-k tài liệu được truy xuất hay không. Metric này giúp nhóm đánh giá nhanh Retriever có lấy đúng chunk nền tảng cho câu trả lời hay không.
>
> Tôi triển khai hàm `calculate_mrr()` để đo chất lượng thứ hạng của retrieval bằng cách tìm vị trí đầu tiên của tài liệu ground truth trong danh sách `retrieved_ids`. Khi tài liệu đúng xuất hiện càng sớm, MRR càng cao, qua đó phản ánh rõ hiệu quả ranking thay vì chỉ biết có hit hay không.
>
> Tôi xây hàm `evaluate_case()` và `score()` để đóng gói retrieval metrics thành một bước đánh giá độc lập, nhận đầu vào là `expected_retrieval_ids` từ golden set và `retrieved_ids` từ Agent. Thiết kế này giúp Retrieval Eval có contract riêng, không phụ thuộc vào điểm chấm của Judge cho phần generation.
>
> Tôi phối hợp tích hợp Retrieval Eval vào `engine/runner.py` theo đúng thứ tự pipeline: Agent truy xuất và sinh câu trả lời, Retrieval Evaluator tính `hit_rate`/`mrr`, sau đó Multi-Judge mới chấm chất lượng câu trả lời. Nhờ đó, báo cáo cuối cùng phân biệt rõ lỗi ở bước lấy context với lỗi ở bước sinh đáp án.

2. **Technical Depth**:
> Hit Rate là metric nhị phân ở mức case: nếu một trong các `expected_retrieval_ids` xuất hiện trong top-k kết quả truy xuất thì case đó được tính là hit. Tôi dùng metric này để trả lời câu hỏi nền tảng nhất của hệ thống RAG: Retriever có tìm được đúng chunk không.
>
> MRR (Mean Reciprocal Rank) bổ sung chiều sâu cho Hit Rate vì nó không chỉ quan tâm tài liệu đúng có xuất hiện hay không, mà còn quan tâm nó xuất hiện ở vị trí thứ mấy. Nếu tài liệu đúng đứng đầu danh sách thì điểm là 1.0; nếu đứng thứ hai thì 0.5; càng xuống thấp thì điểm càng giảm. Điều này rất quan trọng vì một Retriever có hit nhưng xếp sai thứ hạng vẫn có thể làm Generator ưu tiên nhầm context.
>
> Tôi chủ động tách Retrieval Eval khỏi Generation Eval vì hai lớp lỗi này có bản chất khác nhau. Nếu retrieval sai mà chỉ nhìn final answer score, nhóm sẽ khó xác định nguyên nhân gốc nằm ở vector search, ranking hay prompt generation. Khi có module `RetrievalEvaluator` riêng và trường `retrieval` riêng trong benchmark result, nhóm có thể khoanh lỗi chính xác hơn trong quá trình failure analysis.

3. **Problem Solving**:
> Vấn đề kỹ thuật lớn nhất là tránh để Retrieval Eval bị “trộn” với Generation Eval trong cùng một điểm số tổng quát. Nếu chỉ nhìn `final_score` từ Judge thì một câu trả lời sai có thể đến từ hai nguyên nhân hoàn toàn khác nhau: Retriever lấy sai tài liệu, hoặc Generator diễn giải sai dù đã có đúng context.
>
> Giải pháp tôi thực hiện là tách module đánh giá retrieval thành lớp riêng `RetrievalEvaluator`, với đầu vào và đầu ra tách biệt khỏi Judge. Trong `BenchmarkRunner`, phần retrieval được tính trước và lưu vào khóa `retrieval`, còn phần generation được đánh giá riêng ở khóa `judge`. Cách làm này giúp báo cáo benchmark minh bạch hơn và hỗ trợ trực tiếp cho bước phân tích nguyên nhân gốc.
>
> Ngoài ra, tôi bổ sung `evaluate_batch()` để có thể tổng hợp `avg_hit_rate` và `avg_mrr` ở cấp độ toàn bộ benchmark. Nhờ đó, nhóm không chỉ biết từng case retrieval ra sao mà còn có thể theo dõi chất lượng retrieval theo từng phiên bản Agent khi chạy regression.
