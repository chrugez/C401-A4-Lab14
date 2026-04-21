**Họ và tên**: Hoàng Thái Dương - 2A202600073
**Retrieval & SDG (Nhóm Data)**: Retrieval Master: Làm sạch file CSV, chuẩn bị metadata cho tài liệu, gắn Ground Truth ID. Viết file hướng dẫn các case khó (Edge cases) để "test thử độ bền" của Agent.

1. **Engineering Contribution**:
-  **Module phụ trách**: 
> Pipeline chuẩn hóa dữ liệu Retrieval trong `data/QA_dataset/`, bao gồm làm sạch nguồn SQuAD, sinh metadata theo chunk, gắn Ground Truth ID cho từng test case, và hoàn thiện tài liệu `analysis/HARD_CASES_GUIDE.md`/`data/HARD_CASES_GUIDE.md` để định nghĩa các edge cases quan trọng.

- **Đóng góp cụ thể**:
> Tôi tham gia làm sạch và hợp nhất dữ liệu từ hai nguồn `SQuAD-v1.1.csv` và `SQuAD-v1.2.csv`, chuẩn hóa lại schema để tạo ra các artifact trung gian như `documents_cleaned.csv`.
> Tôi chuẩn bị metadata tài liệu trong `document_metadata.csv` với các trường như `chunk_id`, `doc_id`, `title`, `path`, `qa_count`, `word_count`, `char_count` và `preview`, giúp nhóm Retrieval/Eval có thể truy vết chính xác mỗi chunk.
> Tôi gắn Ground Truth ID cho từng case benchmark thông qua `ground_truth_mapping.csv` và đảm bảo các trường `expected_retrieval_ids` trong `golden_set.jsonl` và `qa_dataset.csv` map đúng về chunk nguồn.
> Tôi hoàn thiện file hướng dẫn hard cases để nhóm có thể chủ động tạo và review các loại case như `adversarial`, `multi-hop`, `out-of-scope`, `ambiguous`, từ đó kiểm tra độ bền của Agent trước benchmark chính thức.

2. **Technical Depth**:
> Chuẩn hóa dữ liệu Retrieval: Tôi hiểu rằng Retrieval Eval chỉ đáng tin khi ID của tài liệu nguồn ổn định và nhất quán xuyên suốt pipeline. Vì vậy, tôi tập trung tạo `chunk_id` và `doc_id` theo quy tắc cố định, đồng thời giữ mapping từ context gốc sang metadata trong `document_metadata.csv`. Nhờ đó, các module phía sau có thể truy ngược từ một câu hỏi sang đúng chunk ground truth mà không bị lệch do trùng context hoặc thay đổi format dữ liệu.
> Hit Rate và MRR: Tôi nắm rõ rằng Hit Rate trả lời câu hỏi "Retriever có tìm thấy đúng tài liệu hay không", còn MRR trả lời câu hỏi "Tài liệu đúng xuất hiện sớm đến mức nào". Với góc nhìn Data Engineer, nếu ground truth mapping không chính xác thì cả hai metric đều mất ý nghĩa. Vì vậy, tôi chuẩn bị `ground_truth_mapping.csv` và kiểm tra tính khớp giữa `golden_set.jsonl`, `qa_dataset.csv` và metadata chunk để đảm bảo module eval đo đúng chất lượng retrieval thay vì đo sai do lỗi dữ liệu.
> Tách biệt Retrieval khỏi Generation: Tôi hiểu một nguyên tắc quan trọng của hệ thống đánh giá là phải tách lỗi retrieval ra khỏi lỗi generation. Nếu Agent trả lời sai nhưng retriever đã lấy đúng chunk, nguyên nhân có thể nằm ở prompting hoặc reasoning. Ngược lại, nếu retriever lấy sai chunk thì không nên đổ lỗi cho model trả lời. Chính vì vậy, tôi chuẩn bị ground truth theo từng chunk để nhóm Core Eval có thể đánh giá Retrieval Quality độc lập trước khi chấm Answer Quality.

3. **Problem Solving**:
> Vấn đề: Dữ liệu nguồn ban đầu đến từ hai file SQuAD khác version nên schema không đồng nhất, có chỗ chỉ có chỉ số ký tự, có chỗ có thêm chỉ số theo word. Ngoài ra, nếu không chuẩn hóa tốt thì rất dễ phát sinh trùng lặp context hoặc gắn sai Ground Truth ID giữa các file benchmark.
> Giải pháp: tôi cùng nhóm Data thống nhất quy tắc merge dữ liệu, tạo `chunk_id` ổn định theo context, chuẩn hóa metadata cho từng chunk và sinh các file `documents_cleaned.csv`, `document_metadata.csv`, `qa_dataset.csv`, `ground_truth_mapping.csv` để toàn bộ pipeline downstream dùng chung một nguồn chuẩn.
> Vấn đề: Các hard cases ban đầu mới dừng ở mức ý tưởng, chưa đủ rõ để review dataset hoặc dùng lại cho failure analysis.
> Giải pháp: tôi hoàn thiện guide hard cases theo hướng thực chiến hơn, bổ sung taxonomy case, checklist review `golden_set`, tiêu chí nhận biết dataset chưa ổn, và cách dùng `challenge_tag` để phân cụm lỗi sau benchmark. Điều này giúp phần Data nối mượt hơn với phần failure analysis của nhóm System & Analysis.
