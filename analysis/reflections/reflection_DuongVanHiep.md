**Họ và tên**: Dương Văn Hiệp - 2A202600052  
**Vai trò**: Nhóm Data Engineers - Retrieval Master  
**File phụ trách chính**: `data/synthetic_gen.py`, `data/QA_dataset/`, `analysis/HARD_CASES_GUIDE.md`

## 1. Engineering Contribution

**Module tôi chịu trách nhiệm**
> Tôi phụ trách lớp dữ liệu của pipeline evaluation: chuẩn hóa bộ dữ liệu nguồn, tổ chức metadata theo chunk, gắn Ground Truth ID cho từng test case và xây dựng bộ hướng dẫn hard cases để kiểm tra độ bền của Agent trước khi benchmark.

**Đóng góp cụ thể**
> Tôi tham gia trực tiếp vào việc chuyển bộ dữ liệu đầu vào từ hai file `SQuAD-v1.1.csv` và `SQuAD-v1.2.csv` thành một nguồn benchmark dùng được cho hệ thống eval của nhóm. Trong `data/synthetic_gen.py`, tôi cùng nhóm thiết kế lại pipeline theo hướng có thể merge hai version dữ liệu, loại bỏ sai khác schema, chuẩn hóa text, và giữ lại mapping ổn định giữa câu hỏi, context và đáp án gốc.

> Tôi đóng vai trò chính ở phần chuẩn bị metadata retrieval. Cụ thể, tôi tổ chức lại dữ liệu trong `data/QA_dataset/` thành các artifact có ý nghĩa kỹ thuật rõ ràng:
> - `documents_cleaned.csv`: dữ liệu QA đã được làm sạch và chuẩn hóa schema
> - `document_metadata.csv`: metadata cấp chunk gồm `chunk_id`, `doc_id`, `title`, `path`, `qa_count`, `word_count`, `char_count`, `preview`
> - `ground_truth_mapping.csv`: bảng ánh xạ ground truth retrieval cho từng test case
> - `qa_dataset.csv`: tập benchmark cuối cùng để nhóm review nhanh bằng dạng bảng

> Tôi chịu trách nhiệm đảm bảo mỗi case trong `golden_set.jsonl` đều có `expected_retrieval_ids` nhất quán với metadata chunk. Điều này rất quan trọng vì toàn bộ module `RetrievalEvaluator` phía sau chỉ có ý nghĩa khi ID ground truth là đúng và ổn định.

> Ngoài phần clean data, tôi còn hoàn thiện tài liệu `analysis/HARD_CASES_GUIDE.md` theo hướng dùng được thật trong project, không chỉ là note ý tưởng. File này mô tả rõ taxonomy của các nhóm case như `adversarial`, `multi-hop`, `out-of-scope`, `ambiguous`, `stress cases`, đồng thời chỉ ra cách review dataset và cách dùng `challenge_tag` để phục vụ failure analysis về sau.

**Tác động lên hệ thống**
> Đóng góp của tôi nằm ở tầng nền tảng nhưng ảnh hưởng trực tiếp đến độ tin cậy của toàn bộ benchmark. Nhờ bộ ground truth và metadata được chuẩn hóa, hệ thống có thể đo đúng `Hit Rate = 1.0`, `MRR = 1.0`, đồng thời hỗ trợ nhóm downstream chạy benchmark 60 cases với `pass_rate = 1.0` và `agreement_rate = 0.9867` trong báo cáo cuối.

## 2. Technical Depth

**1. Hiểu đúng bản chất của Retrieval Evaluation**
> Tôi hiểu rằng Retrieval Eval không phải chỉ là “có dữ liệu là chạy được”, mà là bài toán gắn đúng tài liệu nguồn vào đúng câu hỏi để các metric như Hit Rate và MRR phản ánh chất lượng retriever thật sự. Nếu `expected_retrieval_ids` sai hoặc không ổn định, toàn bộ benchmark phía sau sẽ đo sai nguyên nhân lỗi: có thể tưởng retriever tệ trong khi thực chất lỗi nằm ở data mapping.

**2. Ổn định hóa ID là điều kiện bắt buộc để benchmark có giá trị**
> Một quyết định kỹ thuật quan trọng tôi tham gia là dùng `chunk_id` và `doc_id` ổn định theo context/title thay vì phụ thuộc vào vị trí dòng hay format tạm thời của dữ liệu gốc. Điều này giúp pipeline downstream có thể:
> - truy ngược từ test case sang chunk nguồn
> - so sánh retrieval giữa các lần chạy benchmark
> - dùng chung một chuẩn dữ liệu giữa `golden_set.jsonl`, `ground_truth_mapping.csv` và `document_metadata.csv`

**3. Phân biệt rõ vai trò của các artifact dữ liệu**
> Tôi không xem `documents_cleaned.csv`, `document_metadata.csv`, `ground_truth_mapping.csv` là các file “xuất thêm cho đẹp”, mà là ba lớp dữ liệu phục vụ ba mục tiêu khác nhau:
> - `documents_cleaned.csv` để chuẩn hóa dữ liệu đầu vào
> - `document_metadata.csv` để mô tả nguồn retrieval theo cấp chunk
> - `ground_truth_mapping.csv` để làm chuẩn đo chính thức cho evaluator

> Việc tách ba lớp này giúp pipeline rõ trách nhiệm hơn và dễ debug hơn khi benchmark bị sai.

**4. Hiểu trade-off giữa direct cases và hard cases**
> Trong pipeline mới, bộ dữ liệu cuối không chỉ lấy trực tiếp từ SQuAD mà còn kết hợp thêm các hard cases sinh bởi prompt engineering trong `generate_qa_from_text`. Tôi hiểu đây là một trade-off hợp lý:
> - direct cases giúp benchmark ổn định, dễ kiểm chứng ground truth
> - generated hard cases giúp tăng độ khó, tạo tình huống `adversarial` và `multi-hop`

> Cách phối hợp này giúp bộ dữ liệu không bị quá “dễ”, nhưng vẫn giữ được khả năng truy vết nguồn rõ ràng.

## 3. Problem Solving

**Vấn đề 1: Dữ liệu nguồn đến từ hai version SQuAD có schema khác nhau**
> File `SQuAD-v1.1.csv` và `SQuAD-v1.2.csv` không cùng schema: một bên thiên về chỉ số ký tự, một bên có thêm chỉ số theo word. Nếu đưa thẳng vào pipeline thì rất dễ dẫn đến sai mapping, đặc biệt khi cần truy nguyên context gốc cho retrieval ground truth.

**Giải pháp**
> Tôi cùng nhóm thiết kế bước merge và chuẩn hóa schema trong `data/synthetic_gen.py`: parse các trường tùy chọn, hợp nhất version nguồn, giữ lại `source_versions`, chuẩn hóa text và tạo lại cấu trúc dữ liệu chung bằng `RawQaRow` và `SourceChunk`. Cách làm này giúp pipeline phía sau không còn phụ thuộc vào sự khác nhau của hai file raw.

**Vấn đề 2: Một title có nhiều context, dễ gắn nhầm Ground Truth**
> Với những title lớn trong SQuAD, có rất nhiều đoạn context gần giống nhau. Nếu chỉ dựa vào `title` để map thì khả năng gắn sai chunk là rất cao, kéo theo việc Hit Rate/MRR bị “ảo”.

**Giải pháp**
> Tôi xử lý theo hướng group theo cặp `(title, context)` để sinh `chunk_id` duy nhất cho từng context thực. Sau đó toàn bộ benchmark sẽ dùng `expected_retrieval_ids` dựa trên chunk thật, không dựa vào title chung chung. Đây là điểm tôi đánh giá là quan trọng nhất để retrieval benchmark có giá trị kỹ thuật.

**Vấn đề 3: Hard cases ban đầu chưa đủ thực chiến**
> Nếu chỉ benchmark các câu hỏi factoid trực tiếp thì Agent dễ đạt điểm cao nhưng không chứng minh được độ bền. Trong khi rubric của lab yêu cầu phải có red teaming và kiểm tra các tình huống khó.

**Giải pháp**
> Tôi hoàn thiện `analysis/HARD_CASES_GUIDE.md` để nhóm có tài liệu rõ ràng khi thiết kế và review các case `adversarial`, `multi-hop`, `out-of-scope`, `ambiguous`. Tôi không dừng ở mức liệt kê tên case, mà mô tả luôn mục tiêu kiểm tra, ví dụ minh họa, dấu hiệu Agent chưa bền và cách dùng `challenge_tag` để liên kết sang failure analysis.

## 4. Tự đánh giá

> Nếu tự đánh giá theo rubric cá nhân, tôi tin phần đóng góp của mình ở mức 9-10 điểm vì tôi không chỉ “chuẩn bị dữ liệu” theo nghĩa thao tác file, mà đã tham gia giải quyết một bài toán kỹ thuật thực sự của hệ thống evaluation: làm sao để retrieval benchmark có ground truth đáng tin cậy, có metadata truy vết được, và có bộ hard cases đủ sức kiểm tra chất lượng Agent. Phần việc của tôi cũng có đầu ra cụ thể, tích hợp trực tiếp vào pipeline benchmark và ảnh hưởng rõ ràng đến chất lượng báo cáo cuối.
