**Họ và tên**: Trinh Duc Anh
**Vai trò**: Multi-Judge Architect 
**File phụ trách chính**: `engine/llm_judge.py`, phối hợp `engine/runner.py`

## 1. Engineering Contribution

**Module tôi chịu trách nhiệm**
> phụ trách thiết kế và triển khai tầng chấm điểm đa giám khảo (Multi-Judge Consensus Engine), bao gồm: gọi nhiều judge model, tính agreement rate, xử lý conflict/tie-breaking, và đảm bảo luồng gọi judge chạy bất đồng bộ để giảm thời gian benchmark toàn hệ thống.

**Đóng góp cụ thể**
> Trong `engine/llm_judge.py`, xây dựng cấu trúc `JudgeProfile` để mô hình hóa từng judge theo `provider`, `model`, `weight`, `enabled`. Cách này giúp hệ thống có thể mở rộng từ 2 judge lên N judge mà không cần thay đổi contract của runner.

> Triển khai hàm `evaluate_multi_judge()` làm điểm vào chính cho consensus: thu thập kết quả từ các judge, tính `individual_scores`, `agreement_rate`, phát hiện conflict theo `conflict_threshold`, sau đó quyết định điểm cuối bằng `weighted_average` hoặc arbitration khi có xung đột.

> Về cơ chế tie-breaking, triển khai `_arbitrate()` theo hướng deterministic để hệ thống luôn có kết quả ổn định giữa các lần chạy: bắt đầu từ weighted average, cộng bonus cho tín hiệu “đúng theo ground truth” và “refusal đúng khi unanswerable”. Cách này tránh tình trạng conflict làm pipeline dừng hoặc phải xử lý tay.

> Về hiệu năng,   dùng `asyncio.gather()` để gọi song song các judge trong `_evaluate_profile()` thông qua `evaluate_multi_judge()`. Đồng thời ở `engine/runner.py`, mỗi batch test case cũng chạy song song (`run_all` + `asyncio.gather`), giúp hệ thống tăng throughput rõ rệt khi benchmark nhiều case.

>   chuẩn hóa usage tracking (`prompt_tokens`, `completion_tokens`, `tokens_used`) từ từng judge rồi cộng dồn vào output cuối để phục vụ báo cáo chi phí và tối ưu token budget.

## 2. Technical Depth

**1. Multi-Judge không chỉ là “gọi nhiều model”**
> TThiết kế theo tư duy reliability: mỗi judge là một nguồn tín hiệu, còn consensus là bước hợp nhất có kiểm soát. Vì vậy output không chỉ có `final_score` mà còn lưu đầy đủ `individual_scores`, `individual_rationales`, `conflict_detected`, `consensus_method`, giúp truy vết nguyên nhân lệch điểm.

**2. Agreement Rate như một chỉ số calibration**
> dùng chênh lệch điểm lớn nhất để quy đổi về thang đồng thuận `[0,1]` (`_agreement_rate`). Chỉ số này giúp phân biệt rõ:
> - trường hợp điểm cuối cao nhưng judges không đồng thuận (rủi ro cao),
> - trường hợp điểm cuối cao và đồng thuận cao (đáng tin cậy hơn).

**3. Tie-breaking theo deterministic arbitration**
> Khi gap giữa judges vượt ngưỡng (`conflict_threshold`), hệ thống chuyển sang `_arbitrate()` thay vì lấy trung bình đơn giản. Cách này giảm bias do một judge outlier và cho phép encode business rule (reward grounded/correct refusal) minh bạch, dễ audit.

**4. Async 2 tầng để tối ưu runtime**
> Ap dụng bất đồng bộ ở cả 2 tầng:
> - tầng test case: chạy song song nhiều case trong batch (`BenchmarkRunner.run_all`)
> - tầng judge: mỗi case gọi song song nhiều judge (`LLMJudge.evaluate_multi_judge`)
>
> Thiết kế này cho hiệu quả tốt hơn nhiều so với tuần tự (case-by-case, judge-by-judge), đặc biệt khi số lượng case tăng lên 50+ theo yêu cầu lab.

## 3. Problem Solving

**Vấn đề 1: Conflict giữa các judge làm điểm cuối thiếu ổn định**
> Nếu chỉ lấy mean score, một judge quá khắt khe hoặc quá dễ có thể kéo lệch kết quả tổng.

**Giải pháp**
> Thêm cơ chế phát hiện conflict bằng `max_gap >= conflict_threshold` và nhánh arbitration riêng. Nhờ đó hệ thống có quy tắc xử lý bất đồng rõ ràng thay vì phụ thuộc vào cảm tính.

**Vấn đề 2: Rủi ro gián đoạn khi API lỗi hoặc thiếu key**
> Trong môi trường lab, API key có thể thiếu hoặc API timeout, dễ làm hỏng cả pipeline benchmark.

**Giải pháp**
> Thiết kế fallback deterministic trong `_evaluate_profile()`: nếu gọi model thất bại và `fallback_enabled=True`, hệ thống vẫn chấm được điểm + ghi rõ lý do fallback. Điều này đảm bảo benchmark không bị gãy.

**Vấn đề 3: Chi phí và tốc độ khi benchmark nhiều mẫu**
> Chạy tuần tự sẽ khiến thời gian tăng theo số case * số judge.

**Giải pháp**
> dùng async gather song song ở cả runner và judge layer, đồng thời gom usage để theo dõi chi phí theo từng lần chạy, tạo nền tảng cho các vòng tối ưu cost/performance tiếp theo.


