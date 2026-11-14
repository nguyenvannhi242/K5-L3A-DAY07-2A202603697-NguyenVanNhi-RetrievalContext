# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk:
  - Tài liệu gốc: *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (`NIST.AI.100-1.pdf`, DOI: `10.6028/NIST.AI.100-1`, phiên bản 1.0 - January 2023).
  - Quyền sử dụng: Public Domain (U.S. Government Work, NIST; được phép chia sẻ và tái sử dụng công khai).
  - Số trang: 48 trang vật lý, gồm 42 trang nội dung hợp lệ (trang 6–47; loại bỏ trang 1–5 phần bìa/mục lục và trang 48 bìa sau).
  - Số chunk (kế thừa chiến lược C từ Day 6 - `by_paragraph`, `max_size=800`): 170 chunks (100,897 ký tự).
- Vector store, embedding model, phiên bản và số chiều:
  - Vector store: PostgreSQL 16.16 + `pgvector` 0.8.7 (Docker container `day6_student_starter-postgres-1`, database `day6_lab`, bảng `chunks`, metric Cosine `<=>`). Kết nối cục bộ qua `postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab`.
  - Embedding model: `BAAI/bge-small-en-v1.5` (thư viện `sentence-transformers`), 384 chiều, chuẩn hóa L2 (`normalize_embeddings=True`), dùng prefix `"Represent this sentence for searching relevant passages: "` cho query.
- Agent model, cách chạy và phân chia dev/test:
  - Agent model: DeepSeek API (`deepseek-flash`), cấu hình qua biến môi trường `.env` (`DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`).
  - Lệnh chuẩn hóa dữ liệu từ Day 6: `python src/prepare_day7_data.py` (xuất `starter/data/converted/chunks.json`, `questions.json`, `cases.json`).
  - Lệnh chạy Baseline TF-IDF (dev): `python starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf`.
  - Phân chia dev/test (tổng 24 câu): `dev` gồm 17 câu (15 câu có đáp án `q01`–`q15` + 2 câu thiếu bằng chứng `q21`–`q22`); `test` gồm 7 câu (5 câu có đáp án `q16`–`q20` + 2 câu thiếu bằng chứng `q23`–`q24`).
- Cách xác định câu có/thiếu evidence và tiêu chí chấm:
  - Câu có đáp án: mỗi câu có `expected_answer` và `gold` (`source_id`, `page`, `evidence` khớp nguyên văn trong một chunk đúng nguồn và trang; `q09` ở tập `dev` được chuẩn hóa `evidence_quote` sang chunk `nist-ai-100-1:C:p24:c001` để đảm bảo toàn bộ chuỗi nằm trọn trong một chunk khi chạy `validate()`).
  - Câu thiếu bằng chứng (`q21`–`q24`): hỏi cùng chủ đề NIST AI RMF 1.0 (xử phạt hành chính bắt buộc, lệ phí chứng nhận, ngưỡng accuracy tối thiểu, ánh xạ EU AI Act / ISO 42001) và đã kiểm tra toàn văn 42 trang (170 chunks) xác nhận không có thông tin; không đưa các câu này vào mẫu số tính retrieval metrics.
  - Tiêu chí chấm: tách riêng Retrieval (`Hit@1`, `Evidence Hit@3`, `Context Evidence Rate`, độ trễ, chi phí) và Agent (`Tool use`, `Grounded answer` 0/1/2, `Citation hợp lệ`, `Abstention` đúng / `False abstention`).

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Lexical TF-IDF (`starter/baseline.py`), `top_k=3`, không filter, không rerank, `budget=800` chars | Điểm chuẩn lexical offline trên bộ dữ liệu Day 6 | dev | 12/15 (80.0%) | 12/15 (80.0%) | 7/15 (46.7%) | 0.40 ms/query (index 0.010s) / \$0.00 |
| Phương pháp 2 | | | dev | | | | |
| Phương pháp 3 | | | dev | | | | |
| Cải thiện 1 | | | dev | | | | |
| Cải thiện 2 | | | dev | | | | |
| Cấu hình đã chọn | | | test | | | | |
| Baseline | | | test | | | | |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
