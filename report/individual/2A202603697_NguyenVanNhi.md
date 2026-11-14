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
| Baseline (`dev-tfidf`) | Lexical TF-IDF (`TfidfRetriever`), `top_k=3`, không filter, không rerank, `budget=800` chars | Điểm chuẩn lexical offline trên bộ dữ liệu Day 6 | dev | 12/15 (80.0%) | 12/15 (80.0%) | 7/15 (46.7%) | 0.97 ms/query (index 0.009s) / \$0.00 |
| Phương pháp 2 (`dev-dense`) | Dense pgvector Day 6 (`BAAI/bge-small-en-v1.5`), `top_k=3`, `auto_filter` (`section`), không rerank, `budget=800` chars | Truy xuất ngữ nghĩa giúp bắt paraphrase tốt hơn khớp từ khóa thuần túy | dev | 11/15 (73.3%) | 12/15 (80.0%) | 8/15 (53.3%) | 61.10 ms/query (load 12.689s) / \$0.00 |
| Phương pháp 3a (`dev-bm25`) | Okapi BM25 (giữ mã `1.1`, lọc stopwords), `top_k=3`, `auto_filter`, `budget=800` chars | Giữ nguyên mã mục `MAP 1.1`/`MEASURE 1.1` và bỏ stopwords giúp giảm nhiễu từ chức năng | dev | 12/15 (80.0%) | 13/15 (86.7%) | 7/15 (46.7%) | 0.57 ms/query (index 0.008s) / \$0.00 |
| Phương pháp 3b (`dev-hybrid-b800`) | Hybrid RRF (Dense + BM25, `rrf_k=20`), `top_k=3`, `auto_filter`, `budget=800` chars | Kết hợp ngữ nghĩa Dense và khớp mã/từ khóa BM25 để nâng Hit@3 | dev | 12/15 (80.0%) | 14/15 (93.3%) | 7/15 (46.7%) | 46.89 ms/query (load 12.697s) / \$0.00 |
| Cải thiện 1 (`dev-imp1-hybrid-b2400`) | Hybrid RRF, `top_k=3`, `auto_filter`, mở rộng `budget=2400` chars | Tăng context budget từ 800 lên 2400 để không bị rớt chunk dài (>730 chars) và giữ đủ top-3 | dev | 12/15 (80.0%) | 14/15 (93.3%) | 14/15 (93.3%) | 71.69 ms/query / \$0.00 |
| Cải thiện 2 (`dev-imp2-hybrid-neighbor-b2400`) | Hybrid RRF + Lead-in Neighbor Expansion, `top_k=3`, `auto_filter`, `budget=2400` chars | Kéo chunk liền kề khi chunk dẫn đầu kết thúc bằng `:` hoặc chunk sau bắt đầu bằng bullet `•` | dev | 12/15 (80.0%) | 15/15 (100.0%) | 15/15 (100.0%) | 67.24 ms/query / \$0.00 |
| Cấu hình đã chọn | | | test | | | | |
| Baseline | | | test | | | | |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

- **Lỗi dev #1 — Tràn `Context Budget = 800` gây mất bằng chứng đã tìm thấy (`q02`, `q03`, `q11`, `q13`, `q14`)**:
  - *Nguyên nhân*: Các chunk của chiến lược C (`by_paragraph`, `max_size=800`) dài tới ~799 ký tự; khi `pack_context(..., budget=800)` cộng thêm chuỗi header `[id | source | page N]\n` (~60–70 ký tự), tổng độ dài chunk top-1 vượt quá 800 ký tự nên bị loại khỏi context, khiến hàm đóng gói nhảy sang lấy một chunk ngắn hơn nhưng không chứa bằng chứng ở hạng thấp hơn (như `q02` lấy `p07:c005` dài 342 chars thay vì `p09:c001`). Vì vậy trên `dev-hybrid-b800`, `Hit@3` đạt `14/15 (93.3%)` nhưng `Context Evidence Rate` chỉ còn `7/15 (46.7%)`.
  - *Giả thuyết & kết quả*: Khi tăng `budget` từ `800` lên `2400` ký tự (`dev-imp1-hybrid-b2400`, đủ chứa trọn cả 3 chunk top-3), `Context Evidence Rate` tăng từ `7/15 (46.7%)` lên `14/15 (93.3%)`.
- **Lỗi dev #2 — Tách câu dẫn và danh sách bullet qua ranh giới chunk (`q09`, trang 24) và lệch từ khóa chuyên môn/paraphrase (`q01`, `q12`, `q13`)**:
  - *Nguyên nhân*: Ở `q01` (*"define an AI system"* vs văn bản *"refers to an AI system as..."*), TF-IDF và BM25 bị trượt vì khác từ vựng (`define` vs `refers to`); ngược lại ở `q13` (*"MEASURE 1.1"*), Dense embedding bị nhiễu giữa các mục gần nhau trong bảng, còn BM25 bắt chính xác mã `MEASURE 1.1`. Riêng `q09` có câu dẫn `"Framework users are expected to benefit from:"` nằm ở `p24:c000`, còn toàn bộ danh sách bullet đáp án nằm ở chunk kế tiếp `p24:c001`, nên cả Dense và BM25 đơn lẻ đều chấm điểm cao cho `p24:c000` và bỏ sót `p24:c001` khỏi top-3.
  - *Giả thuyết & kết quả*: Kết hợp Dense + BM25 bằng RRF giải quyết triệt để `q01`, `q12`, `q13` (`Hit@3` đạt `14/15`). Bật thêm cơ chế Lead-in Neighbor Expansion (`dev-imp2-hybrid-neighbor-b2400`) kéo được `p24:c001` ngay sau `p24:c000`, giúp `Evidence Hit@3` và `Context Evidence Rate` trên `dev` cùng đạt `15/15 (100.0%)`.
- **Chốt cấu hình trên `dev` (trước khi chạy `test`)**: Chọn **Cấu hình Cải thiện 2 (`HybridRRFRetriever` với `enable_neighbor_expansion=True`, `auto_filter=True`, `top_k=3`, `budget=2400`)** vì đạt đồng thời `Hit@3 = 100%` và `Context Evidence Rate = 100%` trên `dev` với độ trễ trung vị `67.24 ms/query` và chi phí embedding/vector store `\$0.00` (chạy cục bộ hoàn toàn).

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
