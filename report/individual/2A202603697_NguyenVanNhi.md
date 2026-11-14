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
| Cấu hình đã chọn (`test-imp2-hybrid-neighbor-b2400`) | Hybrid RRF + Lead-in Neighbor Expansion, `top_k=3`, `auto_filter`, `budget=2400` chars | Chốt từ `dev` và kiểm định độc lập trên `test` | test | 4/5 (80.0%) | 5/5 (100.0%) | 5/5 (100.0%) | 65.31 ms/query (load 16.700s) / \$0.00 |
| Baseline (`test-tfidf`) | Lexical TF-IDF (`TfidfRetriever`), `top_k=3`, không filter, không rerank, `budget=800` chars | Điểm chuẩn lexical offline trên `test` | test | 5/5 (100.0%) | 5/5 (100.0%) | 3/5 (60.0%) | 0.71 ms/query (index 0.015s) / \$0.00 |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

- **Lỗi dev #1 — Tràn `Context Budget = 800` gây mất bằng chứng đã tìm thấy (`q02`, `q03`, `q11`, `q13`, `q14`)**:
  - *Nguyên nhân*: Các chunk của chiến lược C (`by_paragraph`, `max_size=800`) dài tới ~799 ký tự; khi `pack_context(..., budget=800)` cộng thêm chuỗi header `[id | source | page N]\n` (~60–70 ký tự), tổng độ dài chunk top-1 vượt quá 800 ký tự nên bị loại khỏi context, khiến hàm đóng gói nhảy sang lấy một chunk ngắn hơn nhưng không chứa bằng chứng ở hạng thấp hơn (như `q02` lấy `p07:c005` dài 342 chars thay vì `p09:c001`). Vì vậy trên `dev-hybrid-b800`, `Hit@3` đạt `14/15 (93.3%)` nhưng `Context Evidence Rate` chỉ còn `7/15 (46.7%)`.
  - *Giả thuyết & kết quả*: Khi tăng `budget` từ `800` lên `2400` ký tự (`dev-imp1-hybrid-b2400`, đủ chứa trọn cả 3 chunk top-3), `Context Evidence Rate` tăng từ `7/15 (46.7%)` lên `14/15 (93.3%)`.
- **Lỗi dev #2 — Tách câu dẫn và danh sách bullet qua ranh giới chunk (`q09`, trang 24) và lệch từ khóa chuyên môn/paraphrase (`q01`, `q12`, `q13`)**:
  - *Nguyên nhân*: Ở `q01` (*"define an AI system"* vs văn bản *"refers to an AI system as..."*), TF-IDF và BM25 bị trượt vì khác từ vựng (`define` vs `refers to`); ngược lại ở `q13` (*"MEASURE 1.1"*), Dense embedding bị nhiễu giữa các mục gần nhau trong bảng, còn BM25 bắt chính xác mã `MEASURE 1.1`. Riêng `q09` có câu dẫn `"Framework users are expected to benefit from:"` nằm ở `p24:c000`, còn toàn bộ danh sách bullet đáp án nằm ở chunk kế tiếp `p24:c001`, nên cả Dense và BM25 đơn lẻ đều chấm điểm cao cho `p24:c000` và bỏ sót `p24:c001` khỏi top-3.
  - *Giả thuyết & kết quả*: Kết hợp Dense + BM25 bằng RRF giải quyết triệt để `q01`, `q12`, `q13` (`Hit@3` đạt `14/15`). Bật thêm cơ chế Lead-in Neighbor Expansion (`dev-imp2-hybrid-neighbor-b2400`) kéo được `p24:c001` ngay sau `p24:c000`, giúp `Evidence Hit@3` và `Context Evidence Rate` trên `dev` cùng đạt `15/15 (100.0%)`.
- **Chốt cấu hình trên `dev` (trước khi chạy `test`)**: Chọn **Cấu hình Cải thiện 2 (`HybridRRFRetriever` với `enable_neighbor_expansion=True`, `auto_filter=True`, `top_k=3`, `budget=2400`)** vì đạt đồng thời `Hit@3 = 100%` và `Context Evidence Rate = 100%` trên `dev` với độ trễ trung vị `67.24 ms/query` và chi phí embedding/vector store `\$0.00` (chạy cục bộ hoàn toàn).
- **Chấm điểm Context 0/1/2 cho 5 câu `test` có đáp án (`starter/results/test-context-audit.json`)**:
  - `Baseline (test-tfidf)`: Mức `0`: **1 câu** (`q17` - `p19:c000` bị loại vì vượt `budget=800`); Mức `1`: **1 câu** (`q18` - `p37:c001` bị cắt, context chỉ còn `p13:c002` nói chung về residual risk nhưng thiếu định nghĩa MANAGE 1.4); Mức `2`: **3 câu** (`q16`, `q19`, `q20`). Trung bình **1.40 / 2**.
  - `Selected (test-imp2-hybrid-neighbor-b2400)`: Mức `0`: **0 câu**; Mức `1`: **0 câu**; Mức `2`: **5/5 câu** (`q16`–`q20` đều giữ đầy đủ evidence và nguồn/trang gốc). Trung bình **2.00 / 2**.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
  - **Controller**: `DeepSeekController` (`deepseek-flash`, `temperature=0.0`, `max_tokens=4096`, `response_format=json_object`) kết nối vào vòng lặp `run_agent()` của `starter/agentic.py`. Coi toàn bộ nội dung tài liệu là dữ liệu thuần túy, không thực thi chỉ dẫn trong chunk.
  - **Tool & Tham số**:
    - `baseline`: `TfidfRetriever`, `top_k=3`, `budget=800` chars, `max_searches=2`.
    - `selected`: `HybridRRFRetriever` (Dense `pgvector` Day 6 + BM25 + Lead-in Neighbor Expansion + `auto_filter`), `top_k=5`, `budget=3500` chars, `max_searches=3` (mở rộng từ `budget=2400` sau thử nghiệm sơ bộ trên `dev` để tránh việc 3 chunk lượt search 1 chiếm gần hết ngân sách khiến lượt search 2 không nạp thêm được chunk mới ở các câu cần truy vấn lại).
  - **Điều kiện dừng**: Dừng khi model xuất `action="answer"` kèm citations hợp lệ nằm trong `context_ids` (và xuất hiện `[chunk-id]` trong câu trả lời), hoặc tự nguyện từ chối (`"Không đủ bằng chứng để trả lời."`, `citations=[]`), hoặc chạm ngưỡng `max_searches`.
  - **Cách chọn context & xử lý lỗi**: Dùng `pack_context` (dedup theo `id`, giữ thứ tự ưu tiên theo score/rank trong giới hạn ký tự `budget`). Parser JSON (`parse_json_object`) trích xuất block `{...}` hợp lệ đầu tiên và thử lại tối đa 1 lần nếu phản hồi cắt ngọn/dư ký tự; lỗi API/DB thực sự được ghi nhận riêng vào `status="error"` theo quy định của `agent_metrics.py`, không thay thế bằng kết quả giả.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline (`dev-agent-baseline`) / `dev` | 17/17 (100%) | 1: 8 câu, 2: 9 câu | 0: 7, 1: 2, 2: 6 · TB **0.93 / 2** | 8/8 (100%) | 8/8 (100%) | 2/2 (100%) | 7/15 (46.7%) |
| Selected (`dev-agent-selected`) / `dev` | 17/17 (100%) | 1: 15 câu, 2: 1 câu, 3: 1 câu | 0: 1, 1: 0, 2: 14 · TB **1.87 / 2** | 28/28 (100%) | 15/15 (100%) | 1/2 (50.0%) | 1/15 (6.7%) |
| Baseline (`test-agent-baseline`) / `test` | 7/7 (100%) | 1: 2 câu, 2: 5 câu | 0: 3, 1: 0, 2: 2 · TB **0.80 / 2** | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 3/5 (60.0%) |
| Selected (`test-agent-selected`) / `test` | 7/7 (100%) | 1: 5 câu, 2: 1 câu, 3: 1 câu | 0: 0, 1: 0, 2: 5 · TB **2.00 / 2** | 5/5 (100%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0.0%) |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 search (`Executive Summary ISO 26000:2010...`), khớp `p07:c000` (trang 7) | Trích đúng định nghĩa trách nhiệm xã hội của ISO 26000:2010 `[nist-ai-100-1:C:p07:c000]` | `answer_grade=2`, `cit=[true]` (Baseline: `2`) |
| `q17` | Có đáp án | 1 search (`AI RMF accuracy measurements...`), khớp `p19:c000` (trang 19) | Trả lời đủ: realistic test sets đại diện điều kiện sử dụng và chi tiết test methodology trong tài liệu `[nist-ai-100-1:C:p19:c000]` | `answer_grade=2`, `cit=[true]` (Baseline: `0`, mất chunk do budget 800) |
| `q18` | Có đáp án | 1 search (`MANAGE 1.4 negative residual risk defined`), khớp `p37:c001` (trang 37) | Định nghĩa chính xác sum of all unmitigated risks và yêu cầu ghi nhận cho downstream acquirers & end users `[nist-ai-100-1:C:p37:c001]` | `answer_grade=2`, `cit=[true]` (Baseline: `0`, mất chunk do budget 800) |
| `q19` | Có đáp án | 1 search (`mathematical models representing complex human phenomena...`), khớp `p45:c001` (trang 45) | Giải thích đúng rủi ro mất ngữ cảnh (loss of context) gây khó hiểu tác động cá nhân/xã hội `[nist-ai-100-1:C:p45:c001]` | `answer_grade=2`, `cit=[true]` (Baseline: `0`, từ chối nhầm) |
| `q20` | Có đáp án | 1 search (`Appendix D first two guiding attributes...`), khớp `p47:c000` (trang 47) | Nêu đủ 2 thuộc tính đầu: (1) risk-based, resource-efficient, pro-innovation, voluntary; (2) consensus-driven, open/transparent process `[nist-ai-100-1:C:p47:c000]` | `answer_grade=2`, `cit=[true]` (Baseline: `2`) |
| `q23` | Thiếu bằng chứng | 2 searches (`Measure minimum accuracy percentage threshold...`, `MEASURE accuracy benchmark metrics...`) | `"Không đủ bằng chứng để trả lời."` (`citations=[]`) | `abstention_correct=true` (đối chiếu p.33–36 không có ngưỡng phần trăm tối thiểu) |
| `q24` | Thiếu bằng chứng | 3 searches (`Appendix A ... EU AI Act ... ISO/IEC 42001` với 3 biến thể query) | `"Không đủ bằng chứng để trả lời."` (`citations=[]`) | `abstention_correct=true` (đối chiếu Appendix A–D không có EU AI Act hay ISO 42001) |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
  - Lệnh chạy `dev`: `python src/deepseek_agent.py --split dev --config both` và tổng hợp bằng `python starter/agent_metrics.py`.
  - Lệnh chạy `test`: `python src/run_retrieval_experiments.py --split test`, `python src/deepseek_agent.py --split test --config both`, và `python starter/agent_metrics.py --split test`.
  - `dev-agent-baseline`: `starter/results/dev-agent-baseline/{traces,reviews,summary}.json` (`36,525` tokens, `\$0.005877`, `55.24s`, `provider_errors=0`).
  - `dev-agent-selected`: `starter/results/dev-agent-selected/{traces,reviews,summary}.json` (`45,016` tokens, `\$0.007295`, `58.08s`, `provider_errors=0`).
  - `test-agent-baseline`: `starter/results/test-agent-baseline/{traces,reviews,summary}.json` (`16,392` tokens, `\$0.002644`, `25.13s`, `provider_errors=0`).
  - `test-agent-selected`: `starter/results/test-agent-selected/{traces,reviews,summary}.json` (`20,628` tokens, `\$0.003205`, `24.01s`, `provider_errors=0`).
  - Audit Context 5 câu `test`: `starter/results/test-context-audit.json`.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - Trên `dev`, chuyển từ `baseline` sang `selected` giúp điểm Grounded Answer trung bình tăng gấp đôi từ **0.93/2** lên **1.87/2** (14/15 câu có đáp án đạt mức 2 tuyệt đối), giảm từ chối nhầm từ **7/15** xuống **1/15**.
  - Trên tập `test` độc lập, cấu hình `selected` đạt kết quả toàn diện: **5/5 câu có đáp án đạt `2/2`** (trung bình **2.00/2**, so với Baseline **0.80/2**), **5/5 citation hợp lệ (100%)**, **2/2 câu thiếu bằng chứng từ chối chính xác (100%)**, **0/5 từ chối nhầm** và **0 lỗi hệ thống**.
  - *Ví dụ lỗi `q01` trên `dev-agent-selected`*: Khi tự viết lại query ở lượt 1 thành `'AI system definition AI RMF 1.0'`, từ khóa `'1.0'` khiến BM25 ưu tiên các chunk đầu trang chứa header `'NIST AI 100-1 AI RMF 1.0'` thay vì `p06:c001`; sang lượt 2–3 model thêm `'Appendix B'` làm `infer_metadata_filter` kích hoạt lọc `section='appendix'`, dẫn tới không quay lại được trang 6 và từ chối nhầm.
  - *Ví dụ trường hợp ranh giới `q21` (`dev`, `answerable=False`)*: Model trả lời đúng bản chất tài liệu là tự nguyện (`voluntary`) không quy định phạt hành chính kèm 3 citation hợp lệ (`p07:c002`, `p47:c000`, `p26:c000`) thay vì xuất `status="abstain"`, nên theo quy tắc chấm cứng `agent_metrics.py` bị tính `abstention_correct=false`.
  - *Đánh đổi Chất lượng – Thời gian – Chi phí*: So với TF-IDF `budget=800` (`0.71 ms/query` retrieval), cấu hình `selected` (Hybrid RRF + `top_k=5` + `budget=3500`) mất thêm thời gian tải model cục bộ một lần (~12–16s lúc khởi tạo) và ~65 ms/query khi retrieval, cùng mức tăng chi phí LLM trên tập `test` chỉ **\$0.00056** (`\$0.003205` vs `\$0.002644`, thậm chí tổng thời gian chạy 7 câu `test` nhanh hơn nhẹ `24.01s` vs `25.13s` do đa số câu trả lời ngay ở lượt search 1), nhưng mang lại mức tăng đột phá về Grounded Answer (`0.80 -> 2.00` trên `test`; `0.93 -> 1.87` trên `dev`).
  - *Giới hạn*: Bộ lọc suy luận `infer_metadata_filter` dựa trên quy tắc từ khóa nên nhạy với việc LLM tự thêm tên section khác (như `'Appendix B'` ở `q01`); `pack_context` ưu tiên giữ các chunk từ lượt search đầu nên nếu `budget` không đủ lớn, các lượt search sau dễ bị thiếu chỗ; tập `test` gồm 7 câu theo quy mô tối thiểu của bài cơ bản trên 1 tài liệu chuẩn 42 trang hợp lệ (`NIST AI 100-1`).

## Thử thách nâng cao (nếu thực hiện)

- Không thực hiện (bài làm tập trung hoàn thành trọn vẹn Phần cơ bản 85/85 điểm).
