# Code (`src/`)

Thư mục này chứa toàn bộ cài đặt thật kết nối dữ liệu Day 6 (PostgreSQL + `pgvector`, `BAAI/bge-small-en-v1.5`) và DeepSeek API (`deepseek-flash`).

## Thành phần file

- `prepare_day7_data.py`: Xuất 170 chunks của chiến lược C từ database `day6_lab` (`127.0.0.1:55433`), chuẩn hóa câu hỏi Day 6 và bổ sung 4 câu thiếu bằng chứng thành `starter/data/converted/{chunks,questions,cases}.json`.
- `retrievers.py`: Cài đặt `TfidfRetriever`, `BM25Retriever` (giữ mã mục dạng số chấm `1.1`, lọc stopwords), `Day6DenseRetriever` (truy vấn cosine `<=>` trên `pgvector` với `BAAI/bge-small-en-v1.5` 384 chiều), `HybridRRFRetriever` (hợp nhất Dense + BM25 bằng RRF và tùy chọn Lead-in Neighbor Expansion) và `infer_metadata_filter`.
- `run_retrieval_experiments.py`: Chạy benchmark retrieval trên tập `dev` hoặc `test` cho Baseline, Dense, BM25, Hybrid RRF và 2 cấu hình cải thiện.
- `deepseek_agent.py`: Cài đặt `DeepSeekController` tích hợp với `starter/agentic.py::run_agent` để chạy Agentic RAG thật, ghi trace tool call, citation, token và chi phí ước tính.

## Yêu cầu môi trường & Lệnh chạy tái lập

1. Khởi động container PostgreSQL + `pgvector` từ Day 6:
   ```bash
   docker start day6_student_starter-postgres-1
   ```
2. Tạo file `.env` tại thư mục gốc (không commit lên Git):
   ```env
   DEEPSEEK_API_KEY=<your-deepseek-api-key>
   DEEPSEEK_MODEL=deepseek-flash
   DAY6_DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab
   ```
3. Chạy kiểm thử và tái lập toàn bộ thực nghiệm:
   ```bash
   python -m unittest discover -s tests -v
   python src/prepare_day7_data.py
   python src/run_retrieval_experiments.py --split dev
   python src/deepseek_agent.py --split dev --config both
   python src/run_retrieval_experiments.py --split test
   python src/deepseek_agent.py --split test --config both
   python starter/agent_metrics.py --cases starter/data/converted/cases.json --traces starter/results/test-agent-selected/traces.json --reviews starter/results/test-agent-selected/reviews.json --split test --output starter/results/test-agent-selected/summary.json
   ```
