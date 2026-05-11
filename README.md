# AI Technical Assistant - RAG Chatbot

Một hệ thống RAG (Retrieval-Augmented Generation) chạy hoàn toàn cục bộ (Local), hỗ trợ đọc và tra cứu tài liệu kỹ thuật song ngữ (Tiếng Việt - Tiếng Anh) với độ chính xác cao nhờ cơ chế Reranker và bộ nhớ hội thoại.

## Tính năng nổi bật

- **Chạy Local 100%**: Sử dụng Ollama để vận hành LLM, đảm bảo tính riêng tư và bảo mật dữ liệu.

- **Xử lý Tài liệu Song ngữ**:: Tối ưu hóa cho tài liệu kỹ thuật chứa cả Tiếng Việt và thuật ngữ Tiếng Anh chuyên ngành.

- **Tìm kiếm Nâng cao (Reranking)**:: Kết hợp FAISS Vector Search và Cross-Encoder Reranker để lọc ra những nguồn tin cậy nhất.

- **Cập nhật Tăng cường (Incremental Update)**:: Tự động phát hiện và chỉ cập nhật những file PDF mới được thêm vào folder dữ liệu.

- **Trí nhớ Hội thoại (History-Aware)**:: Hiểu được các đại từ và ngữ cảnh từ các câu hỏi trước đó (Tối ưu 5 lượt chat gần nhất).

- **Giao diện Web hiện đại**:: Xây dựng bằng Streamlit với hiệu ứng Streaming (gõ chữ thời gian thực) và trích dẫn nguồn (Source Citation) minh bạch.

## Công nghệ sử dụng

- **Language Model**: Qwen 2.5 (via Ollama)

- **Embedding Model**: BGE-M3 (Đa ngôn ngữ)

- **Reranker**: MS-Marco MiniLM (Siêu nhẹ cho CPU)

- **Framework**: LangChain

- **Vector Database**: FAISS

- **UI**: Streamlit

## Cấu trúc dự án

- **data/**: Thư mục chứa các file PDF đầu vào
- **database/**: Module quản lý xử lý dữ liệu và FAISS
- **chains/**: Module quản lý logic AI, Prompts và Reranker
- **faiss_index/**: Lưu trữ cơ sở dữ liệu vector và sổ ghi chép file
- **app.py**: Giao diện Web chính (Streamlit)

## Hướng dẫn cài đặt

1. Cài đặt Ollama và Model
   Tải Ollama tại ollama.com và chạy các lệnh sau để tải model:

```bash
ollama pull qwen2.5
ollama pull bge-m3
```

2. Cài đặt môi trường Python

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

3. Chuẩn bị dữ liệu
   Copy các file PDF bạn muốn tra cứu vào thư mục ./data/.

## Cách vận hành

Chạy giao diện Web

```bash
streamlit run app.py
```

## Lưu ý

- Hệ thống được cấu hình chạy Reranker trên CPU để tiết kiệm VRAM cho card đồ họa.

- Lần đầu tiên chạy, hệ thống sẽ tải model Reranker (khoảng 2.2GB) từ HuggingFace, các lần sau sẽ khởi động tức thì.

- Nhập exit để thoát khi dùng giao diện Terminal.
