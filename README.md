# Agent Local Project

## 📌 Tổng quan
Project là hệ thống AI local với khả năng chat, build, và quản lý memory. Hỗ trợ đa model thông qua `MODEL_CANDIDATES` và cấu hình linh hoạt qua `SERVER_CONFIG`.

## 🚀 Cài đặt

### 1. Yêu cầu
- Python 3.10+
- Flask 2.3.2
- gunicorn 20.1.0
- Ollama (cài đặt qua `OllamaSetup.exe`)

### 2. Cấu hình
- Chỉnh sửa `config.py` để thiết lập:
  - `MODEL_CANDIDATES`: Danh sách model hỗ trợ
  - `SERVER_CONFIG`: Cấu hình server (port, cors, v.v.)
- Kiểm tra file `verify_config.py` để xác minh cấu hình

### 3. Cài đặt Ollama
1. Chạy `OllamaSetup.exe` (kích thước ~2GB)
2. Cấu hình model qua CLI: `ollama run <model_name>`

### 4. Tùy chọn
- Chạy script PowerShell: `_write_orch.ps1` để orchestrate deployment

## 📡 Sử dụng

### 1. Chạy server
```bash
python server.py
``` 

### 2. Giao diện
- Truy cập: `http://localhost:5000`
- Tương tác qua:
  - Chat stream: `/api/chat`
  - Build: `/api/build`
  - Memory: `/api/memory`

### 3. API
- **Chat**: POST `/api/chat` (trả về SSE)
- **Build**: POST `/api/build` (trả về SSE)
- **Memory**: GET `/api/memory` (theo `session_id`)

## 📁 Cấu trúc thư mục
```
agent_local/
├── core/                # Logic cốt lõi
├── memory/              # Quản lý memory
├── templates/           # Frontend (HTML/JS/CSS)
├── tests/               # Unit tests
├── config.py            # Cấu hình chính
├── verify_config.py     # Kiểm tra config
├── requirements.txt     # Dependency
├── _write_orch.ps1      # Script PowerShell
├── OllamaSetup.exe      # Cài đặt Ollama
└── server.py            # Backend chính
``` 

## ⚠️ Lưu ý
1. File `server.py` hiện bị truncate ở dòng `memory_man` – cần kiểm tra nội dung đầy đủ
2. Cấu hình CORS đã mở rộng cho mọi origin (`cors_allowed_origins="*"`)
3. Hỗ trợ cancel cho chat/build qua `/api/build/cancel` và tương ứng

## 🤝 Góp ý
- Tối ưu memory manager (hiện đang bị truncate)
- Thêm middleware logging
- Tách config thành file riêng
- Hỗ trợ cancel cho chat stream

## 📚 Tài liệu
- Đọc `agent.md` để hiểu định dạng input/output

Phản hồi nếu bạn cần chỉnh sửa hoặc thêm thông tin cụ thể!