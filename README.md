# Hermes Agent

Hệ thống AI local đa tác nhân (multi-agent) chạy hoàn toàn trên máy tính cá nhân, không cần kết nối cloud. Được xây dựng trên [Ollama](https://ollama.com) + Flask, Hermes phối hợp ba agent độc lập — **Primary**, **Reviewer**, **Supervisor** — để tạo ra câu trả lời và code chất lượng cao thông qua vòng lặp tự kiểm tra.

---

## Kiến trúc

```
Người dùng
    │
    ▼
┌─────────────┐     SSE stream      ┌──────────────────────────────────┐
│  Web UI     │ ◄────────────────── │          Orchestrator            │
│  (port 7799)│                     │                                  │
└─────────────┘                     │  Primary  →  Reviewer            │
                                    │      └──────────► Supervisor     │
┌─────────────┐                     │                                  │
│  HCI Panel  │ ◄─── SocketIO ────► │  MemoryManager  │  SystemMonitor │
│  /hci/      │                     │  HermesMemory   │  SubScheduler  │
└─────────────┘                     └──────────────────────────────────┘
                                                │
                                          Ollama API
                                     (localhost:11434)
```

| Agent | Vai trò | Model mặc định |
|---|---|---|
| Primary | Xử lý chat & sinh code | `llama3.1:8b` |
| Reviewer | Kiểm tra chất lượng output | `mistral:v0.3` |
| Supervisor | Đánh giá cuối & quality gate | `qwen3` |

Mỗi model được load/unload khỏi VRAM sau mỗi lượt gọi (`keep_alive=0` cho Reviewer/Supervisor), đảm bảo chỉ một model chiếm VRAM tại một thời điểm.

---

## Tính năng

- **Chat streaming** — phản hồi real-time qua Server-Sent Events
- **Build mode** — sinh toàn bộ project code theo yêu cầu, có feedback loop
- **Memory hệ thống** — short-term (per session) + long-term + cross-session facts
- **HCI Dashboard** — giao diện quản trị bảo vệ bằng mật khẩu tại `/hci/`
- **System monitor** — theo dõi CPU, RAM, VRAM realtime qua SocketIO
- **Subconscious Scheduler** — tác vụ nền tự động chạy định kỳ
- **Skill system** — kỹ năng định nghĩa bằng YAML, tự động phát hiện và nạp
- **Multi-GPU profile** — cấu hình sẵn cho RTX 3060 12GB và GTX 1650 4GB

---

## Yêu cầu

- Python 3.10+
- [Ollama](https://ollama.com/download) đã cài và đang chạy
- NVIDIA GPU (khuyến nghị 6GB+ VRAM) hoặc CPU

---

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/DuyhocAI/tri_agent_local.git
cd tri_agent_local

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Kéo model (ví dụ)
ollama pull llama3.1:8b
ollama pull mistral

# 4. (Tùy chọn) Cấu hình mật khẩu HCI
$env:HERMES_HCI_PASSWORD_HASH = python -c "import hashlib; print(hashlib.sha256(b'matkhau').hexdigest())"

# 5. Chạy
python server.py
```

Truy cập `http://localhost:7799` để sử dụng.

---

## Cấu hình

Tất cả cài đặt nằm trong [`config.py`](config.py):

| Biến | Mô tả |
|---|---|
| `MODEL_CANDIDATES` | Danh sách model ưu tiên cho từng role |
| `PRIMARY_OPTIONS` | Tham số Ollama cho Primary agent |
| `REVIEWER_OPTIONS` | Tham số Ollama cho Reviewer |
| `SUPERVISOR_OPTIONS` | Tham số Ollama cho Supervisor |
| `SERVER_CONFIG` | Host/port của server |
| `MEMORY_CONFIG` | Giới hạn và TTL cho memory |

Chạy `python verify_config.py` để kiểm tra cấu hình trước khi khởi động.

---

## API

| Endpoint | Method | Mô tả |
|---|---|---|
| `/api/chat` | POST | Chat streaming (SSE) |
| `/api/build` | POST | Sinh code theo yêu cầu (SSE) |
| `/api/build/cancel` | POST | Hủy build đang chạy |
| `/api/memory` | GET | Xem memory của session |
| `/api/memory/clear` | POST | Xóa memory |
| `/api/system` | GET | Thống kê hệ thống |
| `/api/models` | GET | Danh sách model đã resolve |
| `/hci/` | GET | HCI Dashboard |

---

## Cấu trúc thư mục

```
hermes/
├── agents/          # BaseAgent và các specialist agent
├── core/            # Orchestrator, SystemMonitor, ResourceGuard, Retry
├── hci/             # HCI Blueprint (dashboard, auth, terminal, API)
├── memory/          # MemoryManager + HermesMemory
├── skills/          # Skill definitions (YAML) + SkillRegistry
├── subconscious/    # SubconsciousScheduler (tác vụ nền)
├── templates/       # Frontend HTML/JS/CSS
├── tests/           # Unit tests
├── config.py        # Cấu hình toàn hệ thống
├── server.py        # Entry point
└── requirements.txt
```
