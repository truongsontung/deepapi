# DeepAPI — OpenAI-Compatible API Bridge for DeepSeek

DeepAPI cung cấp REST API tương thích OpenAI, cho phép gọi DeepSeek Chat thông qua giao diện chuẩn `/v1/chat/completions`. Hỗ trợ **tool calling** (function calling) qua XML parsing.

## Tính năng

- ✅ API tương thích OpenAI (`/v1/models`, `/v1/chat/completions`)
- ✅ **Tool Calling** — parse XML tool calls từ model output (4 định dạng)
- ✅ Streaming & Non-streaming SSE
- ✅ Multi-account rotation (xoay vòng tài khoản)
- ✅ Auto-continue khi response dài
- ✅ PoW solver qua cloakbrowser (Chromium headless)
- ✅ Hỗ trợ model: deepseek-v4-flash, deepseek-v4-pro, deepseek-chat, deepseek-reasoner...

## Cài đặt

### Yêu cầu

- Python 3.12+
- Node.js 24+
- Ubuntu 24.04 (hoặc Linux tương tự)

### 1. Cài dependencies

```bash
# Python
pip install --break-system-packages flask>=3.0.0 cloakbrowser>=0.3.30

# Node.js + Playwright
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt-get install -y nodejs
npx playwright install-deps chromium
npx playwright install chromium
```

### 2. Cấu hình tài khoản

Tạo file `.env`:

```ini
# Tài khoản DeepSeek (email:password, phân cách bằng dấu phẩy)
DEEPSEEK_ACCOUNTS=email1@gmail.com:password1,email2@gmail.com:password2

# API key cho client gọi vào server này
API_KEY=deepcode2026
PORT=5001
HOST=0.0.0.0
```

### 3. Chạy server

```bash
# Chạy trực tiếp
python3 server.py

# Hoặc dùng systemd (khuyến nghị)
sudo tee /etc/systemd/system/deepapi.service << 'EOF'
[Unit]
Description=DeepSeek API Server
After=network.target

[Service]
Type=simple
User=vps2
WorkingDirectory=/home/vps2/deepapi
Environment=PYTHONUNBUFFERED=1
Environment=TZ=Asia/Shanghai
ExecStart=/usr/bin/python3 /home/vps2/deepapi/server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now deepapi
```

## Sử dụng

### Gọi API

```bash
# Liệt kê models
curl http://localhost:5001/v1/models \
  -H "Authorization: Bearer deepcode2026"

# Chat completion
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer deepcode2026" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "stream": true,
    "messages": [{"role": "user", "content": "Xin chào"}]
  }'

# Tool calling
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer deepcode2026" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "stream": true,
    "messages": [{"role": "user", "content": "tạo file /home/user/test.txt nội dung hello"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "write",
        "description": "Create a file",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"}
          },
          "required": ["file_path", "content"]
        }
      }
    }]
  }'
```

### Cấu hình DeepCode CLI

Trong `~/.deepcode/settings.json`:

```json
{
  "env": {
    "MODEL": "deepseek-v4-pro",
    "BASE_URL": "http://localhost:5001/v1",
    "API_KEY": "deepcode2026"
  },
  "thinkingEnabled": true,
  "reasoningEffort": "max"
}
```

## Kiến trúc

```
Client → POST /v1/chat/completions (OpenAI format)
    → server.py: auth → build_prompt (inject tools) → call DeepSeek
    → deepseek_client.py: cloakbrowser → fetch() DeepSeek API
    → parse_sse_lines() → tách text, thinking, tool calls
    → Nếu có XML tool calls → convert sang OpenAI tool_calls format
    → Response: SSE stream hoặc JSON
```

## Tool Calling

Server hỗ trợ 4 định dạng XML tool calls từ model output:

1. `<function_call name="w"><args>{json}</args></function_call>`
2. `<write><file_path>...</file_path><content>...</content></write>`
3. `<tool_calls><invoke name="..."><parameter>...</parameter></invoke></tool_calls>`
4. `<function_call name="read"><file_path>...</file_path></function_call>`

Server tự động inject tool descriptions vào system prompt để model biết output XML.

## Quản lý

```bash
sudo systemctl status deepapi   # Xem trạng thái
sudo systemctl restart deepapi  # Khởi động lại
journalctl -u deepapi -f        # Xem logs
```
