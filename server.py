"""
DeepSeek API Server - OpenAI Compatible with Tool Calling
Flask WSGI server (không dùng asyncio, không conflict với cloakbrowser)
"""

import sys
import os

# Force UTF-8 encoding for stdout and stderr on Windows to avoid UnicodeEncodeError
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key] = val

load_env()

import json
import time
import uuid
import threading
import re
from flask import Flask, request, Response, jsonify

from deepseek_client import (
    login, create_session, get_pow,
    call_completion, call_continue,
    delete_session, parse_sse_lines,
    collect_response, make_session, get_model_type,
)

# ============================================================
# CONFIG
# ============================================================

VALID_API_KEYS = {
    os.environ.get("API_KEY", "sk-my-secret-key-1"),
}

ACCOUNTS = []
accounts_env = os.environ.get("DEEPSEEK_ACCOUNTS", "")
if accounts_env:
    for acc_str in accounts_env.split(","):
        acc_str = acc_str.strip()
        if ":" in acc_str:
            parts = acc_str.split(":", 1)
            ACCOUNTS.append({
                "email": parts[0].strip(),
                "password": parts[1].strip(),
                "token": None
            })

if not ACCOUNTS:
    email = os.environ.get("DEEPSEEK_EMAIL", "").strip()
    password = os.environ.get("DEEPSEEK_PASSWORD", "").strip()
    if not email or not password:
        raise ValueError("LỖI: Chưa cấu hình DEEPSEEK_EMAIL hoặc DEEPSEEK_PASSWORD trong file .env!")
    ACCOUNTS.append({
        "email":    email,
        "password": password,
        "token":    None,
    })

AVAILABLE_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-r1",
    "deepseek-v3",
    # Qwen aliases (cho Qwen Code Companion)
    "qwen-plus",
    "qwen-max",
    "qwen-turbo",
    "qwen2.5-coder-32b-instruct",
    "qwen2.5-72b-instruct",
]

MODEL_ALIASES = {
    # OpenAI aliases
    "gpt-4o":        "deepseek-v4-flash",
    "gpt-4":         "deepseek-v4-flash",
    "gpt-3.5-turbo": "deepseek-v4-flash",
    "o3":            "deepseek-v4-pro",
    "o1":            "deepseek-reasoner",
    # Qwen Code Companion aliases → DeepSeek models
    "qwen-plus":                     "deepseek-v4-flash",
    "qwen-turbo":                    "deepseek-v4-flash",
    "qwen-max":                      "deepseek-v4-pro",
    "qwen2.5-coder-32b-instruct":    "deepseek-v4-flash",
    "qwen2.5-72b-instruct":          "deepseek-v4-pro",
    "qwen2.5-coder-7b-instruct":     "deepseek-v4-flash",
    "qwen-coder-plus":               "deepseek-v4-flash",
    "qwen-coder-turbo":              "deepseek-v4-flash",
    "qwen-long":                     "deepseek-v4-pro",
}

# ============================================================
# TOKEN MANAGER
# ============================================================

_account_lock = threading.Lock()
_current_account_index = 0


def prelogin_all_accounts():
    """Login all accounts in background at startup to avoid first-request timeout."""
    def _login_all():
        for i, acc in enumerate(ACCOUNTS):
            if acc.get("token"):
                continue
            try:
                print(f"[auth] Pre-login tai khoan #{i+1}: {acc.get('email')}")
                token = login(email=acc.get("email"), password=acc.get("password"))
                acc["token"] = token
                print(f"[auth] Pre-login OK #{i+1}: {token[:20]}...")
            except Exception as e:
                print(f"[auth] Pre-login loi #{i+1} ({acc.get('email')}): {e}")
    threading.Thread(target=_login_all, daemon=True).start()

def get_active_token() -> str:
    global _current_account_index
    with _account_lock:
        if not ACCOUNTS:
            raise RuntimeError("Không có tài khoản DeepSeek nào được cấu hình!")
            
        for _ in range(len(ACCOUNTS)):
            acc = ACCOUNTS[_current_account_index]
            if not acc.get("token"):
                try:
                    print(f"[auth] Đang login tài khoản #{_current_account_index + 1}: {acc.get('email')}")
                    token = login(
                        email=acc.get("email"),
                        password=acc.get("password")
                    )
                    acc["token"] = token
                    print(f"[auth] Login OK cho tài khoản #{_current_account_index + 1}: {token[:20]}...")
                except Exception as e:
                    print(f"[auth] Tài khoản #{_current_account_index + 1} ({acc.get('email')}) đăng nhập lỗi: {e}")
                    # Chuyển sang tài khoản tiếp theo nếu tài khoản này lỗi đăng nhập
                    _current_account_index = (_current_account_index + 1) % len(ACCOUNTS)
                    continue
            
            token = acc["token"]
            # Xoay vòng tài khoản cho lần gọi tiếp theo
            _current_account_index = (_current_account_index + 1) % len(ACCOUNTS)
            return token
            
        raise RuntimeError("Tất cả các tài khoản DeepSeek được cấu hình đều đăng nhập thất bại!")

def invalidate_token(token: str = None):
    with _account_lock:
        if token:
            for acc in ACCOUNTS:
                if acc.get("token") == token:
                    print(f"[auth] Invalidate token của tài khoản: {acc.get('email')}")
                    acc["token"] = None
                    break
        else:
            for acc in ACCOUNTS:
                acc["token"] = None

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# CORS support for web clients
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ============================================================
# AUTH
# ============================================================

def get_caller_key():
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        key = auth[7:].strip()
        if key:
            return key
    key = request.headers.get("X-Api-Key", "").strip()
    if key:
        return key
    return None

def require_auth():
    key = get_caller_key()
    if not key:
        return jsonify({"error": {"message": "Missing API key", "type": "invalid_request_error"}}), 401
    if VALID_API_KEYS and key not in VALID_API_KEYS:
        return jsonify({"error": {"message": "Invalid API key", "type": "invalid_request_error"}}), 401
    return None

# ============================================================
# TOOL CALL PARSER
# ============================================================

def _extract_xml_tags(text: str) -> list:
    """Parse <function_call name="X"><args>JSON</args></function_call> from text."""
    tools = []
    pattern = re.compile(
        r'<function_call\s+name\s*=\s*"(\w+)"\s*>\s*'
        r'<args>(.*?)</args>\s*'
        r'</function_call>',
        re.DOTALL
    )
    for match in pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    return tools


def strip_tool_calls(text: str) -> str:
    """Remove <function_call> XML blocks from text."""
    return re.sub(
        r'<function_call\s+name\s*=\s*"[^"]*"\s*>.*?</function_call>',
        '', text, flags=re.DOTALL
    ).strip()

# ============================================================
# PROMPT BUILDER (with optional tool support)
# ============================================================

def _build_tool_system_prompt(tools: list) -> str:
    """Build system prompt fragment describing available tools in XML format."""
    if not tools:
        return ""

    lines = ["## Available Tools"]
    lines.append("To use a tool, output EXACTLY:")
    lines.append('<function_call name="tool_name">')
    lines.append("<args>")
    lines.append('{"param1": "value1", "param2": "value2"}')
    lines.append("</args>")
    lines.append("</function_call>")
    lines.append("")
    lines.append("Available tools:")

    for t in tools:
        func = t.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        lines.append(f"\n### {name}")
        lines.append(f"Description: {desc}")
        if params:
            lines.append("Parameters:")
            for pname, pinfo in params.items():
                req_mark = " (required)" if pname in required else ""
                pdesc = pinfo.get("description", "")
                lines.append(f"  - {pname}: {pdesc}{req_mark}")

    return "\n".join(lines)



def _has_xml_tools(messages: list) -> bool:
    xml_tool_pattern = re.compile(r'<(write|bash|read|edit|AskUserQuestion|WebSearch|UpdatePlan)>', re.IGNORECASE)
    for msg in messages:
        if msg.get('role') == 'system':
            content = msg.get('content', '')
            if isinstance(content, list):
                content = chr(10).join(item.get('text', '') for item in content if isinstance(item, dict))
            if xml_tool_pattern.search(str(content)):
                return True
    return False

XML_TOOL_INSTRUCTION = "\n\n**CRITICAL: When you need to use a tool, output ONLY the XML tag. No text before, no text after. Start with < and end with >.**"

def build_prompt(messages: list, tools: list = None) -> str:
    """Build a text prompt from OpenAI-format messages, with optional tool support."""
    parts = []
    tool_prompt_inserted = False
    has_explicit_tools = bool(tools)
    has_implicit_tools = not has_explicit_tools and _has_xml_tools(messages)
    has_tools = has_explicit_tools or has_implicit_tools

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = "\n".join(texts)
        elif not isinstance(content, str):
            content = str(content)

        if role == "system":
            if has_implicit_tools and not tool_prompt_inserted:
                content = content + XML_TOOL_INSTRUCTION
                tool_prompt_inserted = True
            parts.append(f"<system>\n{content}\n</system>")
            # Inject tool descriptions after the last system message (only if tools provided)
            if has_explicit_tools and not tool_prompt_inserted:
                tool_prompt = _build_tool_system_prompt(tools)
                if tool_prompt:
                    parts.append(f"<system>\n{tool_prompt}\n</system>")
                tool_prompt_inserted = True
        elif role == "user":
            parts.append(f"Human: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    if isinstance(args_str, dict):
                        args_str = json.dumps(args_str)
                    parts.append(
                        f'Assistant: <function_call name="{func.get("name", "")}">\n'
                        f'<args>\n{args_str}\n</args>\n'
                        f'</function_call>'
                    )
            else:
                parts.append(f"Assistant: {content}")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            parts.append(f"Human: [Tool result for call {tool_call_id}]\n{content}")

    # Insert tool prompt at beginning if no system message existed
    if has_tools and not tool_prompt_inserted:
        tool_prompt = _build_tool_system_prompt(tools)
        if tool_prompt:
            parts.insert(0, f"<system>\n{tool_prompt}\n</system>")

    # If tools are present and no tool history in messages, hint the model to output XML
    has_tool_history = any(
        msg.get("role") == "tool" or
        (msg.get("role") == "assistant" and msg.get("tool_calls"))
        for msg in messages
    )
    suffix = " <function_call name=\"" if (has_explicit_tools and not has_tool_history) else (" <" if has_implicit_tools else "")
    parts.append(f"Assistant:{suffix}")
    return "\n\n".join(parts)


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model.strip().lower(), model.strip())

# ============================================================
# SSE CHUNK FORMATTER
# ============================================================

def make_chunk(completion_id: str, model: str, delta: dict,
               finish_reason=None) -> str:
    obj = {
        "id":      completion_id,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def make_tool_call_chunk(completion_id: str, model: str,
                         index: int, call_id: str,
                         name: str = None, arguments: str = None) -> str:
    """Create an SSE chunk for a tool call delta."""
    tc_delta = {"index": index}
    if call_id:
        tc_delta["id"] = call_id
        tc_delta["type"] = "function"
    if name is not None:
        tc_delta["function"] = {"name": name, "arguments": ""}
    if arguments is not None:
        tc_delta["function"] = {"arguments": arguments}

    delta = {"tool_calls": [tc_delta]}
    if call_id:
        delta["role"] = "assistant"
        delta["content"] = None

    obj = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

# ============================================================
# STREAM GENERATOR
# ============================================================

def stream_generator(token: str, prompt: str, model: str,
                     thinking_enabled: bool, completion_id: str):
    """Generator yield SSE strings theo OpenAI format"""

    sess = make_session()
    yield make_chunk(completion_id, model, {"role": "assistant", "content": ""})

    session_id     = None
    msg_id         = 0
    last_status    = ""

    try:
        session_id = create_session(token, session=sess)
        pow_resp   = get_pow(token, session=sess)

        lines = call_completion(
            token=token, session_id=session_id, prompt=prompt,
            model=model, thinking=thinking_enabled,
            pow_response=pow_resp, http_session=sess,
        )

        def consume(lines_gen):
            nonlocal msg_id, last_status
            for chunk in parse_sse_lines(lines_gen):
                if chunk.get("response_message_id"):
                    msg_id = int(chunk["response_message_id"])

                p = chunk.get("p", "")
                v = chunk.get("v")

                if "status" in p and isinstance(v, str):
                    last_status = v
                if "auto_continue" in p and v is True:
                    last_status = "AUTO_CONTINUE"

                if isinstance(v, str) and "content" in p:
                    if "thinking" in p.lower():
                        yield make_chunk(completion_id, model, {"content": v})
                    else:
                        yield make_chunk(completion_id, model, {"content": v})

        yield from consume(lines)

        # Auto-continue
        for rnd in range(8):
            if last_status.upper() not in ("INCOMPLETE", "AUTO_CONTINUE"):
                break
            if msg_id <= 0:
                break
            print(f"[auto_continue] round {rnd+1}, msg_id={msg_id}")
            pow2 = get_pow(token, session=sess)
            cont = call_continue(token, session_id, msg_id,
                                 pow_response=pow2, http_session=sess)
            last_status = ""
            yield from consume(cont)

        yield make_chunk(completion_id, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        invalidate_token(token)
        err = {"error": {"type": "api_error", "message": str(e)}}
        yield f"data: {json.dumps(err)}\n\n"
    finally:
        # Giữ lại lịch sử chat trên DeepSeek, không xóa session
        pass


def stream_with_tools(token: str, msgs: list, model: str,
                      thinking_enabled: bool, completion_id: str,
                      tools: list):
    """Stream response with tool call detection.
    Uses background thread to collect response + heartbeat keep-alive to prevent timeout."""
    prompt = build_prompt(msgs, tools)
    sess = make_session()

    result_container = []
    error_container = []

    def _collect():
        try:
            sid = create_session(token, session=sess)
            result_container.append(collect_response(
                token=token, session_id=sid, prompt=prompt,
                model=model, thinking=thinking_enabled, http_session=sess,
            ))
        except Exception as e:
            error_container.append(e)

    t = threading.Thread(target=_collect, daemon=True)
    t.start()

    # Keep-alive heartbeat every 3s while waiting (max 180s)
    for _ in range(60):
        if result_container or error_container:
            break
        yield ": keepalive\n\n"
        t.join(timeout=3.0)

    t.join(timeout=10.0)

    if error_container:
        invalidate_token(token)
        err = {"error": {"type": "api_error", "message": str(error_container[0])}}
        yield f"data: {json.dumps(err)}\n\n"
        return

    if not result_container:
        err = {"error": {"type": "api_error", "message": "Request timed out"}}
        yield f"data: {json.dumps(err)}\n\n"
        return

    result = result_container[0]
    text = result.get("text", "")
    tool_calls = _extract_xml_tags(text)

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            cid = f"call_{uuid.uuid4().hex[:12]}"
            yield make_tool_call_chunk(completion_id, model, i, cid, name=tc["name"])
            args = json.dumps(tc["arguments"], ensure_ascii=False)
            yield make_tool_call_chunk(completion_id, model, i, "", arguments=args)

        yield make_chunk(completion_id, model, {}, finish_reason="tool_calls")
        yield "data: [DONE]\n\n"
    else:
        clean = strip_tool_calls(text).strip()
        if clean:
            yield from _yield_text_stream(completion_id, model, clean)
        else:
            yield from _yield_text_stream(completion_id, model, text)


def _yield_text_stream(completion_id: str, model: str, text: str):
    """Yield text as SSE stream chunks (simulated streaming)."""
    yield make_chunk(completion_id, model, {"role": "assistant", "content": ""})
    chunk_size = 8
    for i in range(0, len(text), chunk_size):
        yield make_chunk(completion_id, model, {"content": text[i:i+chunk_size]})
    yield make_chunk(completion_id, model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================
# ROUTES
# ============================================================

@app.get("/healthz")
@app.get("/readyz")
def health():
    return jsonify({"status": "ok"})


@app.get("/v1/models")
@app.get("/models")
def list_models():
    err = require_auth()
    if err:
        return err
    data = [
        {"id": m, "object": "model", "created": 1700000000, "owned_by": "deepseek"}
        for m in AVAILABLE_MODELS
    ]
    return jsonify({"object": "list", "data": data})


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
def chat_completions():
    err = require_auth()
    if err:
        return err

    body = request.get_json(force=True, silent=True) or {}
    model   = resolve_model(body.get("model", "deepseek-v4-flash"))
    msgs    = body.get("messages", [])
    stream  = bool(body.get("stream", False))
    tools   = body.get("tools", [])  # Chỉ dùng tools nếu client gửi
    thinking_flag = body.get("thinking", None)

    if not msgs:
        return jsonify({"error": {"message": "messages required"}}), 400

    thinking_enabled = bool(thinking_flag) if thinking_flag is not None \
                       else (get_model_type(model) == "reasoner")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    try:
        token = get_active_token()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": {"message": f"Auth failed: {e}"}}), 500

    # ── STREAM MODE ──
    if stream:
        if tools:
            return Response(
                stream_with_tools(token, msgs, model, thinking_enabled,
                                  completion_id, tools),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control":    "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection":       "keep-alive",
                },
            )
        else:
            prompt = build_prompt(msgs)
            return Response(
                stream_generator(token, prompt, model, thinking_enabled, completion_id),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control":    "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection":       "keep-alive",
                },
            )

    # ── NON-STREAM MODE ──
    prompt = build_prompt(msgs, tools if tools else None)
    sess       = make_session()
    session_id = None
    try:
        session_id = create_session(token, session=sess)
        result = collect_response(
            token=token, session_id=session_id, prompt=prompt,
            model=model, thinking=thinking_enabled, http_session=sess,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        invalidate_token(token)
        return jsonify({"error": {"message": str(e)}}), 500
    finally:
        pass

    text = result.get("text", "")
    tool_calls = _extract_xml_tags(text)
    final_text = None if tool_calls else text

    prompt_tokens     = len(prompt) // 4
    completion_tokens = len(final_text or "") // 4

    resp = {
        "id":      completion_id,
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": final_text},
            "finish_reason": "tool_calls" if tool_calls else result.get("finish_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }

    if tool_calls:
        openai_tc = []
        for tc in tool_calls:
            openai_tc.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                }
            })
        resp["choices"][0]["message"]["tool_calls"] = openai_tc

    if result.get("thinking"):
        resp["choices"][0]["message"]["thinking"] = result["thinking"]

    return jsonify(resp)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    api_key = os.environ.get("API_KEY", "sk-my-secret-key-1")

    print("=" * 50)
    print("DeepSeek API Bridge (Flask) - OpenAI Compatible + Tool Calling")
    print("=" * 50)
    print(f"Endpoint: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/v1/chat/completions")
    print(f"Models:   http://{host if host != '0.0.0.0' else 'localhost'}:{port}/v1/models")
    print(f"API Key:  {api_key}")
    print(f"Tools:    enabled (opt-in via 'tools' param)")
    print("=" * 50)
    print("[Qwen Code Companion] Custom Provider settings:")
    print(f"  API Base URL : http://{host if host != '0.0.0.0' else 'localhost'}:{port}/v1")
    print(f"  API Key      : {api_key}")
    print("  Model        : qwen-plus  (hoặc deepseek-v4-flash)")
    print("=" * 50)
    print("[info] Khởi động trình duyệt và đăng nhập DeepSeek tự động trong nền...")
    prelogin_all_accounts()
    print("=" * 50)

    # Flask threaded=True: mỗi request chạy trong thread riêng
    # Không dùng asyncio → không conflict với cloakbrowser
    app.run(
        host=host,
        port=port,
        threaded=True,
        debug=False,
    )
