"""
DeepSeek API Server - OpenAI Compatible with Tool Calling
Flask WSGI server (khong dung asyncio, khong conflict voi cloakbrowser)
"""

import sys
import os

# Force UTF-8 encoding
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
import logging
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

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

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
        raise ValueError("Chua cau hinh DEEPSEEK_EMAIL hoac DEEPSEEK_PASSWORD trong file .env!")
    ACCOUNTS.append({
        "email": email,
        "password": password,
        "token": None,
    })

AVAILABLE_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-r1",
    "deepseek-v3",
    "qwen-plus",
    "qwen-max",
    "qwen-turbo",
    "qwen2.5-coder-32b-instruct",
    "qwen2.5-72b-instruct",
]

MODEL_ALIASES = {
    "gpt-4o": "deepseek-v4-flash",
    "gpt-4": "deepseek-v4-flash",
    "gpt-3.5-turbo": "deepseek-v4-flash",
    "o3": "deepseek-v4-pro",
    "o1": "deepseek-reasoner",
    "qwen-plus": "deepseek-v4-flash",
    "qwen-turbo": "deepseek-v4-flash",
    "qwen-max": "deepseek-v4-pro",
    "qwen2.5-coder-32b-instruct": "deepseek-v4-flash",
    "qwen2.5-72b-instruct": "deepseek-v4-pro",
    "qwen2.5-coder-7b-instruct": "deepseek-v4-flash",
    "qwen-coder-plus": "deepseek-v4-flash",
    "qwen-coder-turbo": "deepseek-v4-flash",
    "qwen-long": "deepseek-v4-pro",
}

MAX_TOOL_ROUNDS = 15

# ============================================================
# TOKEN MANAGER
# ============================================================

_account_lock = threading.Lock()
_current_account_index = 0

def get_active_token() -> str:
    global _current_account_index
    with _account_lock:
        if not ACCOUNTS:
            raise RuntimeError("Khong co tai khoan DeepSeek nao duoc cau hinh!")

        for _ in range(len(ACCOUNTS)):
            acc = ACCOUNTS[_current_account_index]
            if not acc.get("token"):
                try:
                    log.info(f"Dang login tai khoan #{_current_account_index + 1}: {acc.get('email')}")
                    token = login(email=acc.get("email"), password=acc.get("password"))
                    acc["token"] = token
                    log.info(f"Login OK: {token[:20]}...")
                except Exception as e:
                    log.error(f"Tai khoan #{_current_account_index + 1} ({acc.get('email')}) login loi: {e}")
                    _current_account_index = (_current_account_index + 1) % len(ACCOUNTS)
                    continue

            token = acc["token"]
            _current_account_index = (_current_account_index + 1) % len(ACCOUNTS)
            return token

        raise RuntimeError("Tat ca tai khoan DeepSeek deu dang nhap that bai!")

def invalidate_token(token: str = None):
    with _account_lock:
        if token:
            for acc in ACCOUNTS:
                if acc.get("token") == token:
                    log.info(f"Invalidate token: {acc.get('email')}")
                    acc["token"] = None
                    break
        else:
            for acc in ACCOUNTS:
                acc["token"] = None

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

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
    """Parse XML function_call tags from model output.
    
    Supports two formats:
    1. <function_call name="tool_name"><args>JSON</args></function_call>
    2. <tool_name><arg1>val1</arg1><arg2>val2</arg2></tool_name>  (legacy)
    """
    tools = []
    
    # Format 1: <function_call name="xxx"><args>json</args></function_call>
    pattern1 = re.compile(
        r'<function_call\s+name\s*=\s*"(\w+)"\s*>\s*'
        r'<args>(.*?)</args>\s*'
        r'</function_call>',
        re.DOTALL
    )
    for match in pattern1.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    
    if tools:
        return tools
    
    # Format 2: Legacy XML like <write><file_path>...</file_path><content>...</content></write>
    known_tools = {'write': ['file_path', 'content'],
                   'bash': ['command', 'description', 'run_in_background', 'sideEffects'],
                   'read': ['file_path', 'offset', 'limit'],
                   'edit': ['snippet_id', 'file_path', 'old_string', 'new_string'],
                   'AskUserQuestion': ['questions'],
                   'WebSearch': ['query'],
                   'UpdatePlan': ['plan', 'explanation']}
    
    # Format 3: <tool_calls><invoke name="X"><parameter name="Y" string="true|false">VALUE</parameter></invoke></tool_calls>
    tc_pattern = re.compile(
        r'<tool_calls>\s*(.*?)\s*</tool_calls>',
        re.DOTALL
    )
    tc_match = tc_pattern.search(text)
    if tc_match:
        inner = tc_match.group(1)
        invoke_pattern = re.compile(
            r'<invoke\s+name="(\w+)">\s*(.*?)\s*</invoke>',
            re.DOTALL
        )
        for inv in invoke_pattern.finditer(inner):
            tname = inv.group(1)
            inv_body = inv.group(2)
            args = {}
            param_pattern = re.compile(
                r'<parameter\s+name="(\w+)"(?:\s+string="(true|false)")?\s*>(.*?)</parameter>',
                re.DOTALL
            )
            for pm in param_pattern.finditer(inv_body):
                pname = pm.group(1)
                is_str = pm.group(2) != 'false'
                pval = pm.group(3).strip()
                if not is_str:
                    try:
                        pval = json.loads(pval)
                    except:
                        pass
                args[pname] = pval
            if args:
                tools.append({"name": tname, "arguments": args})
        if tools:
            return tools
    
    # Format 4: <function_call name="X"><param>value</param></function_call>
    fc_pattern = re.compile(
        r'<function_call\s+name=\"(\w+)\"\s*>\s*(.*?)\s*</function_call>',
        re.DOTALL
    )
    for fc_match in fc_pattern.finditer(text):
        tname = fc_match.group(1)
        inner = fc_match.group(2)
        args = {}
        for aname in known_tools.get(tname, []):
            ap = re.compile(rf'<{aname}>(.*?)</{aname}>', re.DOTALL)
            am = ap.search(inner)
            if am:
                val = am.group(1).strip()
                if val.startswith('[') or val.startswith('{'):
                    try: val = json.loads(val)
                    except: pass
                args[aname] = val
        if args:
            tools.append({"name": tname, "arguments": args})
    if tools:
        return tools

    for tool_name, arg_names in known_tools.items():
        pattern2 = re.compile(
            rf'<{tool_name}>\s*(.*?)\s*</{tool_name}>',
            re.DOTALL
        )
        for match in pattern2.finditer(text):
            inner = match.group(1)
            args = {}
            for aname in arg_names:
                ap = re.compile(rf'<{aname}>(.*?)</{aname}>', re.DOTALL)
                am = ap.search(inner)
                if am:
                    val = am.group(1).strip()
                    # Try to parse as JSON if it looks like JSON
                    if val.startswith('[') or val.startswith('{'):
                        try:
                            val = json.loads(val)
                        except json.JSONDecodeError:
                            pass
                    args[aname] = val
            if args:
                tools.append({"name": tool_name, "arguments": args})
    
    return tools


def strip_tool_calls(text: str) -> str:
    """Remove tool call XML from text, return clean content."""
    text = re.sub(r'<function_call\s+name\s*=\s*"[^"]*"\s*>.*?</function_call>', '', text, flags=re.DOTALL)
    for tool_name in ['write', 'bash', 'read', 'edit', 'AskUserQuestion', 'WebSearch', 'UpdatePlan']:
        text = re.sub(rf'<{tool_name}>.*?</{tool_name}>', '', text, flags=re.DOTALL)
    return text.strip()


# ============================================================
# PROMPT BUILDER (with tool support)
# ============================================================

def _format_tool_schema(tool_def: dict) -> str:
    """Format a single OpenAI tool definition into XML description for system prompt."""
    func = tool_def.get("function", {})
    name = func.get("name", "unknown")
    desc = func.get("description", "")
    params = func.get("parameters", {}).get("properties", {})
    required = func.get("parameters", {}).get("required", [])

    lines = [f'<tool name="{name}">']
    lines.append(f'  <description>{desc}</description>')
    lines.append('  <parameters>')
    for pname, pinfo in params.items():
        req = ' (required)' if pname in required else ''
        pdesc = pinfo.get('description', '')
        ptype = pinfo.get('type', 'string')
        enum_hint = ''
        if 'enum' in pinfo:
            enum_hint = f' [allowed: {", ".join(repr(e) for e in pinfo["enum"])}]'
        lines.append(f'    <{pname} type="{ptype}"{req}>{pdesc}{enum_hint}</{pname}>')
    lines.append('  </parameters>')
    lines.append('</tool>')
    return '\n'.join(lines)


def build_tool_system_prompt(tools: list) -> str:
    """Build system prompt addition that describes available tools."""
    if not tools:
        return ""
    
    tool_descs = [_format_tool_schema(t) for t in tools]
    tool_names = [t.get("function", {}).get("name", "?") for t in tools]
    return f"""## Tools: {', '.join(tool_names)}

**CRITICAL: Output ONLY the XML tag. Start with <. No text before. No text after. VIOLATION = FAILURE.**

**TIPS for finding files/projects:**
- If looking for a project/directory, use: find /home/vps2 -maxdepth 3 -type d -iname '*NAME*' 2>/dev/null
- If looking for a file, use: find /home/vps2 -maxdepth 4 -type f -iname '*NAME*' 2>/dev/null
- Always search broadly first, then narrow down.

<write>
<file_path>/path</file_path>
<content>text</content>
</write>

<bash>
<command>cmd</command>
<description>what</description>
</bash>

<read>
<file_path>/path</file_path>
</read>

{chr(10).join(tool_descs)}"""


def build_prompt(messages: list, tools: list = None) -> str:
    """Build a text prompt from OpenAI-format messages, with optional tool support."""
    parts = []
    tool_system_added = False

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle content as list (multimodal)
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
            parts.append(f"<system>\n{content}\n</system>")
            # Inject tool descriptions after the last system message
            if not tool_system_added and tools:
                tool_prompt = build_tool_system_prompt(tools)
                if tool_prompt:
                    parts.append(f"<system>\n{tool_prompt}\n</system>")
                tool_system_added = True
        elif role == "user":
            parts.append(f"Human: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                # Format tool calls for the model to see
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    if isinstance(args_str, dict):
                        args_str = json.dumps(args_str)
                    parts.append(f"Assistant: <function_call name=\"{func.get('name', '')}\">\n<args>\n{args_str}\n</args>\n</function_call>")
            else:
                parts.append(f"Assistant: {content}")
        elif role == "tool":
            # Tool result
            tool_call_id = msg.get("tool_call_id", "")
            parts.append(f"Human: [Tool result for call {tool_call_id}]\n{content}")

    # If no system message existed but tools are present, add them
    if tools and not tool_system_added:
        tool_prompt = build_tool_system_prompt(tools)
        if tool_prompt:
            parts.insert(0, f"<system>\n{tool_prompt}\n</system>")

    parts.append("Assistant:" + (" <" if tools else ""))  # Force XML only when tools
    return "\n\n".join(parts)


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model.strip().lower(), model.strip())

# ============================================================
# SSE FORMATTERS
# ============================================================

def make_chunk(completion_id: str, model: str, delta: dict,
               finish_reason=None) -> str:
    obj = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def make_tool_call_chunk(completion_id: str, model: str,
                         tool_calls: list) -> str:
    """Create SSE chunk with tool_calls delta."""
    deltas = []
    for i, tc in enumerate(tool_calls):
        deltas.append({
            "index": i,
            "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
            }
        })
    
    obj = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": deltas
            },
            "finish_reason": "tool_calls",
        }],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ============================================================
# STREAM GENERATOR (with tool support)
# ============================================================

def _do_deepseek_call(token: str, prompt: str, model: str,
                      thinking_enabled: bool):
    """Call DeepSeek and collect full response text. Returns dict with text and metadata."""
    sess = make_session()
    session_id = None
    msg_id = 0
    last_status = ""
    
    try:
        session_id = create_session(token, session=sess)
        pow_resp = get_pow(token, session=sess)

        lines = call_completion(
            token=token, session_id=session_id, prompt=prompt,
            model=model, thinking=thinking_enabled,
            pow_response=pow_resp, http_session=sess,
        )

        full_text = ""
        for chunk in parse_sse_lines(lines):
            if chunk.get("response_message_id"):
                msg_id = int(chunk["response_message_id"])

            p = chunk.get("p", "")
            v = chunk.get("v")

            if "status" in p and isinstance(v, str):
                last_status = v
            if "auto_continue" in p and v is True:
                last_status = "AUTO_CONTINUE"

            if isinstance(v, str) and "content" in p:
                full_text += v  # Keep everything, filter later

        # Auto-continue
        for rnd in range(8):
            if last_status.upper() not in ("INCOMPLETE", "AUTO_CONTINUE"):
                break
            if msg_id <= 0:
                break
            log.info(f"auto_continue round {rnd+1}, msg_id={msg_id}")
            pow2 = get_pow(token, session=sess)
            cont = call_continue(token, session_id, msg_id,
                                 pow_response=pow2, http_session=sess)
            last_status = ""
            for chunk in parse_sse_lines(cont):
                if chunk.get("response_message_id"):
                    msg_id = int(chunk["response_message_id"])
                p = chunk.get("p", "")
                v = chunk.get("v")
                if "status" in p and isinstance(v, str):
                    last_status = v
                if isinstance(v, str) and "content" in p:
                    full_text += v  # Keep everything, filter later

        return {"text": full_text, "finish_reason": "stop"}

    finally:
        pass  # Keep session for history



def _stream_deepseek_realtime(token: str, prompt: str, model: str,
                               thinking_enabled: bool, completion_id: str):
    """Real streaming: yield each token as it arrives from DeepSeek."""
    sess = make_session()
    session_id = None
    msg_id = 0
    last_status = ""
    full_text = ""
    
    try:
        session_id = create_session(token, session=sess)
        pow_resp = get_pow(token, session=sess)

        lines = call_completion(
            token=token, session_id=session_id, prompt=prompt,
            model=model, thinking=thinking_enabled,
            pow_response=pow_resp, http_session=sess,
        )

        for chunk in parse_sse_lines(lines):
            if chunk.get("response_message_id"):
                msg_id = int(chunk["response_message_id"])

            p = chunk.get("p", "")
            v = chunk.get("v")

            if "status" in p and isinstance(v, str):
                last_status = v
            if "auto_continue" in p and v is True:
                last_status = "AUTO_CONTINUE"

            if isinstance(v, str) and "content" in p:
                full_text += v
                yield ("token", v)

        # Auto-continue
        for rnd in range(8):
            if last_status.upper() not in ("INCOMPLETE", "AUTO_CONTINUE"):
                break
            if msg_id <= 0:
                break
            log.info(f"auto_continue round {rnd+1}, msg_id={msg_id}")
            pow2 = get_pow(token, session=sess)
            cont = call_continue(token, session_id, msg_id,
                                 pow_response=pow2, http_session=sess)
            last_status = ""
            for chunk in parse_sse_lines(cont):
                if chunk.get("response_message_id"):
                    msg_id = int(chunk["response_message_id"])
                p = chunk.get("p", "")
                v = chunk.get("v")
                if "status" in p and isinstance(v, str):
                    last_status = v
                if isinstance(v, str) and "content" in p:
                    full_text += v
                    yield ("token", v)

        yield ("done", full_text)

    except Exception as e:
        invalidate_token(token)
        yield ("error", str(e))
    finally:
        pass

def _yield_text_stream(completion_id: str, model: str, text: str):
    """Yield text as SSE stream chunks."""
    yield make_chunk(completion_id, model, {"role": "assistant", "content": ""})
    # Yield in chunks to simulate streaming
    chunk_size = 8
    for i in range(0, len(text), chunk_size):
        yield make_chunk(completion_id, model, {"content": text[i:i+chunk_size]})
    yield make_chunk(completion_id, model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


def stream_with_tools(token: str, msgs: list, model: str,
                      thinking_enabled: bool, completion_id: str,
                      tools: list = None):
    """Stream response. If model outputs XML tool calls, convert to
    OpenAI streaming tool_calls format (multiple chunks)."""
    
    prompt = build_prompt(msgs, tools)
    
    try:
        result = _do_deepseek_call(token, prompt, model, thinking_enabled)
    except Exception as e:
        invalidate_token(token)
        err = {"error": {"type": "api_error", "message": str(e)}}
        yield f"data: {json.dumps(err)}\n\n"
        return

    text = result.get("text", "")
    log.info(f"Stream {len(text)} chars (tools={'yes' if tools else 'no'})")
    
    parsed = _extract_xml_tags(text)
    
    if parsed and tools:
        # Filter thinking from display text but keep tool calls
        log.info(f"Tool calls: {[t['name'] for t in parsed]}")
        # Stream tool calls in proper OpenAI format
        # Chunk 1: role + tool call declaration (OpenAI puts them together)
        tc_deltas = []
        for i, tc in enumerate(parsed):
            cid = f"call_{uuid.uuid4().hex[:12]}"
            tc_deltas.append({
                "index": i, "id": cid, "type": "function",
                "function": {"name": tc["name"], "arguments": ""}
            })
        yield make_chunk(completion_id, model, {
            "role": "assistant", "content": None,
            "tool_calls": tc_deltas
        })
        # Chunk 2-N: stream arguments for each tool call
        for i, tc in enumerate(parsed):
            args = json.dumps(tc["arguments"], ensure_ascii=False)
            yield make_chunk(completion_id, model, {
                "tool_calls": [{"index": i, "function": {"arguments": args}}]
            })
        # Finish
        yield make_chunk(completion_id, model, {}, finish_reason="tool_calls")
        yield "data: [DONE]\n\n"
    elif not tools:
        # No tools needed - use real streaming
        log.info(f"Real-time streaming (no tools)")
        try:
            yield make_chunk(completion_id, model, {"role": "assistant", "content": ""})
            for typ, val in _stream_deepseek_realtime(token, prompt, model, thinking_enabled, completion_id):
                if typ == "token":
                    yield make_chunk(completion_id, model, {"content": val})
                elif typ == "error":
                    err = {"error": {"type": "api_error", "message": val}}
                    yield f"data: {json.dumps(err)}\n\n"
                    return
            yield make_chunk(completion_id, model, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            invalidate_token(token)
            err = {"error": {"type": "api_error", "message": str(e)}}
            yield f"data: {json.dumps(err)}\n\n"
        return

    else:
        # Filter thinking content from display
        clean = text
        # Remove thinking sections if present
        import re
        clean = re.sub(r'<thinking>.*?</thinking>', '', clean, flags=re.DOTALL)
        clean = clean.strip()
        if not clean:
            clean = text.strip()  # Fallback
        yield from _yield_text_stream(completion_id, model, clean)
        yield from _yield_text_stream(completion_id, model, text)


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
    model = resolve_model(body.get("model", "deepseek-v4-flash"))
    msgs = body.get("messages", [])
    stream = bool(body.get("stream", False))
    tools = body.get("tools", [])
    
    # Always inject at least basic tools so model knows about them
    if not tools:
        tools = [
            {"type":"function","function":{"name":"write","description":"Create a file","parameters":{"type":"object","properties":{"file_path":{"type":"string","description":"Absolute path"},"content":{"type":"string","description":"File content"}},"required":["file_path","content"]}}},
            {"type":"function","function":{"name":"bash","description":"Run shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command"},"description":{"type":"string","description":"What it does"}},"required":["command"]}}},
            {"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"file_path":{"type":"string","description":"Absolute path"}},"required":["file_path"]}}},
        ]
    
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
        return Response(
            stream_with_tools(token, msgs, model, thinking_enabled,
                              completion_id, tools),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── NON-STREAM MODE ──
    prompt = build_prompt(msgs, tools)

    sess = make_session()
    result = {}
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

    text = result.get("text", "")
    nctool_calls = _extract_xml_tags(text)
    final_text = strip_tool_calls(text).strip() if nctool_calls else text
    if nctool_calls:
        log.info(f"Non-stream: found tool calls: {[t['name'] for t in nctool_calls]}")

    prompt_tokens = len(prompt) // 4
    completion_tokens = len(final_text) // 4

    resp = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_text},
            "finish_reason": "tool_calls" if nctool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    if nctool_calls:
        openai_tool_calls = []
        for tc in nctool_calls:
            openai_tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                }
            })
        resp["choices"][0]["message"]["tool_calls"] = openai_tool_calls

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
    print(f"Tools:    enabled (XML parsing)")
    print("=" * 50)

    log.info("Khoi dong trinh duyet va dang nhap DeepSeek...")
    threading.Thread(target=get_active_token, daemon=True).start()

    app.run(
        host=host,
        port=port,
        threaded=True,
        debug=False,
    )
