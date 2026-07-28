#!/usr/bin/env python3
"""
s02: Tool Use — 在 s01 基础上新增 4 个工具 + 分发映射。

运行: python s02_tool_use/code.py
需要: pip install anthropic python-dotenv + .env 中配置 ANTHROPIC_API_KEY

本文件 = s01 的全部代码 + 以下新增:
  + run_read / run_write / run_edit / run_glob 四个工具实现
  + TOOL_HANDLERS 分发映射（替代 s01 中硬编码的 run_bash 调用）
  + safe_path 路径安全校验

循环本身（agent_loop）与 s01 完全一致。
"""

import os, subprocess
from pathlib import Path
from datetime import datetime
import json

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# ═══════════════════════════════════════════════════════════
#  FROM s01 (unchanged)
# ═══════════════════════════════════════════════════════════

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def get_formatted_ts():
    now = datetime.now()
    last = getattr(get_formatted_ts, "_last", None)
    delta = 0.0 if last is None else (now - last).total_seconds()
    get_formatted_ts._last = now
    return f"{now.strftime('%H:%M:%S.%f')[:-3]} +{delta:.3f}"


def _json_default(obj):
    """json.dumps 的兜底：把 Pydantic 对象（如 TextBlock/ToolUseBlock）转成 dict。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# ═══════════════════════════════════════════════════════════
#  NEW in s02: 4 个新工具
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════
#  NEW in s02: 工具定义（s01 只有一个 bash，现在扩展到 5 个）
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "path": {"type": "string"}, 
                "limit": {"type": "integer"},
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "path": {"type": "string"}, 
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "path": {"type": "string"}, 
                "old_text": {"type": "string"}, 
                "new_text": {"type": "string"}
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "pattern": {"type": "string"}
            },
            "required": ["pattern"]
        }
    }
]

# ═══════════════════════════════════════════════════════════
#  NEW in s02: 工具分发映射（s01 是硬编码 run_bash，现在改为查表）
# ═══════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "bash": run_bash, 
    "read_file": run_read, 
    "write_file": run_write,
    "edit_file": run_edit, 
    "glob": run_glob,
}


# ═══════════════════════════════════════════════════════════
#  NEW in s02: Token 差分分析 — 用 4 个 max_tokens=1 的请求
#  分解 input_tokens → system / messages / tools / formatting
# ═══════════════════════════════════════════════════════════

def _measure(system: str, messages: list, tools: list) -> int:
    """发一个 max_tokens=1 的请求，返回 input_tokens（含 cache_read）。"""
    resp = client.messages.create(
        model=MODEL, system=system, messages=messages,
        tools=tools, max_tokens=1,
    )
    return resp.usage.input_tokens + (resp.usage.cache_read_input_tokens or 0)


def token_breakdown(system: str, messages: list, tools: list) -> dict:
    """差分法分解 input_tokens → {system, messages, tools, formatting, total}。

    4 个请求，只差一个变量，相减得到各部分：
      full       = system + messages + tools
      no_system  = ""      + messages + tools
      no_tools   = system  + messages + ""
      baseline   = ""      + messages + ""
      → messages = baseline
      → system   = no_tools  - baseline
      → tools    = no_system - baseline
      → formatting = full - system - messages - tools
    """
    full      = _measure(system, messages, tools)
    no_system = _measure("", messages, tools)
    no_tools  = _measure(system, messages, [])
    baseline  = _measure("", messages, [])

    msgs = baseline
    sys_t = no_tools - baseline
    tools_t = no_system - baseline
    fmt = full - sys_t - msgs - tools_t

    return {"system": sys_t, "messages": msgs, "tools": tools_t,
            "formatting": fmt, "total": full}


# ═══════════════════════════════════════════════════════════
#  agent_loop — 与 s01 结构完全一致，只改了工具执行那部分
#  s01: output = run_bash(block.input["command"])
#  s02: output = TOOL_HANDLERS[block.name](**block.input)
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    counter = 1
    while True:
        print(f'\n\n{get_formatted_ts()} >>>', counter, flush=True)
        print('>>> messages:', json.dumps(messages, indent=4, ensure_ascii=False, default=_json_default), flush=True)
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        print(f'\n{get_formatted_ts()} >>> response:', flush=True)
        print(json.dumps(response.model_dump(), indent=4, ensure_ascii=False), flush=True)

        # ── Token Breakdown（差分法，3 次额外 API 调用）──
        bd = token_breakdown(SYSTEM, messages, TOOLS)
        print(f'\n  ┌─ Token Breakdown (input_tokens = {bd["total"]}) ────────┐')
        print(f'  │ {"component":<12s}  {"tokens":>6s}  {"pct":>6s}  │')
        print(f'  │ {"─"*40} │')
        for k in ("system", "messages", "tools", "formatting"):
            pct = f'{bd[k] / bd["total"] * 100:5.1f}%' if bd["total"] else "  ---"
            print(f'  │ {k:<12s}  {bd[k]:>6d}  {pct:>6s}  │')
        print(f'  │ {"─"*40} │')
        print(f'  │ {"total":<12s}  {bd["total"]:>6d}  {"100.0%":>6s}  │')
        print(f'  │ {"output":<12s}  {response.usage.output_tokens:>6d}  {"":>6s}  │')
        print(f'  └─ {"─"*40} ┘\n')

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print('\n>>>>>>>> Tool Use <<<<<<<<<<')
                print(f'{get_formatted_ts()}', flush=True)
                print(f"\033[33m> {block.name} {block.input}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                print('\n>>>>>>>> Tool Result <<<<<<<<<<')
                print(f'{get_formatted_ts()}', flush=True)
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})
        counter += 1


if __name__ == "__main__":
    print("s02: Tool Use — 在 s01 基础上加了 4 个工具")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
