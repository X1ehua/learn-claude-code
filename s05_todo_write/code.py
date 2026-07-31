#!/usr/bin/env python3
"""
s05: TodoWrite — add a planning tool on top of s04 hooks.

  +---------+      +-------+      +------------------+
  |  User   | ---> |  LLM  | ---> | TOOL_HANDLERS    |
  | prompt  |      |       |      |  bash            |
  +---------+      +---+---+      |  read_file       |
                        ^         |  write_file      |
                        | result  |  edit_file       |
                        +---------+  glob            |
                                      todo_write ← NEW
                                   +------------------+
                                        |
                         in-memory current_todos
                                        |
                        if rounds_since_todo >= 3:
                          inject <reminder>

Changes from s04:
  + todo_write tool + run_todo_write() implementation
  + Nag reminder (inject reminder after 3 rounds without todo update)
  + SYSTEM prompt includes "plan before execute" guidance
  + rounds_since_todo counter in agent_loop
  Loop unchanged: new tool auto-dispatches via TOOL_HANDLERS.

Run: python s05_todo_write/code.py
Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""

import ast, json, os, subprocess
from datetime import datetime
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
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
CURRENT_TODOS: list[dict] = []

# s05 change: SYSTEM prompt adds planning guidance
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s04 (unchanged): Tool Implementations
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

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
#  FROM s02: 观察用辅助函数（时间戳 + JSON 序列化）
# ═══════════════════════════════════════════════════════════

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
#  NEW in s05: todo_write tool — plan only, no execution
# ═══════════════════════════════════════════════════════════

def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None

def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    # s05: new tool
    {
        "name": "todo_write",
        "description": "Create and manage a task list for your current coding session.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "todos": {
                    "type": "array", 
                    "items": {
                        "type": "object", 
                        "properties": {
                            "content": {"type": "string"}, 
                            "status": {
                                "type": "string", 
                                "enum": ["pending", "in_progress", "completed"]
                            }
                        }, 
                        "required": ["content", "status"]
                    }
                }
            }, 
            "required": ["todos"]
        }
    },
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
}


# ═══════════════════════════════════════════════════════════
#  FROM s04 (unchanged): Hook System
# ═══════════════════════════════════════════════════════════

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None

# s04 hooks preserved
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]

def permission_hook(block):
    """PreToolUse: deny list check."""
    if block.name == "bash":
        for p in DENY_LIST:
            if p in block.input.get("command", ""):
                print(f"\n\033[31m⛔ Blocked: '{p}'\033[0m")
                return "Permission denied"
    return None

def log_hook(block):
    """PreToolUse: log tool calls."""
    print(f"\033[{get_formatted_ts()} >>> [90m[HOOK] {block.name}\033[0m")
    return None

def context_inject_hook(query: str):
    """UserPromptSubmit: log working directory."""
    print(f"\033[{get_formatted_ts()} >>> [90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

def summary_hook(messages: list):
    """Stop: print tool call count."""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[{get_formatted_ts()} >>> [90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("Stop", summary_hook)


# ═══════════════════════════════════════════════════════════
#  FROM s02: Token 差分分析 — 用 4 个 max_tokens=1 的请求
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
#  agent_loop — same as s04 + nag reminder counter
# ═══════════════════════════════════════════════════════════

rounds_since_todo = 0

def agent_loop(messages: list):
    global rounds_since_todo
    counter = 1
    while True:
        # s05: nag reminder — inject if model hasn't updated todos for 3 rounds
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

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
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        rounds_since_todo += 1
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            print('\n>>>>>>>> Tool Use <<<<<<<<<<')
            print(f'{get_formatted_ts()}', flush=True)
            print(f"\033[33m> {block.name} {block.input}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print('\n>>>>>>>> Tool Result <<<<<<<<<<')
            print(f'{get_formatted_ts()}', flush=True)
            print(str(output)[:200])

            trigger_hooks("PostToolUse", block, output)

            # s05: reset nag counter when todo_write is called
            if block.name == "todo_write":
                rounds_since_todo = 0

            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output})

        messages.append({"role": "user", "content": results})
        counter += 1


if __name__ == "__main__":
    print("s05: TodoWrite — plan before execute, nag if you forget")
    print("Type a question, press Enter. Type q to quit.\n")

    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
