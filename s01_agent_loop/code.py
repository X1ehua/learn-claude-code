#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.

Usage:
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""

import os
import subprocess
from datetime import datetime
import json

try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
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

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── Tool definition: just bash ────────────────────────────
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# ── Tool execution ────────────────────────────────────────
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def print_timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _json_default(obj):
    """json.dumps 的兜底：把 Pydantic 对象（如 TextBlock/ToolUseBlock）转成 dict。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def format_response(response) -> str:
    """把 Anthropic Message 对象格式化成易读的文本（保留所有字段）。"""
    msg_fields = [
        "id", "container", "model", "role",
        "stop_reason", "stop_sequence", "stop_details",
        "type", "base_resp",
    ]
    usage_fields = [
        "input_tokens", "output_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens",
        "cache_creation", "inference_geo",
        "output_tokens_details", "server_tool_use",
        "service_tier",
    ]

    lines = ["Message"]
    for f in msg_fields:
        lines.append(f"  {f:<22}: {getattr(response, f, None)!r}")

    lines += ["", "Usage"]
    usage = response.usage
    for f in usage_fields:
        lines.append(f"  {f:<28}: {getattr(usage, f, None)!r}")

    lines += ["", f"Content [{len(response.content)} blocks]"]
    for i, block in enumerate(response.content, 1):
        btype = getattr(block, "type", "?")
        cls_name = type(block).__name__
        lines += ["", f"  [{i}] {cls_name} (type={btype})"]
        if btype == "text":
            lines.append(f"      citations: {getattr(block, 'citations', None)!r}")
            lines.append("      text:")
            for ln in str(getattr(block, "text", "")).splitlines() or [""]:
                lines.append(f"        {ln}")
        elif btype == "tool_use":
            lines.append(f"      id:     {getattr(block, 'id', None)!r}")
            lines.append(f"      name:   {getattr(block, 'name', None)!r}")
            lines.append(f"      caller: {getattr(block, 'caller', None)!r}")
            lines.append("      input:")
            for k, v in (getattr(block, "input", None) or {}).items():
                lines.append(f"        {k}: {v!r}")
        else:
            # 未知 block 类型：列出所有属性
            for attr in sorted(vars(block).keys()) if hasattr(block, "__dict__") else []:
                lines.append(f"      {attr}: {getattr(block, attr)!r}")

    return "\n".join(lines)

# ── The core pattern: a while loop that calls tools until the model stops ──
def agent_loop(messages: list):
    counter = 1
    while True:
        print(f'\n\n{print_timestamp()} >>>', counter, flush=True)
        print('>>> messages:', json.dumps(messages, indent=4, ensure_ascii=False, default=_json_default), flush=True)
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        print(f'\n{print_timestamp()} >>> response:', flush=True)
        print(json.dumps(response.model_dump(), indent=4, ensure_ascii=False), flush=True)
        print('\n')
        # print(format_response(response), flush=True)
        # print('\n')

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return

        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print('\n>>>>>>>> Tool Use <<<<<<<<<<')
                print(f'{print_timestamp()}', flush=True)
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print('\n>>>>>>>> Tool Result <<<<<<<<<<')
                print(f'{print_timestamp()}', flush=True)
                print(output[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # Feed tool results back, loop continues
        messages.append({"role": "user", "content": results})
        counter += 1


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
