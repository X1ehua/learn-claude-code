#!/usr/bin/env python3
"""
token_breakdown.py — 用差分 API 调用分解 input_tokens 的组成。

原理：API 只返回总的 input_tokens，不告诉你 system / messages / tools 各占多少。
但我们可以发 4 个只差一个变量的请求，相减就得到各部分的 token 数：

  full        = system + messages + tools  → 548
  no_system   = ""      + messages + tools
  no_tools    = system  + messages + ""
  baseline    = ""      + messages + ""

  → messages_tokens = baseline
  → system_tokens   = no_tools  - baseline
  → tools_tokens    = no_system - baseline
  → formatting      = full - (system + messages + tools)  ← API 内部结构标记

这比本地 tokenizer 更准：用的是 API 自己的分词器，数字精确匹配。
代价：3 次额外 API 调用（每次 ~500 token，约 $0.0005）。

运行: . ../env.sh; python3 token_breakdown.py
"""

import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path(__file__).parent.resolve()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

USER_QUERY = "读取当前目录下 README.md 最后10行并发给我"

# ── 工具定义（与 code.py 完全一致）──────────────────────────
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]


def measure(system: str, tools: list, label: str) -> int:
    """发一个请求，返回 input_tokens。"""
    resp = client.messages.create(
        model=MODEL,
        system=system,
        messages=[{"role": "user", "content": USER_QUERY}],
        tools=tools,
        max_tokens=1,
    )
    total_input = resp.usage.input_tokens + resp.usage.cache_read_input_tokens
    print(f"  {label:30s}  input={resp.usage.input_tokens:4d}  cache_read={resp.usage.cache_read_input_tokens:4d}  total={total_input:4d}")
    return total_input


def measure_output_tokens() -> int:
    """量一下 output_tokens（模型生成的 tool_use block）。"""
    resp = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=[{"role": "user", "content": USER_QUERY}],
        tools=TOOLS,
        max_tokens=8000,
    )
    return resp.usage.output_tokens


def main():
    print("=" * 70)
    print("Token Breakdown — 差分 API 调用")
    print("=" * 70)
    print(f"\nModel:   {MODEL}")
    print(f"WorkDir: {WORKDIR}")
    print(f"System:  \"{SYSTEM}\"")
    print(f"Query:   \"{USER_QUERY}\"")
    print(f"Tools:   {len(TOOLS)} 个 ({', '.join(t['name'] for t in TOOLS)})")
    print()

    print("─" * 70)
    print("1. 量 4 个请求（只差一个变量）：\n")

    # 4 个差分请求
    full       = measure(SYSTEM, TOOLS,  "full (system+msg+tools)")
    no_system  = measure("",      TOOLS,  "no_system (msg+tools)")
    no_tools   = measure(SYSTEM, [],      "no_tools (system+msg)")
    baseline   = measure("",      [],      "baseline (msg only)")

    print()
    print("─" * 70)
    print("2. 相减得到各部分：\n")

    messages_tokens = baseline
    system_tokens   = no_tools - baseline
    tools_tokens    = no_system - baseline
    computed_total  = system_tokens + messages_tokens + tools_tokens
    formatting      = full - computed_total

    rows = [
        ("system prompt",    system_tokens),
        ("user message",     messages_tokens),
        ("tools (5个)",      tools_tokens),
        ("API 格式开销",      formatting),
        ("──────────",       None),
        ("input_tokens 合计", full),
    ]

    for label, tokens in rows:
        if tokens is None:
            print(f"  {'─' * 30}")
        else:
            pct = f"{tokens / full * 100:5.1f}%" if full else "  ---"
            print(f"  {label:30s}  {tokens:4d} tokens  ({pct})")

    print()
    print("─" * 70)
    print("3. output_tokens（模型生成的 tool_use block）：\n")

    output_tokens = measure_output_tokens()

    print(f"\n  output_tokens = {output_tokens}")
    print(f"  这是模型生成的 read_file 工具调用的 token 数")
    print(f"  (工具名 + 参数名 + 路径值 + 结构标记)")

    print()
    print("─" * 70)
    print("4. 总结：\n")

    print(f"  input_tokens  = {full:4d}  = system({system_tokens}) + message({messages_tokens}) + tools({tools_tokens}) + fmt({formatting})")
    print(f"  output_tokens = {output_tokens:4d}  = tool_use block (read_file + path 参数 + 路径值)")
    print(f"  cache_read    = 128   = 上一轮缓存的 system prompt")
    print(f"  实际计费 input = input_tokens({full}) + cache_read(128) = {full + 128}")

    print("\n" + "=" * 70)
    print("注意：")
    print("  • 差分法用 API 自己的分词器，数字精确匹配（不像本地 tokenizer 可能有格式差异）")
    print("  • API 格式开销 = API 内部的 role 标记、结构 token 等，不是你的文本")
    print("  • cache_read_input_tokens 是上次调用缓存的 system prompt，本轮复用")
    print("  • 总 3 次额外 API 调用，成本约 $0.0005")
    print("=" * 70)


if __name__ == "__main__":
    main()
