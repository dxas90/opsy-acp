# opsy-acp

An [ACP](https://github.com/i-am-bee/acp) coding agent with automatic local-context injection and MCP server support. Drop it into any ACP-compatible client (Claude Desktop, VS Code, Zed, Cursor) and it picks up your git state, project structure, runtimes, and installed MCP tools automatically.

## Features

- **Auto context detection** — on every session start (and after summarization), a bash script captures git branch/changes, project language, package manager, runtimes, directory tree, and Makefile; the result is injected into the system prompt.
- **Project snapshots** — detected context is written to `~/.opsy/projects/<project>/context.md` so it persists across sessions and is human-readable.
- **Persistent sessions** — conversations are checkpointed to `~/.opsy/agent_sessions.db` (SQLite via LangGraph).
- **MCP tool auto-loading** — reads your existing MCP configs from Claude Code, Claude Desktop, VS Code/Copilot, Zed, and Cursor; no duplication needed.
- **Three interaction modes** — control how much the agent acts autonomously vs. asking for confirmation.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Installation

### From GitHub (recommended)

```sh
uv tool install git+https://github.com/dxas90/opsy-acp
```

After that, `opsy-acp` is available globally:

```sh
opsy-acp
```

### Local development

```sh
git clone https://github.com/dxas90/opsy-acp
cd opsy-acp
uv sync
uv run opsy-acp
```

## Configuration

### API keys

Set at least one provider key in your environment or in a `.env` file at the project root:

```sh
ANTHROPIC_API_KEY=sk-ant-...   # Claude models
OPENAI_API_KEY=sk-...          # GPT models
# Ollama needs no key — it must be running locally on port 11434
```

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_CHECKPOINT_DB` | `~/.opsy/agent_sessions.db` | SQLite checkpoint database path |
| `LOG_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, …) |

### Available models

| Model | Provider | ID |
| --- | --- | --- |
| Claude Sonnet 4.6 | Anthropic | `anthropic:claude-sonnet-4-6` |
| Claude Haiku 4.5 | Anthropic | `anthropic:claude-haiku-4-5` |
| GPT-5 | OpenAI | `openai:gpt-5` |
| Qwen 3.6 27B | Ollama (local) | `ollama:qwen3.6:27b` |

To expose additional models, add entries to `_ANTHROPIC_MODELS`, `_OPENAI_MODELS`, or `_OLLAMA_MODELS` in [src/opsy_acp/server.py](src/opsy_acp/server.py).

### Interaction modes

Selected per session in the ACP client's mode picker:

| Mode | ID | Behaviour |
| --- | --- | --- |
| Ask before edits | `ask_before_edits` | Prompts before every file write, shell command, and plan |
| **Accept edits** *(default)* | `accept_edits` | Auto-accepts file writes; prompts before shell commands and plans |
| Accept everything | `accept_everything` | Fully autonomous — no confirmation prompts |

## MCP server auto-loading

At startup, opsy-acp scans the following config files and loads the MCP servers it finds. The ACP client can also advertise its own servers per session; those are merged in (client wins on name collision).

| Tool | Config file |
| --- | --- |
| Claude Code | `~/.claude/mcp_servers.json` |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%/Claude/claude_desktop_config.json` |
| VS Code / GitHub Copilot (macOS) | `~/Library/Application Support/Code/User/mcp.json` |
| VS Code / GitHub Copilot (Windows) | `%APPDATA%/Code/User/mcp.json` |
| Zed | `~/.zed/settings.json` or `~/.config/zed/settings.json` |
| Cursor | `~/.cursor/mcp.json` |

Supported transports: `stdio`, `sse`, `streamable_http`.

## Data directory

```text
~/.opsy/
├── agent_sessions.db          # LangGraph SQLite checkpoint store
└── projects/
    └── -Users-you-my-project/ # one folder per working directory
        └── context.md         # latest auto-detected environment snapshot
```

`context.md` is overwritten on every detection run. It contains git status, project language, package manager, runtimes, directory listing, tree, and Makefile excerpt — everything the agent needs to orient itself.

## Development

```sh
uv sync --all-extras   # installs dev tools (ruff, pytest)
mise run fmt           # format
mise run lint          # lint
mise run test          # run tests
```

### Project layout

```text
src/opsy_acp/
├── __init__.py    # package entry point
├── __main__.py    # python -m opsy_acp support
└── server.py      # all agent logic
```

### Adding a new model

Edit the appropriate list constant near the top of [src/opsy_acp/server.py](src/opsy_acp/server.py):

```python
_ANTHROPIC_MODELS: list[dict[str, str]] = [
    {"value": "anthropic:claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
    {"value": "anthropic:claude-opus-4-7",   "name": "Claude Opus 4.7"},  # add here
    ...
]
```

### Adding a new interaction mode

Add a `SessionMode` to `_SESSION_MODES` and a matching entry in `_get_interrupt_config()` in [src/opsy_acp/server.py](src/opsy_acp/server.py).

## License

MIT
