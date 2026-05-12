"""ACP coding agent with local context injection and MCP server support.

This module implements a Deep Agents / ACP coding agent that:
- Detects the local development environment (git, project type, runtimes, etc.)
  and injects it into the model's system prompt via ``LocalContextMiddleware``.
- Persists conversation checkpoints to a SQLite database.
- Exposes three interaction modes controlling which tool calls require user
  confirmation before execution.
- Serves the agent over the ACP protocol using ``AgentServerACP``.
- Loads tools from MCP servers sent by the ACP client (Claude Desktop, VS Code,
  Zed, Cursor, etc.) and makes them available to the agent.

MCP server transports supported:
    stdio       Local process, launched by the client (``McpServerStdio``).
    sse         Remote server via Server-Sent Events (``SseMcpServer``).
    http        Remote server via streamable HTTP (``HttpMcpServer``).

Configuration via environment variables:
    AGENT_CHECKPOINT_DB  Path to the SQLite checkpoint database
                         (default: ``~/.opsy/agent_sessions.db``).
    LOG_LEVEL            Python logging level name (default: ``INFO``).

Usage::

    uv run opsy-acp          # start the ACP server (installed package)
    python -m opsy_acp       # start via module invocation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    NotRequired,
    Protocol,
    cast,
    runtime_checkable,
)

from acp import (
    run_agent as run_acp_agent,
)
from acp.schema import (
    HttpMcpServer,
    McpServerStdio,
    SessionMode,
    SessionModeState,
    SseMcpServer,
)
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from deepagents_acp.server import AgentServerACP, AgentSessionContext
from dotenv import load_dotenv
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import ExecuteResponse
    from deepagents.middleware.summarization import SummarizationEvent
    from langgraph.runtime import Runtime


__all__ = [
    "LocalContextMiddleware",
    "LocalContextState",
    "MCPAwareAgentServer",
    "MCPSessionContext",
    "build_detect_script",
    "main",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model and session-mode constants
# ---------------------------------------------------------------------------

_ANTHROPIC_MODELS: list[dict[str, str]] = [
    {"value": "anthropic:claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
    {"value": "anthropic:claude-haiku-4-5", "name": "Claude Haiku 4.5"},
]

_OPENAI_MODELS: list[dict[str, str]] = [
    {"value": "openai:gpt-5", "name": "GPT-5"},
]

_OLLAMA_MODELS: list[dict[str, str]] = [
    {"value": "ollama:qwen3.6:27b", "name": "Qwen3.6"},
]

# Combined list served to ACP clients; extend either provider list above to
# make new models available without touching serve logic.
_AVAILABLE_MODELS: list[dict[str, str]] = _ANTHROPIC_MODELS + _OPENAI_MODELS + _OLLAMA_MODELS

_DEFAULT_MODE_ID: str = "accept_edits"  # mode used for new sessions before the user changes it

# Ordered list of modes exposed in the ACP session config selector.
_SESSION_MODES: list[SessionMode] = [
    SessionMode(
        id="ask_before_edits",
        name="Ask before edits",
        description="Ask permission before edits, writes, shell commands, and plans",
    ),
    SessionMode(
        id="accept_edits",
        name="Accept edits",
        description="Auto-accept edit operations, but ask before shell commands and plans",
    ),
    SessionMode(
        id="accept_everything",
        name="Accept everything",
        description="Auto-accept all operations without asking permission",
    ),
]


# ---------------------------------------------------------------------------
# Internal protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class _ExecutableBackend(Protocol):
    """Structural protocol for backends that support synchronous shell execution.

    Used only for type-checking ``LocalContextMiddleware``; any backend
    that provides an ``execute(command) -> ExecuteResponse`` method satisfies
    this protocol without explicit registration.
    """

    def execute(self, command: str) -> ExecuteResponse: ...


# ---------------------------------------------------------------------------
# Context detection script
#
# Outputs markdown describing the current working environment. Each section
# is guarded so that missing tools or unsupported environments are silently
# skipped -- external tools like git, tree, python3, and node are checked
# with `command -v` before use.
#
# The script is built from section functions so each piece can be tested
# independently. Independent sections run as parallel background subshells;
# see build_detect_script() for the orchestration logic.
# ---------------------------------------------------------------------------


def _section_header() -> str:
    """CWD line and IN_GIT flag (used by other sections).

    Returns:
        Bash snippet that prints the header and sets `CWD` / `IN_GIT`.
    """
    return r"""CWD="$(pwd)"
echo "## Local Context"
echo ""
echo "**Current Directory**: \`${CWD}\`"
echo ""

# --- Check git once ---
IN_GIT=false
if command -v git >/dev/null 2>&1 \
    && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=true
fi"""


def _section_project() -> str:
    """Language, monorepo, git root, virtual-env detection.

    Returns:
        Bash snippet (requires `CWD` / `IN_GIT` from header).
    """
    return r"""# --- Project ---
PROJ_LANG=""
[ -f pyproject.toml ] || [ -f setup.py ] && PROJ_LANG="python"
[ -z "$PROJ_LANG" ] && [ -f package.json ] && PROJ_LANG="javascript/typescript"
[ -z "$PROJ_LANG" ] && [ -f Cargo.toml ] && PROJ_LANG="rust"
[ -z "$PROJ_LANG" ] && [ -f go.mod ] && PROJ_LANG="go"
[ -z "$PROJ_LANG" ] && { [ -f pom.xml ] || [ -f build.gradle ]; } && PROJ_LANG="java"

MONOREPO=false
{ [ -f lerna.json ] || [ -f pnpm-workspace.yaml ] \
  || [ -d packages ] || { [ -d libs ] && [ -d apps ]; } \
  || [ -d workspaces ]; } && MONOREPO=true

ROOT=""
$IN_GIT && ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

ENVS=""
{ [ -d .venv ] || [ -d venv ]; } && ENVS=".venv"
[ -d node_modules ] && ENVS="${ENVS:+${ENVS}, }node_modules"

HAS_PROJECT=false
{ [ -n "$PROJ_LANG" ] || { [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ]; } \
  || $MONOREPO || [ -n "$ENVS" ]; } && HAS_PROJECT=true

if $HAS_PROJECT; then
  echo "**Project**:"
  [ -n "$PROJ_LANG" ] && echo "- Language: ${PROJ_LANG}"
  [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && echo "- Project root: \`${ROOT}\`"
  $MONOREPO && echo "- Monorepo: yes"
  [ -n "$ENVS" ] && echo "- Environments: ${ENVS}"
  echo ""
fi"""


def _section_package_managers() -> str:
    """Python and Node package manager detection.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Package managers ---
PKG=""
if [ -f uv.lock ]; then PKG="Python: uv"
elif [ -f poetry.lock ]; then PKG="Python: poetry"
elif [ -f Pipfile.lock ] || [ -f Pipfile ]; then PKG="Python: pipenv"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.uv\]' pyproject.toml 2>/dev/null; then PKG="Python: uv"
  elif grep -q '\[tool\.poetry\]' pyproject.toml 2>/dev/null; then PKG="Python: poetry"
  else PKG="Python: pip"
  fi
elif [ -f requirements.txt ]; then PKG="Python: pip"
fi

NODE_PKG=""
if [ -f bun.lockb ] || [ -f bun.lock ]; then NODE_PKG="Node: bun"
elif [ -f pnpm-lock.yaml ]; then NODE_PKG="Node: pnpm"
elif [ -f yarn.lock ]; then NODE_PKG="Node: yarn"
elif [ -f package-lock.json ] || [ -f package.json ]; then NODE_PKG="Node: npm"
fi
[ -n "$NODE_PKG" ] && PKG="${PKG:+${PKG}, }${NODE_PKG}"
[ -n "$PKG" ] && echo "**Package Manager**: ${PKG}" && echo ""
"""


def _section_runtimes() -> str:
    """Python and Node runtime version detection.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Runtimes ---
RT=""
if command -v python3 >/dev/null 2>&1; then
  PV="$(python3 --version 2>/dev/null | awk '{print $2}')"
  [ -n "$PV" ] && RT="Python ${PV}"
fi
if command -v node >/dev/null 2>&1; then
  NV="$(node --version 2>/dev/null | sed 's/^v//')"
  [ -n "$NV" ] && RT="${RT:+${RT}, }Node ${NV}"
fi
[ -n "$RT" ] && echo "**Runtimes**: ${RT}" && echo ""
"""


def _section_git() -> str:
    """Git branch, main branches, uncommitted changes.

    Returns:
        Bash snippet (requires `IN_GIT` from header).
    """
    return r"""# --- Git ---
if $IN_GIT; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  GT="**Git**: Current branch \`${BRANCH}\`"

  MAINS=""
  for b in $(git branch 2>/dev/null | sed 's/^[* ]*//'); do
    case "$b" in
      main) MAINS="${MAINS:+${MAINS}, }\`main\`" ;;
      master) MAINS="${MAINS:+${MAINS}, }\`master\`" ;;
    esac
  done
  [ -n "$MAINS" ] && GT="${GT}, main branch available: ${MAINS}"

  DC=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  if [ "$DC" -gt 0 ]; then
    if [ "$DC" -eq 1 ]; then GT="${GT}, 1 uncommitted change"
    else GT="${GT}, ${DC} uncommitted changes"
    fi
  fi

  echo "$GT"
  echo ""
fi"""


def _section_test_command() -> str:
    """Test command detection (make test / pytest / npm test).

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Test command ---
TC=""
if [ -f Makefile ] && grep -qE '^tests?:' Makefile 2>/dev/null; then TC="make test"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.pytest' pyproject.toml 2>/dev/null \
      || [ -f pytest.ini ] || [ -d tests ] || [ -d test ]; then
    TC="pytest"
  fi
elif [ -f package.json ] \
    && grep -q '"test"' package.json 2>/dev/null; then
  TC="npm test"
fi
[ -n "$TC" ] && echo "**Run Tests**: \`${TC}\`" && echo ""
"""


def _section_files() -> str:
    """Directory listing (filtered, capped at 20).

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Files ---
EXCL='node_modules|__pycache__|\.pytest_cache'
EXCL="${EXCL}|\.mypy_cache|\.ruff_cache|\.tox"
EXCL="${EXCL}|\.coverage|\.eggs|dist|build"
FILES=$(
  { ls -1 2>/dev/null; [ -e .deepagents ] && echo .deepagents; } |
  grep -vE "^(${EXCL})$" |
  sort -u
)
if [ -n "$FILES" ]; then
  TOTAL=$(echo "$FILES" | wc -l | tr -d ' ')
  SHOWN_FILES=$(echo "$FILES" | head -20)
  SHOWN=$(echo "$SHOWN_FILES" | wc -l | tr -d ' ')
  echo "**Files** (${SHOWN} shown):"
  echo "$SHOWN_FILES" | while IFS= read -r f; do
    if [ -d "$f" ]; then echo "- ${f}/"
    else echo "- ${f}"
    fi
  done
  [ "$SHOWN" -lt "$TOTAL" ] && echo "... ($((TOTAL - SHOWN)) more files)"
  echo ""
fi"""


def _section_tree() -> str:
    """`tree -L 3` output.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Tree ---
if command -v tree >/dev/null 2>&1; then
  TREE_EXCL='node_modules|.venv|__pycache__|.pytest_cache'
  TREE_EXCL="${TREE_EXCL}|.git|.mypy_cache|.ruff_cache"
  TREE_EXCL="${TREE_EXCL}|.tox|.coverage|.eggs|dist|build"
  T=$(tree -L 3 --noreport --dirsfirst \
    -I "$TREE_EXCL" 2>/dev/null | head -22)
  if [ -n "$T" ]; then
    echo "**Tree** (3 levels):"
    echo '```text'
    echo "$T"
    echo '```'
    echo ""
  fi
fi"""


def _section_makefile() -> str:
    """First 20 lines of Makefile (falls back to git root in monorepos).

    Returns:
        Bash snippet (requires `ROOT` from `_section_project` and `CWD` from header).
    """
    return r"""# --- Makefile ---
MK=""
if [ -f Makefile ]; then
  MK="Makefile"
elif [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && [ -f "${ROOT}/Makefile" ]; then
  MK="${ROOT}/Makefile"
fi
if [ -n "$MK" ]; then
  echo "**Makefile** (\`${MK}\`, first 20 lines):"
  echo '```makefile'
  head -20 "$MK"
  TL=$(wc -l < "$MK" | tr -d ' ')
  [ "$TL" -gt 20 ] && echo "... (truncated)"
  echo '```'
fi"""


def build_detect_script() -> str:
    """Concatenate all section functions into the full detection script.

    Independent sections run as parallel background jobs writing to temp
    files, then results are concatenated in the original display order.
    The header (CWD / IN_GIT) and project section (sets ROOT) run first
    because later sections depend on their variables.

    Returns:
        Complete bash heredoc ready for `backend.execute()`.
    """
    # Header + project run synchronously (set CWD, IN_GIT, ROOT for others)
    serial_prefix = f"{_section_header()}\n{_section_project()}"

    # These sections are independent — run them in parallel.
    # Subshells inherit parent variables (IN_GIT, ROOT, CWD) via fork.
    # Individual exit codes are not tracked because sections legitimately
    # exit non-zero when they have nothing to report (e.g. no runtimes).
    parallel_sections = [
        ("02_pkgmgr", _section_package_managers()),
        ("03_runtimes", _section_runtimes()),
        ("04_git", _section_git()),
        ("05_testcmd", _section_test_command()),
        ("06_files", _section_files()),
        ("07_tree", _section_tree()),
        ("08_makefile", _section_makefile()),
    ]

    # Build parallel wrapper: each section runs in a subshell writing to a
    # temp file. Stderr is captured per-section to prevent noise leakage.
    parallel_setup = "_DCT=$(mktemp -d) || exit 1\ntrap 'rm -rf \"$_DCT\"' EXIT"
    parallel_block = "\n".join(
        f'(\n{body}\n) > "$_DCT/{name}" 2>"$_DCT/{name}.err" &' for name, body in parallel_sections
    )
    cat_line = "cat " + " ".join(f'"$_DCT/{name}"' for name, _ in parallel_sections)

    body = f"{serial_prefix}\n{parallel_setup}\n{parallel_block}\nwait\n{cat_line}"
    return f"bash <<'__DETECT_CONTEXT_EOF__'\n{body}\n__DETECT_CONTEXT_EOF__\n"


# Computed once at import time so all agent instances share the same script
# string without re-building it on every request.
DETECT_CONTEXT_SCRIPT = build_detect_script()


def _project_dir_for_cwd(cwd: str) -> Path:
    """Return ``~/.opsy/projects/<encoded-cwd>/`` for a working directory.

    Uses the same path-encoding convention as Claude Code: each ``/`` (or
    ``\\`` on Windows) separator is replaced with ``-``, producing a flat
    directory name that is unique per absolute path.

    Example: ``/Users/alice/my-app`` → ``~/.opsy/projects/-Users-alice-my-app/``
    """
    encoded = cwd.replace("\\", "/").replace("/", "-")
    return Path.home() / ".opsy" / "projects" / encoded


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class LocalContextState(AgentState):
    """State for local context middleware."""

    local_context: NotRequired[str]
    """Formatted local context: cwd, project, package managers,
    runtimes, git, test command, files, tree, Makefile.
    """

    _local_context_refreshed_at_cutoff: NotRequired[Annotated[int, PrivateStateAttr]]
    """Cutoff index of the summarization event we last refreshed for.

    Stored in LangGraph checkpointed state (isolated per thread) and private
    (not exposed to subagents via `PrivateStateAttr`). Used to avoid redundant
    re-runs of the detection script for the same summarization event.
    """


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class LocalContextMiddleware(AgentMiddleware):
    """Inject local context (git state, project structure, etc.) into the system prompt.

    Runs a bash detection script via `backend.execute()` on first interaction
    and again after each summarization event, stores the result in state, and
    appends it to the system prompt on every model call.

    Because the script runs inside the backend, it works for both local shells
    and remote sandboxes.
    """

    state_schema = LocalContextState

    def __init__(
        self,
        backend: _ExecutableBackend,
        project_dir: Path | None = None,
    ) -> None:
        """Initialize with a backend that supports shell execution.

        Args:
            backend: Backend instance that provides shell command execution.
            project_dir: Optional ``~/.opsy/projects/<encoded>/`` directory.
                When provided, each successful context detection is written to
                ``project_dir/context.md`` so the snapshot persists across
                sessions and is human-readable.
        """
        self.backend = backend
        self._project_dir = project_dir
        if project_dir:
            project_dir.mkdir(parents=True, exist_ok=True)

    def _save_context(self, output: str) -> None:
        """Persist detected context to ``<project_dir>/context.md``."""
        if not self._project_dir:
            return
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            content = f"<!-- opsy: last refreshed {ts} -->\n\n{output}\n"
            (self._project_dir / "context.md").write_text(content, encoding="utf-8")
            logger.debug("Saved local context snapshot to %s", self._project_dir)
        except Exception:
            logger.warning("Failed to save context snapshot to %s", self._project_dir, exc_info=True)

    def _run_detect_script(self) -> str | None:
        """Run the environment detection script.

        Returns:
            Stripped script output, or `None` on failure/empty output.
        """
        try:
            result = self.backend.execute(DETECT_CONTEXT_SCRIPT)
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(self.backend).__name__,
                exc_info=True,
            )
            return None

        output = result.output.strip() if result.output else ""
        if result.exit_code is None or result.exit_code != 0:
            logger.warning(
                "Local context detection script %s; context will be omitted. Output: %.200s",
                f"exited with code {result.exit_code}"
                if result.exit_code is not None
                else "did not report an exit code",
                output or "(empty)",
            )
            return None
        if not output:
            logger.debug("Local context detection script succeeded but produced no output")
            return None
        self._save_context(output)
        return output

    # override - state parameter is intentionally narrowed from
    # AgentState to LocalContextState for type safety within this middleware.
    def before_agent(  # type: ignore[override]
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002  # Required by interface but not used in local context
    ) -> dict[str, Any] | None:
        """Run context detection on first interaction and refresh after summarization.

        On the first invocation, runs the detection script and stores the result.
        After a summarization event (indicated by a new `_summarization_event`
        in state), re-runs the script to capture any environment changes that
        occurred during the session.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with `local_context` populated on success. On a
                post-summarization refresh failure, returns a state update
                recording the cutoff (without `local_context`) to prevent
                retry loops.

                Returns `None` if context is already set and no refresh is
                needed, or if initial detection fails.
        """
        # --- Post-summarization refresh ---
        # _summarization_event is a private field from SummarizationState.
        # At runtime the merged state dict contains all middleware fields;
        # accessed as untyped dict value because LocalContextState does not
        # (and should not) redeclare it.
        raw_event = state.get("_summarization_event")
        if raw_event is not None:
            event: SummarizationEvent = raw_event
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = self._run_detect_script()
                if output:
                    return {
                        "local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                # Script failed — record cutoff to avoid retry loop,
                # keep existing local_context.
                return {"_local_context_refreshed_at_cutoff": cutoff}

        # --- Initial detection (first invocation) ---
        if state.get("local_context"):
            return None

        output = self._run_detect_script()
        if output:
            return {"local_context": output}
        return None

    def _get_modified_request(self, request: ModelRequest) -> ModelRequest | None:
        """Append local context to the system prompt if available.

        Args:
            request: The model request to potentially modify.

        Returns:
            Modified request with context appended, or `None`.
        """
        state = cast("LocalContextState", request.state)
        local_context = state.get("local_context", "")

        if not local_context:
            return None

        system_prompt = request.system_prompt or ""
        new_prompt = system_prompt + "\n\n" + local_context
        return request.override(system_message=SystemMessage(content=new_prompt))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject local context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return handler(modified_request or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject local context into system prompt (async).

        Args:
            request: The model request being processed.
            handler: The async handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return await handler(modified_request or request)


# ---------------------------------------------------------------------------
# Interrupt configuration
# ---------------------------------------------------------------------------


def _get_interrupt_config(mode_id: str) -> dict[str, bool | InterruptOnConfig]:
    """Return the interrupt configuration for a given session mode.

    The returned dict maps tool names to their interrupt settings. Keys present
    in the dict cause the agent to pause and ask the user before executing that
    tool. Keys absent from the dict are auto-accepted without a prompt.

    ``allowed_decisions`` lists the choices the user will be offered; for
    tool-call interrupts this is always ``["approve", "reject"]``.

    Args:
        mode_id: One of the IDs defined in ``_SESSION_MODES``.

    Returns:
        Interrupt configuration dict suitable for ``create_deep_agent``'s
        ``interrupt_on`` parameter. Returns an empty dict for unknown mode IDs
        (equivalent to ``"accept_everything"``).
    """
    # Maps mode ID -> tools that require user confirmation before execution.
    # Tools NOT listed are auto-accepted for that mode.
    # cast() is needed because pyright cannot infer TypedDict literals from
    # nested dict literals — the runtime structure is correct.
    _confirm: InterruptOnConfig = cast(InterruptOnConfig, {"allowed_decisions": ["approve", "reject"]})
    mode_to_interrupt: dict[str, dict[str, bool | InterruptOnConfig]] = {
        # Prompt for every write and shell operation.
        "ask_before_edits": {
            "edit_file": _confirm,
            "write_file": _confirm,
            "write_todos": _confirm,
            "execute": _confirm,
        },
        # Auto-accept file edits/writes; only prompt for shell and plans.
        "accept_edits": {
            "write_todos": _confirm,
            "execute": _confirm,
        },
        # Auto-accept everything — no interrupts.
        "accept_everything": {},
    }
    return mode_to_interrupt.get(mode_id, {})


# ---------------------------------------------------------------------------
# MCP server integration
# ---------------------------------------------------------------------------

#: Type alias for the union of all ACP MCP server descriptor types.
MCPServerDescriptor = HttpMcpServer | SseMcpServer | McpServerStdio


def _acp_servers_to_mcp_connections(
    servers: list[MCPServerDescriptor],
) -> dict[str, Any]:
    """Convert ACP MCP server descriptors to ``MultiServerMCPClient`` connection dicts.

    Each ACP server type maps to a ``langchain-mcp-adapters`` transport:

    ============== =========================================
    ACP type       ``transport`` value
    ============== =========================================
    ``McpServerStdio``    ``"stdio"``
    ``SseMcpServer``      ``"sse"``
    ``HttpMcpServer``     ``"streamable_http"``
    ============== =========================================

    Args:
        servers: List of ACP MCP server descriptors received from the client.

    Returns:
        Dict keyed by server name suitable for passing to
        ``MultiServerMCPClient(connections=...)``.
    """
    connections: dict[str, Any] = {}
    for server in servers:
        if isinstance(server, McpServerStdio):
            connections[server.name] = {
                "transport": "stdio",
                "command": server.command,
                "args": server.args,
                "env": {e.name: e.value for e in server.env} if server.env else None,
            }
        elif isinstance(server, SseMcpServer):
            headers = {h.name: h.value for h in server.headers} if server.headers else None
            connections[server.name] = {
                "transport": "sse",
                "url": server.url,
                **({"headers": headers} if headers else {}),
            }
        elif isinstance(server, HttpMcpServer):
            headers = {h.name: h.value for h in server.headers} if server.headers else None
            connections[server.name] = {
                "transport": "streamable_http",
                "url": server.url,
                **({"headers": headers} if headers else {}),
            }
    return connections


def _load_jsonc(path: Path) -> Any:
    """Parse a JSON-with-comments file (JSONC), stripping ``//`` line comments.

    VS Code's ``mcp.json`` uses JSONC format.  Standard ``json.loads`` cannot
    handle it, so we strip single-line comments before parsing.

    Args:
        path: Path to the JSONC file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file cannot be parsed after comment removal.
    """
    text = path.read_text(encoding="utf-8-sig")  # utf-8-sig handles optional BOM
    # Strip // line comments that are NOT inside quoted strings.
    # The pattern skips over "..." and '...' before matching // to avoid
    # clobbering URLs (e.g. "https://...") inside string values.
    stripped = re.sub(
        r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|(//[^\n]*)',
        lambda m: "" if m.group(1) is not None else m.group(0),
        text,
    )
    # Remove trailing commas before ] or } (common in JSONC after commented blocks).
    stripped = re.sub(r",\s*([}\]])", r"\1", stripped)
    # strict=False allows literal tab/newline characters inside strings
    # (some editors write config files with embedded whitespace control chars).
    return json.loads(stripped, strict=False)


def _expand_env_vars(value: str) -> str:
    """Expand ``${env:VAR}`` and ``$VAR`` / ``${VAR}`` placeholders in strings.

    VS Code uses the ``${env:VAR}`` syntax in ``mcp.json``; standard shells use
    ``$VAR`` or ``${VAR}``.  Both are resolved from the current environment.

    Args:
        value: String that may contain environment variable references.

    Returns:
        String with all recognised placeholders expanded.  Unknown variables are
        left as-is (same behaviour as ``os.path.expandvars``).
    """
    # ${env:VAR} — VS Code convention
    value = re.sub(
        r"\$\{env:([^}]+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )
    # ${VAR} and $VAR — shell convention
    return os.path.expandvars(value)


def _load_global_mcp_connections() -> dict[str, Any]:
    """Discover and load MCP server configurations from known tool config files.

    Reads MCP server definitions from the following locations in order.
    Duplicate server names are resolved by last-writer-wins (later sources
    in the list override earlier ones):

    1. **Claude Code** — ``~/.claude/mcp_servers.json``
       Format: ``{"mcpServers": {"name": {"type": ..., "url"/"command": ...}}}``

    2. **Claude Desktop** — ``~/Library/Application Support/Claude/claude_desktop_config.json``
       (macOS) or ``%APPDATA%/Claude/claude_desktop_config.json`` (Windows)
       Format: ``{"mcpServers": {"name": {"command": ..., "args": [...], "env": {...}}}}``

    3. **VS Code / GitHub Copilot** — ``~/Library/Application Support/Code/User/mcp.json``
       (macOS) or ``%APPDATA%/Code/User/mcp.json`` (Windows).  Uses JSONC format.
       Format: ``{"servers": {"name": {"type": ..., "url"/"command": ...}}}``

    4. **Zed** — ``~/.zed/settings.json``
       Format: ``{"context_servers": {"name": {"command": {"path": ..., "args": [...]}}}}``

    5. **Cursor** — ``~/.cursor/mcp.json``
       Format: same as Claude Desktop (``mcpServers`` key).

    Environment variable references (``${env:VAR}``, ``${VAR}``, ``$VAR``) in
    URLs, headers, and command paths are expanded against the current environment.

    Servers that cannot be parsed are skipped with a warning.

    Returns:
        Connection dict keyed by server name, ready for ``MultiServerMCPClient``.
    """
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", "")) if os.name == "nt" else None

    # (source_label, path, parser_function)
    # Each parser receives the parsed JSON object and returns dict[str, Any] connections.
    sources: list[tuple[str, Path, Any]] = [
        (
            "Claude Code",
            home / ".claude" / "mcp_servers.json",
            _parse_claude_desktop_format,  # same mcpServers key
        ),
        (
            "Claude Desktop (macOS)",
            home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
            _parse_claude_desktop_format,
        ),
        (
            "VS Code",
            home / "Library" / "Application Support" / "Code" / "User" / "mcp.json",
            _parse_vscode_format,
        ),
        (
            "Zed",
            home / ".zed" / "settings.json",
            _parse_zed_format,
        ),
        (
            "Zed (XDG)",
            home / ".config" / "zed" / "settings.json",
            _parse_zed_format,
        ),
        (
            "Cursor",
            home / ".cursor" / "mcp.json",
            _parse_claude_desktop_format,  # Cursor uses the same format
        ),
    ]

    # Windows paths for Claude Desktop and VS Code
    if appdata:
        sources += [
            (
                "Claude Desktop (Windows)",
                appdata / "Claude" / "claude_desktop_config.json",
                _parse_claude_desktop_format,
            ),
            (
                "VS Code (Windows)",
                appdata / "Code" / "User" / "mcp.json",
                _parse_vscode_format,
            ),
        ]

    merged: dict[str, Any] = {}
    for label, path, parser in sources:
        if not path.exists():
            continue
        try:
            raw = _load_jsonc(path)
            connections = parser(raw)
            if connections:
                logger.debug("Loaded %d MCP server(s) from %s (%s)", len(connections), label, path)
                merged.update(connections)
        except Exception:
            logger.warning("Failed to read MCP config from %s (%s)", label, path, exc_info=True)

    return merged


def _parse_server_entry(name: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single raw server config entry to a ``MultiServerMCPClient`` connection.

    Handles the three transport types (stdio, sse, streamable_http / http) and
    expands environment variable references in all string values.

    Args:
        name: Server name (used only for error logging).
        entry: Raw server config dict from the config file.

    Returns:
        Connection dict for ``MultiServerMCPClient``, or ``None`` if the entry
        cannot be interpreted.
    """
    server_type = entry.get("type", "")
    command = entry.get("command", "")

    if server_type == "stdio" or (not server_type and command):
        # stdio transport: launched as a local child process
        args = [_expand_env_vars(a) for a in entry.get("args", [])]
        raw_env = entry.get("env") or {}
        env = {k: _expand_env_vars(str(v)) for k, v in raw_env.items()} if raw_env else None
        return {
            "transport": "stdio",
            "command": _expand_env_vars(command),
            "args": args,
            **({"env": env} if env else {}),
        }

    url = _expand_env_vars(entry.get("url", ""))
    if not url:
        logger.warning("MCP server %r has no url and no command — skipping", name)
        return None

    raw_headers = entry.get("headers") or {}
    headers = (
        {k: _expand_env_vars(str(v)) for k, v in raw_headers.items()} if raw_headers else None
    )

    if server_type in ("sse",):
        return {
            "transport": "sse",
            "url": url,
            **({"headers": headers} if headers else {}),
        }

    # http / streamable_http / anything else with a URL
    return {
        "transport": "streamable_http",
        "url": url,
        **({"headers": headers} if headers else {}),
    }


def _parse_claude_desktop_format(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse ``mcpServers`` dict used by Claude Desktop, Claude Code, and Cursor.

    Args:
        raw: Parsed JSON from the config file.

    Returns:
        Connection dict keyed by server name.
    """
    servers = raw.get("mcpServers", {})
    connections: dict[str, Any] = {}
    for name, entry in servers.items():
        conn = _parse_server_entry(name, entry)
        if conn is not None:
            connections[name] = conn
    return connections


def _parse_vscode_format(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse ``servers`` dict used by VS Code / GitHub Copilot ``mcp.json``.

    Args:
        raw: Parsed JSONC from the VS Code config file.

    Returns:
        Connection dict keyed by server name.
    """
    servers = raw.get("servers", {})
    connections: dict[str, Any] = {}
    for name, entry in servers.items():
        conn = _parse_server_entry(name, entry)
        if conn is not None:
            connections[name] = conn
    return connections


def _parse_zed_format(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse ``context_servers`` dict used by Zed editor.

    Zed supports two entry shapes:

    - HTTP/SSE: ``{"url": "https://...", "enabled": true}``
    - stdio: ``{"command": {"path": "...", "args": [...]}, "enabled": true}``

    Disabled entries (``"enabled": false``) are skipped.

    Args:
        raw: Parsed JSON from Zed's ``settings.json``.

    Returns:
        Connection dict keyed by server name.
    """
    context_servers = raw.get("context_servers", {})
    connections: dict[str, Any] = {}
    for name, entry in context_servers.items():
        if not entry.get("enabled", True):
            continue

        url = _expand_env_vars(entry.get("url", ""))
        if url:
            # HTTP or SSE entry — treat as streamable_http (Zed default)
            connections[name] = {"transport": "streamable_http", "url": url}
            continue

        cmd_block = entry.get("command", {})
        command = cmd_block.get("path", cmd_block.get("command", ""))
        if not command:
            continue
        args = [_expand_env_vars(a) for a in cmd_block.get("args", [])]
        connections[name] = {
            "transport": "stdio",
            "command": _expand_env_vars(command),
            "args": args,
        }
    return connections


# ---------------------------------------------------------------------------
# Extended session context and server
# ---------------------------------------------------------------------------


class MCPSessionContext(AgentSessionContext):
    """Extended session context that carries pre-loaded MCP tools for the session.

    ``AgentSessionContext`` is a frozen dataclass. We use ``object.__setattr__``
    to set additional fields without triggering the frozen-dataclass guard.
    Tools are loaded async before the sync agent factory is called, so the
    factory itself can remain synchronous.
    """

    mcp_tools: list[Any]

    def __init__(
        self,
        cwd: str,
        mode: str,
        model: str | None = None,
        mcp_tools: list[Any] | None = None,
    ) -> None:
        """Create a session context with pre-loaded MCP tools.

        Args:
            cwd: Working directory for the session.
            mode: Interaction mode ID (e.g. ``"accept_edits"``).
            model: Optional model identifier override.
            mcp_tools: Tools already fetched from MCP servers for this session.
        """
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "mcp_tools", mcp_tools or [])


class MCPAwareAgentServer(AgentServerACP):
    """``AgentServerACP`` subclass that loads MCP tools and injects them into the agent factory.

    MCP tool loading is async (network calls to servers), but the base class
    ``_reset_agent`` is synchronous.  This subclass resolves the mismatch by
    loading tools eagerly in ``new_session`` (which is async) and caching them
    per session, so ``_reset_agent`` can pass already-fetched tools to the
    sync factory via ``MCPSessionContext``.
    """

    def __init__(self, global_mcp_connections: dict[str, Any], **kwargs: Any) -> None:
        """Initialise with the pre-discovered global MCP connection map.

        Args:
            global_mcp_connections: Connections loaded from local tool configs
                at startup (Claude Code, VS Code, Zed, Cursor, etc.).
            **kwargs: Forwarded to ``AgentServerACP.__init__``.
        """
        super().__init__(**kwargs)
        self._global_mcp_connections = global_mcp_connections
        # Maps session_id -> tools already fetched for that session.
        self._session_mcp_tools: dict[str, list[Any]] = {}

    async def _load_mcp_tools(
        self,
        session_id: str,
        session_servers: list[MCPServerDescriptor] | None,
    ) -> list[Any]:
        """Fetch tools from all MCP servers for a session.

        Merges global connections with any session-specific ones advertised by
        the ACP client (client wins on name collision), then probes each server
        individually so that an unreachable server does not prevent tools from
        reachable servers being loaded.

        Args:
            session_id: Used only for log messages.
            session_servers: Optional server list from the ACP client.

        Returns:
            List of LangChain tools collected from all reachable servers.
        """
        connections: dict[str, Any] = dict(self._global_mcp_connections)
        if session_servers:
            connections.update(_acp_servers_to_mcp_connections(session_servers))
            logger.info(
                "Session %s: %d global + %d client MCP server(s)",
                session_id,
                len(self._global_mcp_connections),
                len(session_servers),
            )

        if not connections:
            return []

        # Load each server independently — one unreachable server must not
        # prevent tools from the others from being available.
        all_tools: list[Any] = []
        for name, conn in connections.items():
            try:
                tools = await MultiServerMCPClient({name: conn}).get_tools()
                logger.debug("Session %s: %s → %d tool(s)", session_id, name, len(tools))
                all_tools.extend(tools)
            except Exception:
                logger.warning(
                    "Session %s: failed to load tools from MCP server %r — skipping",
                    session_id,
                    name,
                    exc_info=False,  # keep log noise low; debug for full traceback
                )

        logger.info(
            "Session %s: loaded %d MCP tool(s) from %d/%d server(s): %s",
            session_id,
            len(all_tools),
            sum(1 for n in connections if any(t.name for t in all_tools)),
            len(connections),
            [t.name for t in all_tools],
        )
        return all_tools

    async def new_session(  # type: ignore[override]
        self,
        cwd: str,
        mcp_servers: list[MCPServerDescriptor] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a session, load MCP tools, and cache them for the sync factory."""
        response = await super().new_session(cwd=cwd, mcp_servers=mcp_servers, **kwargs)
        session_id = response.session_id
        tools = await self._load_mcp_tools(session_id, mcp_servers)
        self._session_mcp_tools[session_id] = tools
        return response

    def _reset_agent(self, session_id: str) -> None:  # type: ignore[override]
        """Re-create the agent synchronously using pre-loaded MCP tools."""
        cwd = self._session_cwds.get(session_id, "")
        if cwd:
            self._cwd = cwd
        if isinstance(self._agent_factory, CompiledStateGraph):
            self._agent = self._agent_factory
            return

        mode = self._session_modes.get(
            session_id,
            self._modes.current_mode_id if self._modes is not None else "auto",
        )
        model = self._session_models.get(session_id) if self._models is not None else None
        mcp_tools = self._session_mcp_tools.get(session_id, [])

        context = MCPSessionContext(
            cwd=self._cwd,
            mode=mode,
            model=model,
            mcp_tools=mcp_tools,
        )
        self._agent = self._agent_factory(context)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Agent server
# ---------------------------------------------------------------------------


async def _serve_agent() -> None:
    """Start the ACP agent server with MCP support.

    Loads ``.env``, opens the SQLite checkpoint database, and runs the ACP
    server loop until the process is terminated.

    MCP tool loading order (last wins on name collision):

    1. Global connections discovered from local tool configs (Claude Code,
       Claude Desktop, VS Code/Copilot, Zed, Cursor).
    2. Per-session connections advertised by the ACP client in ``new_session``.
    """
    load_dotenv()

    _default_db = Path.home() / ".opsy" / "agent_sessions.db"
    db_path = os.environ.get("AGENT_CHECKPOINT_DB", str(_default_db))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting agent server (db=%s, default_mode=%s)", db_path, _DEFAULT_MODE_ID)

    # Discover global MCP servers from local tool configs once at startup.
    global_mcp_connections = _load_global_mcp_connections()
    if global_mcp_connections:
        logger.info(
            "Discovered %d global MCP server(s): %s",
            len(global_mcp_connections),
            list(global_mcp_connections),
        )
    else:
        logger.debug("No global MCP servers found in local tool configs")

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

        def build_agent(context: AgentSessionContext) -> CompiledStateGraph:
            """Sync agent factory called by the ACP server on each session reset.

            MCP tools are pre-loaded async in ``MCPAwareAgentServer.new_session``
            and passed in via ``MCPSessionContext.mcp_tools``, so this function
            can stay synchronous as required by the base class.
            """
            _root_dir = context.cwd
            interrupt_config = _get_interrupt_config(context.mode)

            ephemeral_backend = StateBackend()
            shell_env = os.environ.copy()

            # LocalShellBackend provides filesystem access and shell execution
            # with per-command timeout support via the `execute` tool.
            shell_backend = LocalShellBackend(
                root_dir=_root_dir,
                inherit_env=True,
                env=shell_env,
            )
            backend = CompositeBackend(
                default=shell_backend,
                routes={
                    "/memories/": ephemeral_backend,
                    "/conversation_history/": ephemeral_backend,
                },
            )

            mcp_tools: list[Any] = (
                context.mcp_tools  # type: ignore[attr-defined]
                if isinstance(context, MCPSessionContext)
                else []
            )

            return create_deep_agent(
                # Falls back to Deep Agent default model if not provided.
                model=context.model,
                tools=mcp_tools or None,
                checkpointer=checkpointer,
                backend=backend,
                interrupt_on=interrupt_config,
                middleware=[LocalContextMiddleware(backend=backend, project_dir=_project_dir_for_cwd(_root_dir))],
            )

        modes = SessionModeState(
            current_mode_id=_DEFAULT_MODE_ID,
            available_modes=_SESSION_MODES,
        )

        acp_agent = MCPAwareAgentServer(  # type: ignore[abstract]  # satisfies abstract methods via ACP metaclass
            global_mcp_connections=global_mcp_connections,
            agent=build_agent,
            modes=modes,
            models=_AVAILABLE_MODELS,
        )
        await run_acp_agent(acp_agent)


def main() -> None:
    """Configure logging and run the ACP agent server."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_serve_agent())


if __name__ == "__main__":
    main()
