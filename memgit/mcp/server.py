"""Minimal MCP stdio server: newline-delimited JSON-RPC 2.0 over stdin/stdout.

Implements the tools slice of the Model Context Protocol (initialize,
tools/list, tools/call, ping; notifications accepted and ignored) without the
official SDK. Every tunable stays in config: the server name comes from
[mcp].server_name, memory behaviour from [memory]/[git]/[recall]/[prompts].

Protocol output is single-line JSON on stdout only; diagnostics go to stderr.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable, Dict, List, Optional

import memgit
from memgit.config import Settings
from memgit.core.consolidate import Turn
from memgit.core.retrieve import Recaller
from memgit.core.session import MemorySession
from memgit.core.store import MemoryStore, StoreError
from memgit.core.llm import BudgetCounter, LLMClient, LLMError

logger = logging.getLogger("memgit.mcp")

PROTOCOL_VERSION = "2024-11-05"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class ToolError(RuntimeError):
    """Tool-level failure surfaced as an MCP isError result, not a protocol error."""


def rpc_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def rpc_result(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def tool_text(text: str, is_error: bool = False) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class ServerCore:
    """Tool registry + handlers; transport-agnostic so tests can drive it directly."""

    def __init__(
        self,
        settings: Settings,
        llm_factory: Optional[Callable[[Settings], LLMClient]] = None,
    ) -> None:
        self.cfg = settings
        self._llm_factory = llm_factory
        self._client: Optional[LLMClient] = None
        self._budget = BudgetCounter()

    def client(self) -> LLMClient:
        if self._client is None:
            if self._llm_factory is None:
                self._client = LLMClient(self.cfg, budget=self._budget)
            else:
                self._client = self._llm_factory(self.cfg)
        return self._client

    def store(self) -> MemoryStore:
        return MemoryStore(self.cfg)

    # -- tools ----------------------------------------------------------------

    def remember_session(self, arguments: Dict[str, Any]) -> str:
        transcript = arguments["transcript"]
        parsed: List[Turn] = []
        for item in transcript:
            if not isinstance(item, dict):
                raise ToolError("transcript entries must be objects")
            try:
                parsed.append(Turn.model_validate(item))
            except Exception as exc:
                raise ToolError(f"transcript entry invalid: {exc}") from exc
        outcome = MemorySession(self.cfg, self.store(), self.client()).run(parsed)
        if not outcome.ok:
            raise ToolError(outcome.message)
        return f"{outcome.message}\ncommitted: {outcome.commit_sha}"

    def memory_recall(self, arguments: Dict[str, Any]) -> str:
        query = arguments["query"]
        result = Recaller(self.cfg, self.client()).recall(query)
        files = ", ".join(result.files) if result.files else "(none)"
        return (
            f"{result.block}\n---\nfiles: {files}\n"
            f"tokens: {result.tokens} (used_llm={result.used_llm} used_grep={result.used_grep})"
        )

    def memory_history(self, arguments: Dict[str, Any]) -> str:
        limit = arguments.get("limit")
        if limit is None:
            limit = self.cfg.git.log_limit
        if not isinstance(limit, int) or limit < 1:
            raise ToolError("limit must be an integer >= 1")
        commits = self.store().log(limit)
        if not commits:
            return "(no commits yet)"
        return "\n".join(f"{c.short_sha}  {c.date}  {c.subject}" for c in commits)

    def memory_find(self, arguments: Dict[str, Any]) -> str:
        query = str(arguments["query"]).lower()
        limit = arguments.get("limit")
        if limit is None:
            limit = self.cfg.git.log_limit
        if not isinstance(limit, int) or limit < 1:
            raise ToolError("limit must be an integer >= 1")
        commits = self.store().log(limit)
        hits = [c for c in commits if query in c.subject.lower() or query in c.body.lower()]
        if not hits:
            return f"(no commits matching {query!r})"
        return "\n".join(f"{c.short_sha}  {c.date}  {c.subject}" for c in hits)

    def revert_commit(self, arguments: Dict[str, Any]) -> str:
        commit = arguments["commit"]
        try:
            new_sha = self.store().revert(commit)
        except StoreError as exc:
            raise ToolError(f"revert failed: {exc}") from exc
        return f"revert committed: {new_sha}"

    def memory_diff(self, arguments: Dict[str, Any]) -> str:
        commit = arguments["commit"]
        out = self.store().diff_against(commit)
        if not out.strip():
            return "(no changes vs parent)"
        return out

    # -- registry --------------------------------------------------------------

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_recall",
                "description": "Fetch relevant memory context for a query; returns the verbatim memory block.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "what the memory should answer"}},
                },
                "required": ["query"],
            },
            {
                "name": "remember_session",
                "description": (
                    "Consolidate a finished chat transcript into the memory repo with a git commit. "
                    "Facts need verbatim quotes from the transcript."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "transcript": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "turn": {"type": "integer", "minimum": 1},
                                    "role": {"enum": ["user", "assistant"]},
                                    "text": {"type": "string"},
                                },
                                "required": ["turn", "role", "text"],
                            },
                            "minItems": 1,
                        }
                    },
                    "required": ["transcript"],
                },
            },
            {
                "name": "memory_history",
                "description": "List memory commits, newest first (sha, date, subject).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 1}},
                },
            },
            {
                "name": "memory_find",
                "description": "Search commit messages for a substring.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_diff",
                "description": "Show what one commit changed (diff vs parent).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"commit": {"type": "string"}},
                    "required": ["commit"],
                },
            },
            {
                "name": "revert_commit",
                "description": "Roll back one memory commit with a git revert commit.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"commit": {"type": "string"}},
                    "required": ["commit"],
                },
            },
        ]

    _handlers = {
        "memory_recall": memory_recall,
        "remember_session": remember_session,
        "memory_history": memory_history,
        "memory_find": memory_find,
        "memory_diff": memory_diff,
        "revert_commit": revert_commit,
    }

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolError(f"unknown tool: {name}")
        return handler(self, arguments or {})


REQUEST_METHODS = ("initialize", "ping", "tools/list", "tools/call")


def process_message(core: ServerCore, message: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return rpc_error(None, INVALID_REQUEST, "request must be a JSON object")
    request_id = message.get("id", None)
    is_notification = "id" not in message
    if not isinstance(message.get("method", ""), str):
        if is_notification:
            return None
        return rpc_error(request_id, INVALID_REQUEST, "method must be a string")
    method = message["method"]
    params = message.get("params") or {}
    if not isinstance(params, dict):
        if is_notification:
            return None
        return rpc_error(request_id, INVALID_PARAMS, "params must be an object")

    if is_notification:
        return None

    if method == "initialize":
        return rpc_result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": core.cfg.mcp.server_name,
                    "version": memgit.__version__,
                },
            },
        )
    if method == "ping":
        return rpc_result(request_id, {})
    if method == "tools/list":
        return rpc_result(request_id, {"tools": core.tools()})
    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            return rpc_error(request_id, INVALID_PARAMS, "params.name must be a string")
        spec = next((t for t in core.tools() if t["name"] == name), None)
        if spec is None:
            return rpc_error(request_id, INVALID_PARAMS, f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return rpc_error(request_id, INVALID_PARAMS, "params.arguments must be an object")
        missing = [required for required in spec.get("required", []) if required not in arguments]
        if missing:
            return rpc_error(request_id, INVALID_PARAMS, f"missing arguments: {', '.join(missing)}")
        try:
            text = core.call_tool(name, arguments)
        except ToolError as exc:
            return rpc_result(request_id, tool_text(str(exc), is_error=True))
        except LLMError as exc:
            logger.warning("tools/call %s llm error: %s", name, exc)
            return rpc_result(request_id, tool_text(str(exc), is_error=True))
        except Exception as exc:  # keep the server alive; surface loudly on stderr
            logger.exception("tools/call %s crashed", name)
            return rpc_result(request_id, tool_text(f"{type(exc).__name__}: {exc}", is_error=True))
        return rpc_result(request_id, tool_text(text))
    return rpc_error(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")


def run_stdio(
    core: ServerCore,
    stdin: Optional[Any] = None,
    stdout: Optional[Any] = None,
) -> None:
    fin = stdin if stdin is not None else sys.stdin
    fout = stdout if stdout is not None else sys.stdout
    line_number = 0
    for raw in fin:
        line_number += 1
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("unparseable line %d: %s", line_number, exc)
            response = rpc_error(None, PARSE_ERROR, f"parse error: {exc}")
        else:
            response = process_message(core, message)
        if response is None:
            continue
        fout.write(json.dumps(response, ensure_ascii=False) + "\n")
        fout.flush()
