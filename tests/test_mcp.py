import io
import json
from pathlib import Path

import httpx
import pytest

from memgit.cli import bootstrap_settings
from memgit.mcp import server as mcp_server
from memgit.mcp.server import ServerCore, process_message, run_stdio
from memgit.core.llm import BudgetCounter, LLMClient
from memgit.core.store import MemoryStore

CONSOLIDATE_ADD = {
    "ops": [
        {
            "op": "add",
            "file": "tools.md",
            "text": "The user codes with Neovim daily",
            "quote": "Neovim is all I code with",
            "turn": 2,
            "supersedes": None,
            "confidence": 0.93,
        }
    ]
}


class ChatMock:
    def __init__(self, models: list[str], chat_contents: list[str], fail_chat: bool = False) -> None:
        self.models = models
        self.chat_contents = list(chat_contents)
        self.fail_chat = fail_chat

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": m} for m in self.models]})
        if path.endswith("/chat/completions"):
            if self.fail_chat:
                return httpx.Response(500, json={"error": "boom"})
            payload = json.loads(request.content.decode("utf-8"))
            content = self.chat_contents.pop(0)
            return httpx.Response(
                200,
                json={
                    "id": "mock",
                    "object": "chat.completion",
                    "model": payload["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 6},
                },
            )
        return httpx.Response(404, json={"error": path})


@pytest.fixture()
def core_factory(settings_factory, api_key_env, tmp_path: Path):
    def build(chat_contents: list[str], fail_chat: bool = False, extra_overrides: tuple = ()):
        s = settings_factory(
            f"memory.dir={tmp_path / 'mem'}",
            f"logging.usage_log_path={tmp_path / 'usage.jsonl'}",
            f"logging.rejections_log_path={tmp_path / 'rejections.jsonl'}",
            "llm.base_url=http://testserver/v1",
            "llm.max_retries=0",
            "llm.backoff_s=0",
            *extra_overrides,
        )
        mock = ChatMock(["mini-west", "large-east"], chat_contents, fail_chat=fail_chat)
        client = LLMClient(
            s,
            budget=BudgetCounter(),
            http_client=httpx.Client(
                transport=httpx.MockTransport(mock.handler), timeout=s.llm.timeout_s
            ),
        )
        core = ServerCore(s, llm_factory=lambda settings: client)
        return core, mock, s

    return build


@pytest.fixture()
def seeded_core(core_factory):
    core, mock, s = core_factory([])
    store = MemoryStore(s, path=Path(s.memory.dir))
    store.begin_session()
    (Path(s.memory.dir) / "notes.md").write_text(
        '# notes\n\n## Active\n\n- [mem-1] The user codes with Neovim daily | s:mem-sess1 | t:2 | "Neovim is all I code with"\n\n## Superseded\n\n',
        encoding="utf-8",
    )
    store.commit_all("memory session mem-sess1\n\n+1 ~0 -0 (noop 0)\nlearned [mem-1] notes.md")
    store.finish_session()
    store.begin_session()
    store.commit_all("memory session mem-sess2 (empty)")
    store.finish_session()
    return core, s


def test_tools_registry_has_schemas(seeded_core):
    core, _ = seeded_core
    specs = core.tools()
    names = {spec["name"] for spec in specs}
    assert names == {
        "memory_recall",
        "remember_session",
        "memory_history",
        "memory_find",
        "memory_diff",
        "revert_commit",
    }
    for spec in specs:
        assert spec["description"]
        assert spec["inputSchema"]["type"] == "object"


def test_history_and_find_tools(seeded_core):
    core, s = seeded_core
    text = core.call_tool("memory_history", {"limit": 1})
    assert "mem-sess2" in text and "mem-sess1" not in text.splitlines()[-1]
    found = core.call_tool("memory_find", {"query": "sess1"})
    assert "memory session mem-sess1" in found
    missing = core.call_tool("memory_find", {"query": "nope"})
    assert "(no commits matching 'nope')" in missing


def test_revert_and_diff_tools(seeded_core):
    core, s = seeded_core
    commits = MemoryStore(s).log(5)
    first_sha = commits[-1].sha
    empty_commit = commits[0].sha
    with pytest.raises(mcp_server.ToolError) as exc_info:
        core.call_tool("revert_commit", {"commit": empty_commit})
    assert "revert skipped" in str(exc_info.value)
    text = core.call_tool("revert_commit", {"commit": first_sha})
    assert text.startswith("revert committed: ")
    assert not (Path(s.memory.dir) / "notes.md").exists()
    diff = core.call_tool("memory_diff", {"commit": first_sha})
    assert "notes.md" in diff


def test_remember_session_commits_memory(core_factory):
    core, mock, s = core_factory([json.dumps(CONSOLIDATE_ADD)])
    transcript = [
        {"turn": 1, "role": "assistant", "text": "hello!"},
        {"turn": 2, "role": "user", "text": "Neovim is all I code with."},
    ]
    text = core.call_tool("remember_session", {"transcript": transcript})
    assert "memory session mem-sess1" in text
    assert "committed:" in text
    mem = Path(s.memory.dir)
    stored = (mem / "tools.md").read_text(encoding="utf-8")
    assert "Neovim" in stored
    assert len(MemoryStore(s).log(10)) == 1


def test_remember_session_llm_failure_is_error(core_factory):
    core, mock, s = core_factory(["```also garbage```", "not json"], fail_chat=False)
    before = len(MemoryStore(s).log(10)) if Path(s.memory.dir).joinpath(".git").exists() else 0
    transcript = [{"turn": 1, "role": "user", "text": "Neovim is all I code with."}]
    with pytest.raises(mcp_server.ToolError) as exc_info:
        core.call_tool("remember_session", {"transcript": transcript})
    assert "session not consolidated" in str(exc_info.value)
    assert not Path(s.memory.dir).exists() or len(MemoryStore(s).log(10)) == before


def test_process_message_protocol():
    settings = bootstrap_settings(None, None).settings
    core = ServerCore(settings)
    reply = process_message(core, {"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}})
    assert reply["id"] == 7
    assert reply["result"]["serverInfo"]["name"] == settings.mcp.server_name
    assert reply["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION
    assert reply["result"]["capabilities"]["tools"] == {"listChanged": False}

    assert process_message(core, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert process_message(core, {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"id": 3}}) is None

    err = process_message(core, {"jsonrpc": "2.0", "id": 1, "method": "nope"})
    assert err["error"]["code"] == mcp_server.METHOD_NOT_FOUND

    call = process_message(
        core, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "no_such_tool"}}
    )
    assert call["error"]["code"] == mcp_server.INVALID_PARAMS

    bad_params = process_message(
        core, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "memory_recall"}}
    )
    assert bad_params["error"]["code"] == mcp_server.INVALID_PARAMS

    ping = process_message(core, {"jsonrpc": "2.0", "id": 9, "method": "ping"})
    assert ping["result"] == {}


def test_remember_session_invalid_transcript(core_factory):
    core, mock, s = core_factory([])
    with pytest.raises(mcp_server.ToolError) as exc_info:
        core.call_tool("remember_session", {"transcript": [{"turn": 0, "role": "robot", "text": "x"}]})
    assert "transcript entry invalid" in str(exc_info.value)


def test_run_stdio_roundtrip(seeded_core):
    core, s = seeded_core
    incoming = "\n".join(
        [
            "this is not json {",
            '{"jsonrpc": "2.0", "method": "notifications/initialized"}',
            '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}',
            '{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}',
            '{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "memory_history", "arguments": {"limit": 1}}}',
            '{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "revert_commit", "arguments": {"commit": "c0ffee0"}}}',
        ]
    )
    out = io.StringIO()
    run_stdio(core, stdin=io.StringIO(incoming), stdout=out)
    lines = out.getvalue().strip().splitlines()
    assert len(lines) == 5  # garbage, initialize, tools/list, history, revert(isError)
    parse_err = json.loads(lines[0])
    assert parse_err["id"] is None
    assert parse_err["error"]["code"] == mcp_server.PARSE_ERROR
    init = json.loads(lines[1])
    assert init["id"] == 1 and init["result"]["serverInfo"]["name"] == s.mcp.server_name
    listing = json.loads(lines[2])
    assert {t["name"] for t in listing["result"]["tools"]}
    history = json.loads(lines[3])
    text = "".join(c["text"] for c in history["result"]["content"])
    assert "mem-sess2" in text
    revert = json.loads(lines[4])
    assert revert["result"]["isError"] is True
