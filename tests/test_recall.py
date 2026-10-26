import json
from pathlib import Path

import httpx
import pytest

from memgit.core.llm import BudgetCounter, LLMClient, LLMError
from memgit.core.retrieve import Recaller
from memgit.core.store import MemoryStore


BASE = "http://testserver/v1"


class MockHub:
    def __init__(self, *, models: list[str], selection: dict | None = None, fail_chat: bool = False) -> None:
        self.models = models
        self.selection = selection
        self.fail_chat = fail_chat
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": m} for m in self.models]})
        if path.endswith("/chat/completions"):
            self.requests.append(json.loads(request.content.decode("utf-8")))
            if self.fail_chat:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "model": self.models[0],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": json.dumps(self.selection)}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        return httpx.Response(404, json={"error": path})


def seed_memory(tmp_path: Path) -> Path:
    mem = tmp_path / "mem"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "tools.md").write_text(
        '# tools\n\n## Active\n\n'
        '- [mem-1] The user codes with Neovim daily | s:mem-s1 | t:2 | "Neovim is all I code with"\n\n'
        '## Superseded\n\n', encoding="utf-8")
    (mem / "tea.md").write_text(
        '# tea\n\n## Active\n\n'
        '- [mem-4] The user drinks ginger tea in winter | s:mem-s1 | t:3 | "ginger tea all winter long"\n'
        '- [mem-5] The user avoids coffee in the evenings | s:mem-s2 | t:6 | "no coffee after four pm"\n\n'
        '## Superseded\n\n', encoding="utf-8")
    return mem


def make(settings_factory, api_key_env, tmp_path: Path, hub: MockHub, extra: tuple = ()):
    s = settings_factory(
        f"memory.dir={tmp_path / 'mem'}",
        f"llm.base_url={BASE}",
        "llm.max_retries=0",
        "llm.backoff_s=0",
        *extra,
    )
    client = LLMClient(
        s,
        budget=BudgetCounter(),
        http_client=httpx.Client(transport=httpx.MockTransport(hub.handler), timeout=s.llm.timeout_s),
    )
    return with_store(s, client, tmp_path)


def with_store(s, client, tmp_path: Path) -> tuple[Recaller, MemoryStore]:
    store = MemoryStore(s, path=tmp_path / "mem")
    return Recaller(s, client), store


def test_llm_selection_returns_block_only_selected(settings_factory, api_key_env, tmp_path):
    mem = seed_memory(tmp_path)
    hub = MockHub(models=["mini-west"], selection={"files": ["tea.md"]})
    recaller, store = make(settings_factory, api_key_env, tmp_path, hub)
    result = recaller.recall("what do I drink with dinner")
    assert result.files == ["tea.md"]
    assert result.used_llm and not result.used_grep
    assert "mem-4" in result.block and "mem-5" in result.block
    assert "mem-1" not in result.block


def test_llm_selects_unknown_path_filtered(settings_factory, api_key_env, tmp_path):
    mem = seed_memory(tmp_path)
    hub = MockHub(models=["mini-west"], selection={"files": ["tea.md", "madeup.md"]})
    recaller, store = make(settings_factory, api_key_env, tmp_path, hub)
    result = recaller.recall("tea")
    assert result.files == ["tea.md"]


def test_grep_fallback_on_llm_failure(settings_factory, api_key_env, tmp_path):
    mem = seed_memory(tmp_path)
    hub = MockHub(models=["mini-west"], fail_chat=True)
    recaller, store = make(settings_factory, api_key_env, tmp_path, hub)
    result = recaller.recall("what do I drink tonight")
    assert result.used_grep and not result.used_llm
    assert result.files  # tea.md matched via "drink"
    assert any(rel == "tea.md" for rel in result.files)


def test_fallback_disabled_propagates(settings_factory, api_key_env, tmp_path):
    mem = seed_memory(tmp_path)
    hub = MockHub(models=["mini-west"], fail_chat=True)
    recaller, store = make(settings_factory, api_key_env, tmp_path, hub, extra=("recall.fallback_grep=false",))
    with pytest.raises(LLMError):
        recaller.recall("what do I drink tonight")


def test_token_budget_truncates(settings_factory, api_key_env, tmp_path):
    mem = seed_memory(tmp_path)
    hub = MockHub(models=["mini-west"], selection={"files": ["tea.md"]})
    recaller, store = make(settings_factory, api_key_env, tmp_path, hub, extra=("memory.max_recall_tokens=20",))
    result = recaller.recall("tea")
    assert result.truncated
    assert len(result.lines) == 1
    assert result.tokens <= 40


def test_selection_requires_keys(settings_factory, api_key_env, tmp_path):
    mem = seed_memory(tmp_path)
    hub = MockHub(models=["mini-west"], selection={"paths": ["tea.md"]})
    recaller, store = make(
        settings_factory, api_key_env, tmp_path, hub, extra=("recall.fallback_grep=false", "llm.json_retries=0")
    )
    with pytest.raises(LLMError):
        recaller.recall("tea")
