import json
from pathlib import Path

import httpx
import pytest

from memgit.cli import bootstrap_settings
from memgit.core.consolidate import Consolidator, ProposedOp, Turn
from memgit.core.llm import BudgetCounter, CallBudgetExceeded, LLMClient, LLMError
from memgit.core.store import MemoryStore
from memgit.core.verify import STATUS_ACCEPTED_ADD, Verifier

TRANSCRIPT = [
    Turn(turn=1, role="assistant", text="Hello! anything to remember today?"),
    Turn(turn=2, role="user", text="Sure: I exclusively drink ginger tea during winter, it helps me focus."),
    Turn(turn=3, role="assistant", text="Noted, anything else?"),
]
VALID_OUTPUT = {
    "ops": [
        {
            "op": "add",
            "file": "personal/drinks.md",
            "text": "The user drinks ginger tea during winter",
            "quote": "I exclusively drink ginger tea during winter",
            "turn": 2,
            "supersedes": None,
            "confidence": 0.93,
        }
    ]
}


@pytest.fixture()
def mock_settings(settings_factory, api_key_env, tmp_path: Path) -> object:
    overrides = (f"memory.dir={tmp_path / 'mem'}", f"llm.base_url=http://testserver{MODEL_BASE}")
    return settings_factory(*overrides)


MODEL_BASE = "/v1"


def make_client(settings, models: list[str], chat_contents: list[str]):
    requests: list[dict] = []
    call_state = {"chat_used": 0}
    prompt_tokens = {"n": 0}
    completion_tokens = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={"object": "list", "data": [{"id": m} for m in models]},
            )
        if path.endswith("/chat/completions"):
            payload = json.loads(request.content.decode("utf-8"))
            requests.append(payload)
            content = chat_contents[call_state["chat_used"]]
            call_state["chat_used"] += 1
            prompt_tokens["n"] += 120
            completion_tokens["n"] += 30
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": payload["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": prompt_tokens["n"], "completion_tokens": completion_tokens["n"]},
                },
            )
        return httpx.Response(404, json={"error": f"unexpected path {path}"})

    transport = httpx.MockTransport(handler)
    client = LLMClient(
        settings,
        budget=BudgetCounter(),
        http_client=httpx.Client(transport=transport, timeout=settings.llm.timeout_s),
    )
    return client, requests


def test_model_discovery_and_roles(settings_factory, api_key_env):
    s = settings_factory(f"llm.base_url=http://testserver{MODEL_BASE}")
    client, _ = make_client(s, ["large-east", "mini-west"], [])
    assert client.resolve_model("consolidation") == "mini-west"
    assert client.resolve_model("judge") == "large-east"


def test_model_preference_patterns(settings_factory, api_key_env, monkeypatch):
    monkeypatch.setenv("MEMGIT_LLM__MODEL_PREFERENCE_PATTERNS", '["^kadabra"]')
    s = settings_factory(f"llm.base_url=http://testserver{MODEL_BASE}")
    client, _ = make_client(s, ["mini-west", "kadabra-lite", "large-east"], [])
    assert client.resolve_model("consolidation") == "kadabra-lite"


def test_consolidate_validates_and_applies(mock_settings, api_key_env, tmp_path: Path):
    s = mock_settings
    content = json.dumps(VALID_OUTPUT, ensure_ascii=False, indent=2)
    client, requests = make_client(s, ["mini-west", "large-east"], [content])
    store = MemoryStore(s, path=tmp_path / "mem")
    store.begin_session()
    consolidator = Consolidator(s, client)
    result = consolidator.propose(
        transcript=TRANSCRIPT,
        active_entries=store.active_entries(),
        next_id_example=store.next_id_example(),
    )
    assert result.ops[0].op == "add"
    assert not result.used_retry
    assert all(("consolidate" in r or True) for r in requests)
    body = requests[-1]
    assert body["model"] == "mini-west"
    verdicts = Verifier(s, turns=TRANSCRIPT, active_entries=store.active_entries()).verify_all(result.ops).verdicts
    assert [v.status for v in verdicts] == [STATUS_ACCEPTED_ADD]
    added, superseded = store.apply(verdicts)
    assert len(added) == 1 and added[0].id == "mem-1"
    lines = (tmp_path / "mem" / "personal" / "drinks.md").read_text(encoding="utf-8").splitlines()
    assert any("| t:2 |" in ln and "mem-1" in ln and "ginger tea" in ln for ln in lines)
    index_text = (tmp_path / "mem" / "index.md").read_text(encoding="utf-8")
    assert "drinks" in index_text
    assert client.totals()["calls"] == 1


def test_json_retry_feeds_validation_error(mock_settings, api_key_env, tmp_path: Path):
    s = mock_settings
    good = json.dumps(VALID_OUTPUT, ensure_ascii=False)
    client, requests = make_client(s, ["mini-west"], ["that is not json at all", good])
    store = MemoryStore(s, path=tmp_path / "mem")
    store.begin_session()
    consolidator = Consolidator(s, client)
    result = consolidator.propose(
        transcript=TRANSCRIPT,
        active_entries=store.active_entries(),
        next_id_example=store.next_id_example(),
    )
    assert result.used_retry
    assert len(requests) == 2
    retry_messages = requests[-1]["messages"]
    assert len(retry_messages) == 3
    assert retry_messages[1]["role"] == "assistant"
    assert retry_messages[2]["role"] == "user"
    assert "no JSON object found" in retry_messages[2]["content"]


def test_call_budget_guard(settings_factory, api_key_env, tmp_path: Path):
    s = settings_factory("llm.max_calls=1", f"memory.dir={tmp_path / 'mem'}", f"llm.base_url=http://testserver{MODEL_BASE}")
    content = json.dumps(VALID_OUTPUT, ensure_ascii=False)
    client, _ = make_client(s, ["mini-west"], [content, content])
    store = MemoryStore(s, path=tmp_path / "mem")
    store.begin_session()
    consolidator = Consolidator(s, client)
    consolidator.propose(
        transcript=TRANSCRIPT,
        active_entries=store.active_entries(),
        next_id_example=store.next_id_example(),
    )
    store.session_id = None
    store.begin_session()
    with pytest.raises(CallBudgetExceeded):
        consolidator.propose(
            transcript=TRANSCRIPT,
            active_entries=store.active_entries(),
            next_id_example=store.next_id_example(),
        )


def test_missing_key_fails_loudly(settings_factory, monkeypatch, tmp_path: Path):
    s = settings_factory(f"memory.dir={tmp_path / 'mem'}")
    monkeypatch.delenv("MEMGIT_API_KEY", raising=False)
    with pytest.raises(LLMError) as err:
        LLMClient(s)
    assert "MEMGIT_API_KEY" in str(err.value)
