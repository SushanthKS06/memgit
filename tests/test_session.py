import json
from pathlib import Path

import httpx
import pytest

from memgit.cli import bootstrap_settings
from memgit.core.answer import Answerer
from memgit.core.consolidate import Consolidator, Turn
from memgit.core.llm import BudgetCounter, LLMClient, LLMError
from memgit.core.retrieve import Recaller
from memgit.core.session import MemorySession
from memgit.core.store import MemoryStore

SESSION_A_CONFIG = {
    "ops": [
        {
            "op": "add",
            "file": "tools.md",
            "text": "The user's main editor is VS Code",
            "quote": "I use VS Code as my main editor for coding work",
            "turn": 2,
            "supersedes": None,
            "confidence": 0.94,
        }
    ]
}
SESSION_B_CONFIG = {
    "ops": [
        {
            "op": "supersede",
            "file": "tools.md",
            "text": "The user's main editor is now Zed",
            "quote": "I switched away from VS Code to Zed as my main editor",
            "turn": 1,
            "supersedes": "mem-1",
            "confidence": 0.95,
        }
    ]
}


class ChatMock:
    def __init__(self, models: list[str], chat_contents: list[str]) -> None:
        self.models = models
        self.chat_contents = list(chat_contents)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": m} for m in self.models]}
            )
        if path.endswith("/chat/completions"):
            payload = json.loads(request.content.decode("utf-8"))
            self.requests.append(payload)
            content = self.chat_contents.pop(0)
            return httpx.Response(
                200,
                json={
                    "id": "mock",
                    "object": "chat.completion",
                    "model": payload["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 150, "completion_tokens": 50},
                },
            )
        return httpx.Response(404, json={"error": path})


def make(settings, mock: ChatMock):
    transport = httpx.MockTransport(mock.handler)
    client = LLMClient(
        settings,
        budget=BudgetCounter(),
        http_client=httpx.Client(transport=transport, timeout=settings.llm.timeout_s),
    )
    return client


def make_env(settings_factory, api_key_env, tmp_path: Path, contents: list[str]):
    s = settings_factory(
        f"memory.dir={tmp_path / 'mem'}",
        "llm.base_url=http://testserver/v1",
        f"logging.usage_log_path={tmp_path / 'usage.jsonl'}",
        f"logging.rejections_log_path={tmp_path / 'rejections.jsonl'}",
    )
    mock = ChatMock(["mini-west", "large-east"], contents)
    client = make(s, mock)
    store = MemoryStore(s, path=tmp_path / "mem")
    session = MemorySession(s, store, client)
    return s, store, session, mock, client


TURN_A = [
    Turn(turn=1, role="assistant", text="Hi!"),
    Turn(turn=2, role="user", text="You know what, I use VS Code as my main editor for coding work."),
]
TURN_B = [
    Turn(turn=1, role="user", text="Heads up: I switched away from VS Code to Zed as my main editor."),
]


def test_two_sessions_add_then_supersede(settings_factory, api_key_env, tmp_path: Path):
    s, store, session, mock, client = make_env(
        settings_factory, api_key_env, tmp_path, [json.dumps(SESSION_A_CONFIG), json.dumps(SESSION_B_CONFIG)]
    )
    outcome_a = session.run(TURN_A)
    assert outcome_a.ok
    assert len(outcome_a.added) == 1
    assert outcome_a.added[0].id == "mem-1"
    assert outcome_a.commit_sha is not None
    assert "+1 ~0 -0" in outcome_a.message
    tools_path = store.root / "tools.md"
    assert tools_path.is_file()
    text_after_a = tools_path.read_text(encoding="utf-8")
    assert "mem-1" in text_after_a
    assert "## Superseded" in text_after_a

    outcome_b = session.run(TURN_B)
    assert outcome_b.ok
    assert len(outcome_b.superseded) == 1
    old_line, new_id, new_line = outcome_b.superseded[0]
    assert old_line.id == "mem-1"
    assert new_line.id == "mem-2"
    assert "~1" in outcome_b.message
    text_after_b = tools_path.read_text(encoding="utf-8")
    assert "superseded_by:mem-2" in text_after_b
    assert "main editor is now Zed" in text_after_b
    assert len(store.log(10)) == 2
    assert not (tmp_path / "rejections.jsonl").exists()


def test_consolidate_failure_no_partial_writes(settings_factory, api_key_env, tmp_path: Path):
    s, store, session, mock, client = make_env(settings_factory, api_key_env, tmp_path, ["garbage not json", "```still not json```"])
    commits_before = len(store.log(10))
    files_before = sorted(p.name for p in store.root.rglob("*") if p.is_file())
    outcome = session.run(TURN_A)
    assert not outcome.ok
    assert outcome.error
    assert "session not consolidated" in outcome.message
    assert len(store.log(10)) == commits_before
    files_after = sorted(p.name for p in store.root.rglob("*") if p.is_file())
    assert files_before == files_after
    assert not (tmp_path / "rejections.jsonl").exists()


def test_rejections_jsonl_written_on_rejected_ops(settings_factory, api_key_env, tmp_path: Path):
    bad_output = {
        "ops": [
            {
                "op": "add",
                "file": "evil/../traversal.md",
                "text": "The user drinks ginger tea during winter",
                "quote": "I exclusively drink ginger tea during winter",
                "turn": 2,
                "confidence": 0.9,
            },
            {
                "op": "add",
                "file": "handbook/google/faq.md",
                "text": "The user prefers split keyboards for long sessions",
                "quote": "split keyboards help my wrists feel great lately",
                "turn": 2,
                "confidence": 0.8,
            },
        ]
    }
    transcript = [
        Turn(turn=1, role="user", text="I exclusively drink ginger tea during winter and prefer split keyboards for long sessions."),
        Turn(turn=2, role="user", text="I exclusively drink ginger tea during winter and prefer split keyboards for long sessions."),
    ]
    s, store, session, mock, client = make_env(settings_factory, api_key_env, tmp_path, [json.dumps(bad_output)])
    outcome = session.run(transcript)
    assert outcome.ok
    assert len(outcome.rejected) == 2
    reasons = [r for _, r in outcome.rejected]
    assert any(r.startswith("bad_path") for r in reasons)
    rejections_file = tmp_path / "rejections.jsonl"
    assert rejections_file.is_file()
    records = [json.loads(line) for line in rejections_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(records) == 2


def test_answer_prompt_contains_memory_block(settings_factory, api_key_env, tmp_path: Path):
    s = settings_factory(
        f"memory.dir={tmp_path / 'mem'}",
        "llm.base_url=http://testserver/v1",
        "memory.max_recall_files=2",
    )
    (tmp_path / "mem").mkdir()
    (tmp_path / "mem" / "prefs.md").write_text(
        '# prefs\n\n## Active\n\n- [mem-3] The user drinks ginger tea in winter | s:mem-s1 | t:4 | "ginger tea all winter"\n\n## Superseded\n\n',
        encoding="utf-8",
    )
    mock = ChatMock(["mini-west"], ['{"files": ["prefs.md"]}', "you usually drink ginger tea in winter"])
    client = make(s, mock)
    recaller = Recaller(s, client)
    answerer = Answerer(s, client)
    result = recaller.recall("what tea do I drink")
    assert result.used_llm and not result.used_grep
    assert "ginger tea" in result.block
    assert "mem-3" in result.block
    assert "| s:" in result.block
    reply = answerer.answer(
        [Turn(turn=1, role="user", text="what tea do I drink?")],
        result.block,
    )
    assert reply == "you usually drink ginger tea in winter"
    selection_payload = mock.requests[0]
    answer_payload = mock.requests[1]
    assert "prefs.md" in selection_payload["messages"][0]["content"]
    assert "ginger tea" in answer_payload["messages"][0]["content"]
