import json
import threading
from pathlib import Path

import httpx
import pytest

from memgit.config import Settings
from memgit.core.store import MemoryStore
from memgit.web.server import build_server

NOTE_A = (
    "# notes\n\n## Active\n\n"
    '- [mem-1] The user codes with Neovim daily | s:mem-sess1 | t:2 | "Neovim is all I code with"\n\n'
    "## Superseded\n\n"
)
NOTE_B = (
    "# notes\n\n## Active\n\n"
    '- [mem-2] The user codes with Neovim daily and often | s:mem-sess2 | t:1 | "Neovim is all I code with"\n\n'
    "## Superseded\n\n\n"
    '- [mem-1] The user codes with Neovim daily | s:mem-sess1 | t:2 | "Neovim is all I code with" | superseded_by:mem-2\n'
)


@pytest.fixture(autouse=True)
def running_servers():
    keep: list = []
    yield keep
    for server in keep:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def make_env(settings_factory, tmp_path: Path, running_servers):
    def build(commit_count: int = 1, rejections=None) -> tuple[httpx.Client, Settings, str]:
        settings = settings_factory(f"logging.rejections_log_path={tmp_path / 'rej.jsonl'}")
        mem = Path(settings.memory.dir)
        store = MemoryStore(settings)
        store.begin_session()
        (mem / "notes.md").write_text(NOTE_A, encoding="utf-8")
        store.commit_all("memory session mem-sess1\n\n+1 ~0 -0 (noop 0)\nlearned [mem-1] notes.md")
        store.finish_session()
        if commit_count > 1:
            store.begin_session()
            (mem / "notes.md").write_text(NOTE_B, encoding="utf-8")
            store.commit_all("memory session mem-sess2\n\n+1 ~1 -0 (noop 0)\nlearned [mem-2] notes.md\nwas  [mem-1]")
            store.finish_session()
        if rejections is not None:
            (tmp_path / "rej.jsonl").write_text(
                "\n".join(
                    json.dumps({"ts": "2026-01-01T00:00:00+00:00", "session_id": "mem-sess1", "reason": r, "op": op})
                    for r, op in rejections
                )
                + "\n",
                encoding="utf-8",
            )
        ephemeral = settings.model_copy(
            update={"web": settings.web.model_copy(update={"port": 0})}
        )
        known_token = "test-csrf-token-for-existing-tests"
        server = build_server(ephemeral, csrf_token=known_token)
        running_servers.append(server)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}")
        return client, settings, known_token

    return build


def test_history_lists_commits_and_links(make_env):
    client, settings, token = make_env(commit_count=2)
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "memory session mem-sess1" in body
    assert "memory session mem-sess2" in body
    assert "older" not in body  # only 2 commits, single page
    assert "Revert" not in body  # data text carries the session lines, buttons spell revert


def test_commit_view_wide(make_env):
    client, settings, token = make_env(commit_count=2)
    sha = MemoryStore(settings).log(5)[-1].short_sha
    response = client.get(f"/commit/{sha}")
    assert response.status_code == 200
    assert "memory session mem-sess1" in response.text
    assert "diff --git" in response.text


def test_commit_not_found(make_env):
    client, settings, token = make_env(commit_count=1)
    response = client.get("/commit/deadbeef")
    assert response.status_code == 409
    assert "no unique commit" in response.text


def test_revert_click_through(make_env):
    client, settings, token = make_env(commit_count=2)
    store = MemoryStore(settings)
    newest = store.log(5)[0]
    response = client.post(
        "/revert",
        data={"csrf_token": token, "commit": newest.sha},
        follow_redirects=True,
    )
    assert response.status_code == 200
    text = (Path(settings.memory.dir) / "notes.md").read_text(encoding="utf-8")
    assert "[mem-2]" not in text


def test_revert_error_rendered(make_env):
    client, settings, token = make_env(commit_count=1)
    empty_sha = None
    store = MemoryStore(settings)
    store.begin_session()
    empty_sha = store.commit_all("memory session mem-sess0 (empty)")
    store.finish_session()
    response = client.post(
        "/revert",
        data={"csrf_token": token, "commit": empty_sha},
    )
    assert response.status_code == 409
    assert "revert skipped" in response.text


def test_memory_page(make_env):
    client, settings, token = make_env(commit_count=1)
    response = client.get("/memory")
    assert response.status_code == 200
    assert "mem-1" in response.text and "Neovim" in response.text
    assert "1 active" in response.text and "0 superseded" in response.text


def test_rejections_page(make_env):
    rejections = [
        ("bad_path: outside allowed roots", {"file": "x.md"}),
        ("quote_not_found: short", {"quote": "nope"}),
    ]
    client, settings, token = make_env(commit_count=1, rejections=rejections)
    response = client.get("/rejections")
    assert response.status_code == 200
    assert "bad_path" in response.text and "quote_not_found" in response.text


def test_rejections_empty(make_env):
    client, settings, token = make_env(commit_count=1)
    response = client.get("/rejections")
    assert response.status_code == 200
    assert "(no rejections logged yet)" in response.text


def test_unknown_route(make_env):
    client, settings, token = make_env(commit_count=1)
    response = client.get("/definitely/not/a/route")
    assert response.status_code == 404
    assert "no route" in response.text


def test_api_endpoints(make_env):
    client, settings, token = make_env(commit_count=2)
    # Test /api/history
    resp = client.get("/api/history")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    commits = resp.json()
    assert len(commits) == 2
    assert "sha" in commits[0] and "subject" in commits[0]

    # Test /api/commit/<sha>
    sha = commits[0]["short_sha"]
    resp_commit = client.get(f"/api/commit/{sha}")
    assert resp_commit.status_code == 200
    data = resp_commit.json()
    assert "diff" in data and "subject" in data

    # Test /api/memories
    resp_mem = client.get("/api/memories")
    assert resp_mem.status_code == 200
    mem_files = resp_mem.json()
    assert len(mem_files) >= 1
    assert "active_count" in mem_files[0]

    # Test /api/rejections
    resp_rej = client.get("/api/rejections")
    assert resp_rej.status_code == 200
    assert isinstance(resp_rej.json(), list)

    # Test /api/stats
    resp_stats = client.get("/api/stats")
    assert resp_stats.status_code == 200
    stats = resp_stats.json()
    assert "active_memories" in stats and "commits" in stats
