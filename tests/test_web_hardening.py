"""Regression tests for web hardening: CSRF, origin, body-size, read-only.

Tests:
- missing CSRF token => 403
- wrong CSRF token => 403
- wrong origin => 403
- oversize body => 413
- missing Content-Length => 400
- valid POST => 303
- read_only mode => 403 on POST, read-only banner in GET
- negation gap in verify.py
"""
from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest

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


def _build_env(settings_factory, tmp_path, running_servers, read_only: bool = False):
    """Build a test server with a known CSRF token so we can test both paths."""
    overrides = [f"logging.rejections_log_path={tmp_path / 'rej.jsonl'}"]
    if read_only:
        overrides.append("web.read_only=true")
    settings = settings_factory(*overrides)
    mem = Path(settings.memory.dir)
    store = MemoryStore(settings)
    store.begin_session()
    (mem / "notes.md").write_text(NOTE_A, encoding="utf-8")
    store.commit_all("memory session mem-sess1\n\n+1 ~0 -0 (noop 0)\nlearned [mem-1] notes.md")
    store.finish_session()
    store.begin_session()
    (mem / "notes.md").write_text(NOTE_B, encoding="utf-8")
    store.commit_all("memory session mem-sess2\n\n+1 ~1 -0 (noop 0)\nlearned [mem-2] notes.md\nwas  [mem-1]")
    store.finish_session()

    ephemeral = settings.model_copy(
        update={"web": settings.web.model_copy(update={"port": 0})}
    )
    # Use a known CSRF token so tests can submit correct and incorrect tokens
    known_token = "test-csrf-token-known"
    server = build_server(ephemeral, csrf_token=known_token)
    running_servers.append(server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}")
    # Get the sha of the newest commit to revert
    newest_sha = MemoryStore(settings).log(5)[0].sha
    return client, settings, known_token, newest_sha


# ---------------------------------------------------------------------------
# CSRF tests
# ---------------------------------------------------------------------------


def test_post_missing_csrf_token_returns_403(settings_factory, tmp_path, running_servers):
    client, settings, token, sha = _build_env(settings_factory, tmp_path, running_servers)
    # Send POST without csrf_token field at all
    response = client.post(
        "/revert",
        content=f"commit={sha}".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 403
    assert b"CSRF" in response.content or b"403" in response.content


def test_post_wrong_csrf_token_returns_403(settings_factory, tmp_path, running_servers):
    client, settings, token, sha = _build_env(settings_factory, tmp_path, running_servers)
    response = client.post(
        "/revert",
        content=f"csrf_token=WRONG&commit={sha}".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 403


def test_post_valid_csrf_token_succeeds(settings_factory, tmp_path, running_servers):
    client, settings, token, sha = _build_env(settings_factory, tmp_path, running_servers)
    response = client.post(
        "/revert",
        content=f"csrf_token={token}&commit={sha}".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=True,
    )
    assert response.status_code == 200  # followed redirect to /


# ---------------------------------------------------------------------------
# Origin check tests
# ---------------------------------------------------------------------------


def test_post_wrong_origin_returns_403(settings_factory, tmp_path, running_servers):
    client, settings, token, sha = _build_env(settings_factory, tmp_path, running_servers)
    response = client.post(
        "/revert",
        content=f"csrf_token={token}&commit={sha}".encode(),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "http://evil.com",
        },
    )
    assert response.status_code == 403
    assert b"Origin" in response.content or b"403" in response.content


# ---------------------------------------------------------------------------
# Body-size tests
# ---------------------------------------------------------------------------


def test_oversize_body_returns_413(settings_factory, tmp_path, running_servers):
    """Sending a body larger than max_body_bytes (default 65536) must return 413."""
    overrides = [
        f"logging.rejections_log_path={tmp_path / 'rej.jsonl'}",
        "web.max_body_bytes=64",  # tiny limit for test
    ]
    settings = settings_factory(*overrides)
    mem = Path(settings.memory.dir)
    store = MemoryStore(settings)
    store.begin_session()
    (mem / "notes.md").write_text(NOTE_A, encoding="utf-8")
    store.commit_all("session")
    store.finish_session()
    ephemeral = settings.model_copy(update={"web": settings.web.model_copy(update={"port": 0})})
    server = build_server(ephemeral, csrf_token="tok")
    running_servers.append(server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    c = httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}")
    # Body is 100 bytes > 64 limit
    body = b"a" * 100
    response = c.post(
        "/revert",
        content=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 413


def test_missing_content_length_returns_400(settings_factory, tmp_path, running_servers):
    """POST without Content-Length header must return 400."""
    import socket

    overrides = [f"logging.rejections_log_path={tmp_path / 'rej.jsonl'}"]
    settings = settings_factory(*overrides)
    mem = Path(settings.memory.dir)
    store = MemoryStore(settings)
    store.begin_session()
    (mem / "notes.md").write_text(NOTE_A, encoding="utf-8")
    store.commit_all("session")
    store.finish_session()
    ephemeral = settings.model_copy(update={"web": settings.web.model_copy(update={"port": 0})})
    server = build_server(ephemeral, csrf_token="tok")
    running_servers.append(server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    # Send a raw HTTP/1.1 request without Content-Length (Transfer-Encoding: chunked)
    # The server must respond 400.
    raw_request = (
        b"POST /revert HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/x-www-form-urlencoded\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\n"
        b"tok=x\r\n"
        b"0\r\n"
        b"\r\n"
    )
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(raw_request)
        response_bytes = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_bytes += chunk
            if b"\r\n\r\n" in response_bytes:
                break

    status_line = response_bytes.split(b"\r\n")[0].decode("ascii", errors="replace")
    assert "400" in status_line, f"Expected 400 but got: {status_line!r}"



# ---------------------------------------------------------------------------
# Read-only mode
# ---------------------------------------------------------------------------


def test_read_only_post_returns_403(settings_factory, tmp_path, running_servers):
    """In read-only mode, POST /revert must return 403 without touching git."""
    client, settings, token, sha = _build_env(
        settings_factory, tmp_path, running_servers, read_only=True
    )
    response = client.post(
        "/revert",
        content=f"csrf_token={token}&commit={sha}".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 403
    assert b"read-only" in response.content.lower()


def test_read_only_get_shows_banner(settings_factory, tmp_path, running_servers):
    """In read-only mode, GET / must include the read-only banner."""
    client, settings, token, sha = _build_env(
        settings_factory, tmp_path, running_servers, read_only=True
    )
    response = client.get("/")
    assert response.status_code == 200
    assert "Read-only demo mode" in response.text or "read-only" in response.text.lower()


def test_read_write_get_no_banner(settings_factory, tmp_path, running_servers):
    """In normal (read-write) mode, GET / must NOT include the read-only banner."""
    client, settings, token, sha = _build_env(
        settings_factory, tmp_path, running_servers, read_only=False
    )
    response = client.get("/")
    assert response.status_code == 200
    assert "Read-only demo mode" not in response.text


# ---------------------------------------------------------------------------
# Negation gap in verify.py
# ---------------------------------------------------------------------------


def test_negation_blocks_affirmation_with_negation_word(settings_factory):
    """With negations=[\"not\"], 'that's not right' must NOT count as grounding."""
    from memgit.core.consolidate import ROLE_ASSISTANT, ROLE_USER, ProposedOp, Turn
    from memgit.core.verify import STATUS_REJECTED, Verifier

    s = settings_factory(
        "verify.allow_assistant_quotes=true",
        "verify.negations=[\"not\"]",
    )
    turns = [
        Turn(turn=1, role=ROLE_ASSISTANT, text="I configured the runner in us-east-1."),
        Turn(turn=2, role=ROLE_USER, text="that's not right"),
    ]
    verifier = Verifier(s, turns=turns, active_entries=[])
    op = ProposedOp(
        op="add",
        file="infra.md",
        text="Runner is in us-east-1",
        quote="configured the runner in us-east-1",
        turn=1,
        confidence=0.9,
    )
    verdict = verifier.verify_op(op)
    assert verdict.status == STATUS_REJECTED
    assert verdict.reason.startswith("quote_not_found")


def test_negation_empty_list_does_not_block(settings_factory):
    """With negations=[] (default), 'that's right' must still grant grounding."""
    from memgit.core.consolidate import ROLE_ASSISTANT, ROLE_USER, ProposedOp, Turn
    from memgit.core.verify import STATUS_ACCEPTED_ADD, Verifier

    s = settings_factory("verify.allow_assistant_quotes=true")
    assert s.verify.negations == []
    turns = [
        Turn(turn=1, role=ROLE_ASSISTANT, text="I configured the runner in us-east-1."),
        Turn(turn=2, role=ROLE_USER, text="that's right"),
    ]
    verifier = Verifier(s, turns=turns, active_entries=[])
    op = ProposedOp(
        op="add",
        file="infra.md",
        text="Runner is in us-east-1",
        quote="configured the runner in us-east-1",
        turn=1,
        confidence=0.9,
    )
    verdict = verifier.verify_op(op)
    assert verdict.status == STATUS_ACCEPTED_ADD


def test_negation_word_alone_in_turn_does_not_block_if_no_affirmation(settings_factory):
    """'No thanks' with no affirmation word still rejects (for different reason: no affirmation)."""
    from memgit.core.consolidate import ROLE_ASSISTANT, ROLE_USER, ProposedOp, Turn
    from memgit.core.verify import STATUS_REJECTED, Verifier

    s = settings_factory(
        "verify.allow_assistant_quotes=true",
        "verify.negations=[\"not\", \"no\"]",
    )
    turns = [
        Turn(turn=1, role=ROLE_ASSISTANT, text="I configured the runner in us-east-1."),
        Turn(turn=2, role=ROLE_USER, text="no thanks"),  # negation "no" blocks even without affirmation
    ]
    verifier = Verifier(s, turns=turns, active_entries=[])
    op = ProposedOp(
        op="add",
        file="infra.md",
        text="Runner is in us-east-1",
        quote="configured the runner in us-east-1",
        turn=1,
        confidence=0.9,
    )
    verdict = verifier.verify_op(op)
    assert verdict.status == STATUS_REJECTED
