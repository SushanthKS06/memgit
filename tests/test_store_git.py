from pathlib import Path

import pytest

from memgit.core.schema import MemoryLine
from memgit.core.session import render_commit_message
from memgit.core.store import MemoryStore, StoreError


@pytest.fixture()
def store(settings_factory, tmp_path: Path) -> MemoryStore:
    s = settings_factory(f"memory.dir={tmp_path / 'mem'}")
    return MemoryStore(s, path=tmp_path / "mem")


def write_one(store: MemoryStore) -> None:
    path = store.root / "notes.md"
    path.write_text(
        '# notes\n\n## Active\n\n- [mem-1] The user keeps notes in git | s:mem-sess1 | t:1 | "notes in git"\n\n## Superseded\n',
        encoding="utf-8",
    )


def test_commit_log_diff(store: MemoryStore):
    write_one(store)
    store.begin_session()
    sha = store.commit_all("first commit")
    assert len(sha) >= 7
    commits = store.log(5)
    assert len(commits) == 1
    assert commits[0].subject == "first commit"
    diff = store.diff_against(commits[0].sha)
    assert "notes.md" in diff
    assert store.show(commits[0].sha).startswith("commit ")


def test_commit_requires_session(store: MemoryStore):
    write_one(store)
    with pytest.raises(StoreError):
        store.commit_all("should fail")


def test_revert_creates_commit(store: MemoryStore):
    write_one(store)
    store.begin_session()
    sha = store.commit_all("first commit")
    store.finish_session()
    (store.root / "extra.md").write_text("# extra\n\n## Active\n\n## Superseded\n", encoding="utf-8")
    store.begin_session()
    store.commit_all("second commit")
    store.finish_session()
    assert len(store.log(10)) == 2
    revert_sha = store.revert(sha)
    assert revert_sha != sha
    assert not (store.root / "notes.md").exists()
    assert len(store.log(10)) == 3


def test_commit_message_from_templates(settings_factory):
    s = settings_factory()
    new_line = MemoryLine(id="mem-12", fact="The user prefers vim", session="mem-sess3", turn=2, quote="I switched to vim now")
    old_line = MemoryLine(id="mem-4", fact="The user prefers nano", session="mem-sess1", turn=1, quote="I use nano daily")
    message = render_commit_message(
        s,
        session_id="mem-sess3",
        added=[new_line],
        superseded=[(old_line, "mem-12", new_line)],
        rejected=[(new_line.model_copy(update={"file": "x.md"}), "quote_not_found: testing")],
        noop_count=0,
    )
    assert "memory session mem-sess3" in message
    assert "+1 ~1 -1" in message
    assert "learned [mem-12]" in message
    assert "was [mem-4]" in message
    assert "rejected: quote_not_found" in message
    empty = render_commit_message(s, session_id="mem-sess3", added=[], superseded=[], rejected=[], noop_count=1)
    assert "(nothing stored this session)" in empty


def test_rejections_jsonl_appends(settings_factory, tmp_path: Path):
    from memgit.core.session import append_rejections

    s = settings_factory(f"logging.rejections_log_path={tmp_path / 'rej.jsonl'}")
    op = MemoryLine(id="x", fact="f", session="s", turn=1, quote="q")
    from memgit.core.consolidate import ProposedOp

    bad = ProposedOp(op="add", file="b.md", text="t", quote="q", turn=1, confidence=0.5)
    append_rejections(s, "sess-9", [(bad, "bad_path: oops")])
    append_rejections(s, "sess-10", [(bad, "bad_target: gone")])
    lines = (tmp_path / "rej.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    import json

    first = json.loads(lines[0])
    assert first["session_id"] == "sess-9"
    assert first["reason"] == "bad_path: oops"
    assert first["op"]["file"] == "b.md"


def test_compact_prunes_superseded(store: MemoryStore):
    from memgit.core.schema import MemoryFileContent, format_memory_file

    line_act = MemoryLine(id="mem-2", fact="Active fact", session="mem-sess2", turn=1, quote="quote here")
    line_sup = MemoryLine(id="mem-1", fact="Old fact", session="mem-sess1", turn=1, quote="old quote", superseded_by="mem-2")
    content = MemoryFileContent(header="notes", active=[line_act], superseded=[line_sup])
    (store.root / "notes.md").write_text(format_memory_file(content, store.cfg), encoding="utf-8")

    store.begin_session()
    store.commit_all("initial session")
    store.finish_session()

    removed, sha = store.compact()
    assert removed == 1
    assert len(sha) >= 7

    commits = store.log(5)
    assert len(commits) == 2
    assert "compact" in commits[0].subject

    text = (store.root / "notes.md").read_text(encoding="utf-8")
    assert "mem-1" not in text
    assert "mem-2" in text

    # Second compaction is a no-op
    removed2, sha2 = store.compact()
    assert removed2 == 0
    assert sha2 == ""


def test_store_lock_reentrant(store: MemoryStore):
    with store.lock():
        with store.lock():
            assert store.lock()._depth == 2
        assert store.lock()._depth == 1
    assert store.lock()._depth == 0
