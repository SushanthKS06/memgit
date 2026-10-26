import pytest

from memgit.config import Settings
from memgit.core.consolidate import ProposedOp, Turn
from memgit.core.schema import MemoryLine
from memgit.core.verify import (
    STATUS_ACCEPTED_ADD,
    STATUS_ACCEPTED_SUPERSEDE,
    STATUS_NOOP,
    STATUS_REJECTED,
    Verifier,
    normalise,
)

USER_TURN_1 = "I exclusively drink ginger tea during winter, it helps me focus."
USER_TURN_2 = "Reminder that my go-to keyboard is a split ortholinear with thumb clusters."
QUOTABLE_1 = "I exclusively drink ginger tea during winter"
QUOTABLE_2 = "my go-to keyboard is a split ortholinear"
ASSISTANT_ONLY = "totally private scratch text that exists nowhere else in this transcript"


def make_turns(assistant_leak: str = "") -> list[Turn]:
    turns = [
        Turn(turn=1, role="user", text=USER_TURN_1),
    ]
    if assistant_leak:
        turns.append(Turn(turn=2, role="assistant", text=assistant_leak))
    turns.append(Turn(turn=3, role="user", text=USER_TURN_2))
    return turns


def active_fixture(settings: Settings) -> list[tuple[str, MemoryLine]]:
    return [
        (
            "tools.md",
            MemoryLine(
                id="mem-1",
                fact="The user prefers Rust for backend work",
                session="mem-sess1",
                turn=2,
                quote="rust for all backend services",
            ),
        ),
        (
            "prefs.md",
            MemoryLine(
                id="mem-2",
                fact="The user works late on Tuesdays",
                session="mem-sess1",
                turn=3,
                quote="i tend to work late tuesdays",
            ),
        ),
    ]


def op_add(**kwargs) -> ProposedOp:
    base = dict(op="add", file="prefs.md", text="The user drinks ginger tea in winter",
                quote=QUOTABLE_1, turn=1, confidence=0.9)
    base.update(kwargs)
    return ProposedOp(**base)


def op_supersede(**kwargs) -> ProposedOp:
    base = dict(op="supersede", file="tools.md",
                text="The user prefers Go for backend work",
                quote=QUOTABLE_2, turn=3, supersedes="mem-1", confidence=0.9)
    base.update(kwargs)
    return ProposedOp(**base)


def op_noop() -> ProposedOp:
    return ProposedOp(op="noop", file="", text="", quote="", turn=1, confidence=1.0)


@pytest.fixture()
def verifier_env(settings_factory):
    s = settings_factory()
    turns = make_turns()
    active = active_fixture(s)
    return Verifier(s, turns=turns, active_entries=active), s, turns, active


def test_accepts_valid_add(verifier_env):
    verifier, s, turns, active = verifier_env
    verdict = verifier.verify_op(op_add())
    assert verdict.status == STATUS_ACCEPTED_ADD
    assert verdict.corrected_turn == 1
    assert verdict.path == "prefs.md"


def test_rejects_unknown_quote(verifier_env):
    verifier, s, turns, active = verifier_env
    verdict = verifier.verify_op(op_add(quote="this phrase appears in no turn anywhere"))
    assert verdict.status == STATUS_REJECTED
    assert verdict.reason.startswith("quote_not_found")


def test_assistant_turns_never_evidence(settings_factory):
    s = settings_factory()
    turns = make_turns(assistant_leak=ASSISTANT_ONLY)
    verifier = Verifier(s, turns=turns, active_entries=[])
    verdict = verifier.verify_op(op_add(quote=ASSISTANT_ONLY))
    assert verdict.status == STATUS_REJECTED
    assert verdict.reason.startswith("quote_not_found")


def test_rejects_short_quote(verifier_env):
    verifier, s, turns, active = verifier_env
    verdict = verifier.verify_op(op_add(quote="I drink tea"))
    assert verdict.status == STATUS_REJECTED
    assert verdict.reason.startswith("quote_too_short")
    assert str(s.verify.min_quote_chars) in verdict.reason


def test_autocorrects_turn_index(verifier_env):
    verifier, s, turns, active = verifier_env
    verdict = verifier.verify_op(op_add(quote=QUOTABLE_2, turn=1, text="The user's daily keyboard is an ortholinear"))
    assert verdict.status == STATUS_ACCEPTED_ADD
    assert verdict.corrected_turn == 3


def test_bad_target_when_supersedes_unknown(verifier_env):
    verifier, s, turns, active = verifier_env
    verdict = verifier.verify_op(op_supersede(supersedes="mem-99"))
    assert verdict.status == STATUS_REJECTED
    assert verdict.reason.startswith("bad_target")


def test_supersede_target_found(verifier_env):
    verifier, s, turns, active = verifier_env
    verdict = verifier.verify_op(op_supersede(text="The user prefers Rust for backend workday tasks", quote=QUOTABLE_2, turn=3))
    assert verdict.status == STATUS_ACCEPTED_SUPERSEDE


def test_supersede_excluded_from_dup_check(verifier_env):
    verifier, s, turns, active = verifier_env
    dup_text = "The user prefers Rust for backend tasks"
    base = "The user prefers Rust for backend work"
    assert normalise(dup_text, s) != ""
    import difflib

    ratio = difflib.SequenceMatcher(None, normalise(dup_text, s), normalise(base, s)).ratio()
    assert ratio >= s.verify.dup_similarity
    verdict = verifier.verify_op(op_supersede(text=dup_text))
    assert verdict.status == STATUS_ACCEPTED_SUPERSEDE


def test_duplicate_downgrades_to_noop(verifier_env):
    verifier, s, turns, active = verifier_env
    dup = op_add(text="The user works late on Tuesdays mostly", quote=QUOTABLE_1, turn=1)
    verdict = verifier.verify_op(dup)
    assert verdict.status == STATUS_NOOP
    assert verdict.reason == "duplicate_of:mem-2"


def test_bad_path_rejects(verifier_env):
    verifier, s, turns, active = verifier_env
    for bad_file in ("x.txt", "../evil.md", "a/b/c/d.md"):
        verdict = verifier.verify_op(op_add(file=bad_file))
        assert verdict.status == STATUS_REJECTED
        assert verdict.reason.startswith("bad_path")


def test_noop_passes_through(verifier_env):
    verifier, s, turns, active = verifier_env
    verdict = verifier.verify_op(op_noop())
    assert verdict.status == STATUS_NOOP


def test_ops_overflow(settings_factory):
    s = settings_factory("memory.max_ops_per_session=1")
    turns = make_turns()
    verifier = Verifier(s, turns=turns, active_entries=[])
    first = verifier.verify_op(op_add())
    second = verifier.verify_op(
        ProposedOp(op="add", file="prefs.md", text="The user drinks ginger tea in winter mornings too",
                   quote=QUOTABLE_2, turn=3, confidence=0.9)
    )
    assert first.status == STATUS_ACCEPTED_ADD
    assert second.status == STATUS_REJECTED
    assert second.reason.startswith("ops_overflow")


def test_normalise_rules_apply(settings_factory):
    s = settings_factory()
    assert normalise("  Mixed   CASE!!  ", s) == "mixed case"
    assert "GINGER" != normalise("GINGER", s) or normalise("GINGER", s) == "ginger"
