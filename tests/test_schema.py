from pathlib import Path

import pytest

from memgit.config import Settings
from memgit.core.schema import (
    Counters,
    MemoryFileContent,
    MemoryLine,
    ParseError,
    PathError,
    check_rel_path,
    format_line,
    format_memory_file,
    init_counters,
    load_counters,
    next_memory_id,
    next_session_id,
    parse_line,
    parse_memory_file,
    rebuild_index_text,
    store_counters,
)


@pytest.fixture()
def default_settings(settings_factory) -> Settings:
    return settings_factory()


@pytest.fixture()
def custom_settings(settings_factory, monkeypatch) -> Settings:
    monkeypatch.setenv("MEMGIT_MEMORY__FIELD_SEP", " ;; ")
    return settings_factory(
        "memory.bullet=*",
        "memory.id_open=<",
        "memory.id_close=>",
        "memory.id_prefix=mm",
        "memory.quote_char='",
        "memory.superseded_by_prefix=replaced:",
        "memory.active_heading=current",
        "memory.superseded_heading=past",
        "memory.index_line_template=* {header} @ {path}",
        'memory.allowed_extensions=[".note"]',
        "memory.max_path_depth=2",
    )


def test_round_trip_default(default_settings):
    line = MemoryLine(
        id="mem-1",
        fact="The user prefers dark mode everywhere",
        session="mem-sess1",
        turn=4,
        quote="I want dark mode everywhere I can get it",
    )
    sup_line = MemoryLine(
        id="mem-2",
        fact="The user preferred light mode at work",
        session="mem-sess1",
        turn=5,
        quote="light mode at work is better",
        superseded_by="mem-3",
    )
    text = format_memory_file(
        MemoryFileContent(header="prefs", active=[line], superseded=[sup_line]),
        default_settings,
    )
    assert "## Active" in text and "## Superseded" in text
    parsed = parse_memory_file(text, default_settings)
    assert parsed.header == "prefs"
    assert parsed.active == [line]
    assert parsed.superseded == [sup_line]
    assert format_memory_file(parsed, default_settings) == text


def test_round_trip_non_default(custom_settings):
    active = MemoryLine(
        id="mm1",
        fact="The user runs tests daily",
        session="mm-sessA",
        turn=3,
        quote="I run tests every single morning",
    )
    superseded = MemoryLine(
        id="mm2",
        fact="The user ran tests with unittest",
        session="mm-sessB",
        turn=4,
        quote="unittest for now at least",
        superseded_by="mm7",
    )
    text = format_memory_file(
        MemoryFileContent(header="tools-notes", active=[active], superseded=[superseded]),
        custom_settings,
    )
    assert text.startswith("# tools-notes\n\ncurrent\n")
    assert "past" in text
    parsed = parse_memory_file(text, custom_settings)
    assert parsed.header == "tools-notes"
    assert parsed.active == [active]
    assert parsed.superseded == [superseded]
    assert format_memory_file(parsed, custom_settings) == text


def test_parse_handwritten_non_default_ahead(custom_settings):
    handwritten = "\n".join(
        [
            "# hand-written",
            "",
            "current",
            "",
            "* <mm5> The user brews coffee at home ;; s:mm-sess ;; t:2 ;; 'I brew my coffee at home'",
            "",
            "past",
            "",
            "* <mm4> The user bought takeaway coffee daily ;; s:mm-sess ;; t:1 ;; 'I buy takeaway coffee' ;; replaced:mm5",
        ]
    ) + "\n"
    parsed = parse_memory_file(handwritten, custom_settings)
    assert parsed.header == "hand-written"
    assert parsed.active[0].id == "mm5"
    assert parsed.active[0].quote == "I brew my coffee at home"
    assert parsed.superseded[0].superseded_by == "mm5"


def test_parse_rejections(default_settings):
    with pytest.raises(ParseError):
        parse_line("- [mem-7]  | s:a | t:1 | \"quote\"", default_settings)
    with pytest.raises(ParseError):
        parse_line("- [xx-1] fact | s:a | t:1 | \"quote\"", default_settings)
    with pytest.raises(ParseError):
        parse_line("- [mem-1] fields missing", default_settings)
    with pytest.raises(ParseError):
        parse_line("no bullet [mem-1] fact | s:a | t:1 | \"quote\"", default_settings)


def test_check_rel_path_rules(custom_settings):
    assert check_rel_path("sub/pref.note", custom_settings) == "sub/pref.note"
    assert check_rel_path(" My Notes.note ", custom_settings) == "mynotes.note"
    with pytest.raises(PathError):
        check_rel_path("a/b/c.note", custom_settings)
    with pytest.raises(PathError):
        check_rel_path("x.txt", custom_settings)
    with pytest.raises(PathError):
        check_rel_path("../escape.note", custom_settings)
    with pytest.raises(PathError):
        check_rel_path("", custom_settings)


def test_index_rebuild_from_headers(settings_factory, tmp_path: Path):
    custom = settings_factory("memory.index_line_template=* {header} @ {path}", 'memory.allowed_extensions=[".note"]')
    (tmp_path / "alpha.note").write_text("# alpha memo\n\n## Active\n\n## Superseded\n", encoding="utf-8")
    (tmp_path / "beta.note").write_text("# beta memo\n\n## Active\n\n## Superseded\n", encoding="utf-8")
    index = rebuild_index_text(tmp_path, custom)
    assert index == "* alpha memo @ alpha.note\n* beta memo @ beta.note\n"


def test_counters_round_trip(settings_factory, tmp_path: Path):
    custom = settings_factory("memory.id_prefix=mm")
    counters = init_counters(tmp_path, custom)
    assert counters.memory_id == 0
    assert next_memory_id(counters, custom) == "mm1"
    next_session_id(counters, custom)
    store_counters(tmp_path, custom, counters)
    reloaded = load_counters(tmp_path, custom)
    assert reloaded == counters


def test_counters_resume_from_files(settings_factory, tmp_path: Path):
    custom = settings_factory()
    (tmp_path / "main.md").write_text(
        "# tools\n\n## Active\n\n- [mem-9] The user keeps notes in git | s:mem-sess1 | t:1 | \"notes in git\"\n\n## Superseded\n",
        encoding="utf-8",
    )
    counters = init_counters(tmp_path, custom)
    assert counters.memory_id == 9
    assert next_memory_id(counters, custom) == "mem-10"
