"""Deterministic memory-file format: parse, serialise, index, counters, path rules.

Everything format-related is read from ``settings.memory``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from memgit.config import Settings

JSONDict = Dict[str, Any]


class MemoryLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    fact: str
    session: str
    turn: int
    quote: str
    superseded_by: Optional[str] = None


class MemoryFileContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header: str
    active: List[MemoryLine]
    superseded: List[MemoryLine]


class ParseError(ValueError):
    pass


class PathError(ValueError):
    pass


def sanitize_slug(segment: str, settings: Settings) -> str:
    slug = segment.strip().lower()
    return re.sub(settings.memory.slug_strip_pattern, "", slug)


def check_rel_path(path_str: str, settings: Settings) -> str:
    cleaned = path_str.strip().replace("\\", "/")
    if not cleaned:
        raise PathError("empty path")
    parts = cleaned.split("/")
    if any(part in ("..", ".") for part in parts):
        raise PathError(f"path must not contain '.' or '..' segments: {path_str!r}")
    depth_limit = settings.memory.max_path_depth
    if len(parts) > depth_limit:
        raise PathError(
            f"path depth {len(parts)} exceeds memory.max_path_depth={depth_limit}: {path_str!r}"
        )
    sanitized: List[str] = []
    for part in parts:
        slugged = sanitize_slug(part, settings)
        if not slugged:
            raise PathError(f"path segment is empty after slugging: {path_str!r}")
        if slugged.startswith("."):
            raise PathError(f"path segment must not start with '.': {path_str!r}")
        sanitized.append(slugged)
    result = "/".join(sanitized)
    exts = [ext.lower() for ext in settings.memory.allowed_extensions]
    if Path(result).suffix.lower() not in exts:
        raise PathError(
            f"extension {Path(result).suffix!r} not in memory.allowed_extensions={exts}: {path_str!r}"
        )
    return result


def parse_line(line: str, settings: Settings) -> MemoryLine:
    mem = settings.memory
    stripped = line.strip()
    bullet_tok = f"{mem.bullet} "
    if not stripped.startswith(bullet_tok):
        raise ParseError(f"line does not start with bullet {bullet_tok!r}: {line!r}")
    content = stripped[len(bullet_tok):]
    start = content.find(mem.id_open)
    end_id = content.find(mem.id_close)
    if start != 0 or end_id <= len(mem.id_open):
        raise ParseError(f"line lacks memory id wrapper: {line!r}")
    mem_id = content[start + len(mem.id_open):end_id]
    if not mem_id.startswith(mem.id_prefix) or not mem_id[len(mem.id_prefix):].isdigit():
        raise ParseError(f"memory id must be {mem.id_prefix!r}+digits, got {mem_id!r}")
    rest = content[end_id + len(mem.id_close):].lstrip()
    fields = rest.split(mem.field_sep)
    if len(fields) not in (4, 5):
        raise ParseError(
            f"expected 4 or 5 fields separated by {mem.field_sep!r}, got {len(fields)}: {line!r}"
        )
    fact = fields[0].strip()
    if not fact:
        raise ParseError(f"fact text is empty: {line!r}")
    session = _strip_prefix(fields[1].strip(), mem.source_prefix, line)
    turn_txt = _strip_prefix(fields[2].strip(), mem.turn_prefix, line)
    if not turn_txt.isdigit():
        raise ParseError(f"turn field is not a number: {line!r}")
    quote = _unquote(fields[3].strip(), settings)
    superseded_by: Optional[str] = None
    if len(fields) == 5:
        sup_raw = fields[4].strip()
        if not sup_raw.startswith(mem.superseded_by_prefix):
            raise ParseError(f"5th field must start with {mem.superseded_by_prefix!r}: {line!r}")
        superseded_by = sup_raw[len(mem.superseded_by_prefix):].strip() or None
    return MemoryLine(
        id=mem_id,
        fact=fact,
        session=session,
        turn=int(turn_txt),
        quote=quote,
        superseded_by=superseded_by,
    )


def _strip_prefix(text: str, prefix: str, line: str) -> str:
    if not text.startswith(prefix):
        raise ParseError(f"field does not start with {prefix!r}: {line!r}")
    return text[len(prefix):].strip()


def _unquote(text: str, settings: Settings) -> str:
    quote = settings.memory.quote_char
    if len(text) >= 2 and text.startswith(quote) and text.endswith(quote):
        return text[1:-1]
    return text


def format_line(entry: MemoryLine, settings: Settings) -> str:
    mem = settings.memory
    cols = mem.field_sep.join(
        [
            f"{mem.source_prefix}{entry.session}",
            f"{mem.turn_prefix}{entry.turn}",
            f"{mem.quote_char}{entry.quote}{mem.quote_char}",
        ]
    )
    if entry.superseded_by:
        cols = mem.field_sep.join([cols, f"{mem.superseded_by_prefix}{entry.superseded_by}"])
    return f"{mem.bullet} {mem.id_open}{entry.id}{mem.id_close} {entry.fact}{mem.field_sep}{cols}"


def parse_memory_file(text: str, settings: Settings, source: str = "<memory>") -> MemoryFileContent:
    mem = settings.memory
    header = ""
    section: Optional[str] = None
    active: List[MemoryLine] = []
    superseded: List[MemoryLine] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue
        header_tok = f"{mem.header_prefix} "
        if raw.startswith(header_tok) and section is None:
            header = raw[len(header_tok):].strip()
            continue
        if raw.strip() == mem.active_heading.strip():
            section = "active"
            continue
        if raw.strip() == mem.superseded_heading.strip():
            section = "superseded"
            continue
        if section == "active":
            try:
                active.append(parse_line(raw, settings))
            except ParseError as exc:
                raise ParseError(f"{source}:{lineno}: {exc}") from exc
            continue
        if section == "superseded":
            try:
                superseded.append(parse_line(raw, settings))
            except ParseError as exc:
                raise ParseError(f"{source}:{lineno}: {exc}") from exc
            continue
        raise ParseError(f"{source}:{lineno}: line outside of known sections: {raw!r}")
    if not header:
        raise ParseError(f"{source}: header line not found")
    return MemoryFileContent(header=header, active=active, superseded=superseded)


def format_memory_file(content: MemoryFileContent, settings: Settings) -> str:
    mem = settings.memory
    parts = [f"{mem.header_prefix} {content.header}", "", mem.active_heading, ""]
    parts.extend(format_line(e, settings) for e in content.active)
    parts.extend(["", mem.superseded_heading, ""])
    parts.extend(format_line(e, settings) for e in content.superseded)
    return "\n".join(parts).rstrip() + "\n"


def iter_memory_files(
    memory_path: Path, settings: Settings
) -> Iterator[Tuple[str, MemoryFileContent]]:
    root = Path(memory_path)
    if not root.is_dir():
        return
    exts = [e.lower() for e in settings.memory.allowed_extensions]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        if path.name == settings.memory.index_file:
            continue
        if len(path.relative_to(root).parts) > settings.memory.max_path_depth:
            continue
        if path.parent.name == settings.memory.state_dir:
            continue
        rel = path.relative_to(root).as_posix()
        with path.open("r", encoding="utf-8") as fh:
            text = fh.read()
        yield rel, parse_memory_file(text, settings, source=rel)


def ensure_layout(memory_path: Path, settings: Settings) -> None:
    (Path(memory_path) / settings.memory.state_dir).mkdir(parents=True, exist_ok=True)


def rebuild_index_text(memory_path: Path, settings: Settings) -> str:
    template = settings.memory.index_line_template
    rows = [template.format(header=content.header, path=rel) for rel, content in iter_memory_files(memory_path, settings)]
    return "\n".join(rows).rstrip() + "\n" if rows else ""


def write_index(memory_path: Path, settings: Settings, index_text: str) -> None:
    ensure_layout(memory_path, settings)
    with (Path(memory_path) / settings.memory.index_file).open("w", encoding="utf-8") as fh:
        fh.write(index_text)


class Counters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: int
    session_seq: int


def counters_path(memory_path: Path, settings: Settings) -> Path:
    return Path(memory_path) / settings.memory.state_dir / settings.memory.counter_file


def load_counters(memory_path, settings: Settings) -> Counters:
    path = counters_path(memory_path, settings)
    if not path.is_file():
        raise FileNotFoundError(f"counters missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return Counters.model_validate(json.load(fh))


def store_counters(memory_path: Path, settings: Settings, counters: Counters) -> None:
    ensure_layout(memory_path, settings)
    with counters_path(memory_path, settings).open("w", encoding="utf-8") as fh:
        json.dump(counters.model_dump(), fh, indent=2)
        fh.write("\n")


def existing_memory_ids(memory_path: Path, settings: Settings) -> List[str]:
    prefix = settings.memory.id_prefix
    seen: List[str] = []
    for _, content in iter_memory_files(memory_path, settings):
        for entry in content.active + content.superseded:
            if entry.id.startswith(prefix) and entry.id[len(prefix):].isdigit():
                seen.append(entry.id)
    return seen


def init_counters(memory_path: Path, settings: Settings) -> Counters:
    prefix = settings.memory.id_prefix
    numeric_ids: List[int] = []
    for mem_id in existing_memory_ids(memory_path, settings):
        suffix = mem_id[len(prefix):]
        if suffix.isdigit():
            numeric_ids.append(int(suffix))
    base = Counters(
        memory_id=max(numeric_ids, default=0),
        session_seq=0,
    )
    store_counters(memory_path, settings, base)
    return base


def next_memory_id(counters: Counters, settings: Settings) -> str:
    counters.memory_id += 1
    return f"{settings.memory.id_prefix}{counters.memory_id}"


def next_session_id(counters: Counters, settings: Settings) -> str:
    counters.session_seq += 1
    return f"{settings.memory.id_prefix}sess{counters.session_seq}"


def snapshot_active_lines(memory_path: Path, settings: Settings) -> List[MemoryLine]:
    result: List[MemoryLine] = []
    for _, content in iter_memory_files(memory_path, settings):
        result.extend(content.active)
    return result
