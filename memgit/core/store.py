"""Git-backed memory store. Phase 0: layout, snapshots, applying verified ops, index.

Git commit/log/diff/revert arrive with the session layer (Phase 1/2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from memgit.config import Settings
from memgit.core.schema import (
    Counters,
    MemoryFileContent,
    MemoryLine,
    ParseError,
    ensure_layout,
    existing_memory_ids,
    format_memory_file,
    init_counters,
    iter_memory_files,
    load_counters,
    next_memory_id,
    next_session_id,
    rebuild_index_text,
    store_counters,
    write_index,
)


class StoreError(RuntimeError):
    pass


class MemoryStore:
    def __init__(self, settings: Settings, path: Optional[Path] = None) -> None:
        self.cfg = settings
        self.root: Path = Path(path) if path is not None else Path(settings.memory.dir)
        ensure_layout(self.root, settings)
        self.counters: Counters = self._load_or_init_counters()
        self.session_id: Optional[str] = None

    def _load_or_init_counters(self) -> Counters:
        try:
            return load_counters(self.root, self.cfg)
        except FileNotFoundError:
            return init_counters(self.root, self.cfg)

    def begin_session(self) -> str:
        if self.session_id is not None:
            raise StoreError("session already started; finish it before beginning a new one")
        self.session_id = next_session_id(self.counters, self.cfg)
        self._persist_counters()
        return self.session_id

    def active_entries(self) -> List[Tuple[str, MemoryLine]]:
        entries: List[Tuple[str, MemoryLine]] = []
        for rel, content in iter_memory_files(self.root, self.cfg):
            for line in content.active:
                entries.append((rel, line))
        return entries

    def file_texts(self) -> Dict[str, str]:
        texts: Dict[str, str] = {}
        for rel, _ in iter_memory_files(self.root, self.cfg):
            path = self.root / rel
            with path.open("r", encoding="utf-8") as fh:
                texts[rel] = fh.read()
        return texts

    def next_id_example(self) -> str:
        return f"{self.cfg.memory.id_prefix}{self.counters.memory_id + 1}"

    def apply(
        self,
        verdicts,  # list[Verdict]
    ) -> Tuple[List[MemoryLine], List[Tuple[MemoryLine, str]]]:
        """Apply accepted add/supersede verdicts atomically in memory, then write.

        Returns (added_lines, superseded_pairs(old_line, new_id)).
        """
        from memgit.core.verify import STATUS_ACCEPTED_ADD, STATUS_ACCEPTED_SUPERSEDE

        if self.session_id is None:
            raise StoreError("no session started; call begin_session() before apply()")
        files: Dict[str, MemoryFileContent] = {rel: content for rel, content in iter_memory_files(self.root, self.cfg)}
        added: List[MemoryLine] = []
        superseded: List[Tuple[MemoryLine, str]] = []
        for verdict in verdicts:
            if verdict.status == STATUS_ACCEPTED_ADD:
                new_id = next_memory_id(self.counters, self.cfg)
                line = MemoryLine(
                    id=new_id,
                    fact=verdict.op.text.strip(),
                    session=self.session_id,
                    turn=verdict.corrected_turn or verdict.op.turn,
                    quote=verdict.op.quote,
                    superseded_by=None,
                )
                content = self._resolve_content(files, verdict.path)
                content.active.append(line)
                added.append(line)
            elif verdict.status == STATUS_ACCEPTED_SUPERSEDE:
                new_id = next_memory_id(self.counters, self.cfg)
                if verdict.target is None:
                    raise StoreError("supersede verdict carries no target")
                target_rel, old_line = verdict.target
                content_old = files.get(target_rel)
                if content_old is None or old_line is None or old_line.id != verdict.op.supersedes:
                    raise StoreError(
                        f"supersede target vanished between verify and apply: {verdict.op.supersedes!r}"
                    )
                content_old.active.remove(old_line)
                superseded_line = old_line.model_copy(update={"superseded_by": new_id})
                content_old.superseded.append(superseded_line)
                new_line = MemoryLine(
                    id=new_id,
                    fact=verdict.op.text.strip(),
                    session=self.session_id,
                    turn=verdict.corrected_turn or verdict.op.turn,
                    quote=verdict.op.quote,
                    superseded_by=None,
                )
                content_new = self._resolve_content(files, verdict.path)
                content_new.active.append(new_line)
                added.append(new_line)
                superseded.append((old_line, new_id))
        for rel, content in files.items():
            self._write_file(rel, content)
        self._rebuild_index()
        self._persist_counters()
        return added, superseded

    def _resolve_content(self, files: Dict[str, MemoryFileContent], rel: str) -> MemoryFileContent:
        rel = rel.replace("\\", "/")
        content = files.get(rel)
        if content is not None:
            return content
        content = MemoryFileContent(header=rel, active=[], superseded=[])
        files[rel] = content
        return content

    def _write_file(self, rel: str, content: MemoryFileContent) -> None:
        path = self.root / rel
        if len(path.relative_to(self.root).parts) > self.cfg.memory.max_path_depth:
            raise StoreError(f"path exceeds max_path_depth: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(format_memory_file(content, self.cfg))

    def _rebuild_index(self) -> None:
        write_index(self.root, self.cfg, rebuild_index_text(self.root, self.cfg))

    def _persist_counters(self) -> None:
        store_counters(self.root, self.cfg, self.counters)

    def memory_ids(self) -> List[str]:
        return existing_memory_ids(self.root, self.cfg)
