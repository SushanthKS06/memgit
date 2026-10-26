"""Recall: pick relevant files (LLM selection, grep fallback), active lines, token-truncated block."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from memgit.config import Settings
from memgit.core.llm import LLMClient, LLMError, LoadPrompt
from memgit.core.schema import MemoryLine, format_line, iter_memory_files, rebuild_index_text


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: List[str]


def count_tokens(text: str, settings: Settings) -> int:
    method = settings.memory.token_count_method
    if method == "words":
        return len(text.split())
    if method == "chars4":
        return max(1, -(-len(text) // 4))
    raise LLMError(f"unknown memory.token_count_method: {method!r}")


@dataclass
class RecallResult:
    files: List[str]
    lines: List[Tuple[str, MemoryLine]]
    block: str
    tokens: int = 0
    used_llm: bool = False
    used_grep: bool = False
    truncated: bool = False


class Recaller:
    def __init__(self, settings: Settings, client: LLMClient) -> None:
        self.cfg = settings
        self.client = client
        self._model: Optional[str] = None

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.client.resolve_model("consolidation")
        return self._model

    def _index_text(self) -> str:
        path = Path(self.cfg.memory.dir) / self.cfg.memory.index_file
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return rebuild_index_text(Path(self.cfg.memory.dir), self.cfg)

    def recall(self, query: str) -> RecallResult:
        files = self._select_files(query)
        lines = self._active_lines(files)
        block = self._render_block(lines)
        tokens = count_tokens(block, self.cfg)
        return RecallResult(
            files=files,
            lines=lines,
            block=block,
            tokens=tokens,
            used_llm=self._used_llm,
            used_grep=self._used_grep,
            truncated=self._truncated,
        )

    def _select_files(self, query: str) -> List[str]:
        available = [rel for rel, _ in iter_memory_files(Path(self.cfg.memory.dir), self.cfg)]
        budget = self.cfg.memory.max_recall_files
        self._used_llm = False
        self._used_grep = False
        picks: List[str] = []
        if available:
            try:
                llm_picks = self._llm_select(query, available)
                self._used_llm = True
                picks = [rel for rel in llm_picks if rel in available]
            except LLMError as exc:
                if not self.cfg.recall.fallback_grep:
                    raise
                picks = [rel for rel in self._grep_select(query, available)]
                self._used_grep = True
        if not picks and not self._used_grep and self.cfg.recall.fallback_grep and available:
            picks = self._grep_select(query, available)
            self._used_grep = True
        deduped: List[str] = []
        for rel in picks:
            if rel not in deduped:
                deduped.append(rel)
        return deduped[:budget]

    def _llm_select(self, query: str, available: List[str]) -> List[str]:
        prompt = LoadPrompt(self.cfg, self.cfg.prompts.recall_select).render(
            max_files=str(self.cfg.memory.max_recall_files),
            index=self._index_text(),
            query=query,
        )
        output, _ = self.client.chat_json(
            [{"role": "user", "content": prompt}],
            schema=Selection,
            purpose="recall_select",
            model=self.model,
            temperature=self.cfg.llm.temperature,
        )
        return output.files

    def _grep_select(self, query: str, available: List[str]) -> List[str]:
        tokens = [
            tok.lower()
            for tok in query.split()
            if len(tok) >= self.cfg.recall.grep_min_token
        ]
        if not tokens:
            return []
        root = Path(self.cfg.memory.dir)
        hits: List[str] = []
        for rel in available:
            with (root / rel).open("r", encoding="utf-8") as fh:
                raw = fh.read().lower()
            if any(tok in raw for tok in tokens):
                hits.append(rel)
        return hits

    def _active_lines(self, files: List[str]) -> List[Tuple[str, MemoryLine]]:
        budget = self.cfg.memory.max_recall_tokens
        self._truncated = False
        by_rel = {rel: content for rel, content in iter_memory_files(Path(self.cfg.memory.dir), self.cfg)}
        used = 0
        limit_inserted = False
        out: List[Tuple[str, MemoryLine]] = []
        for rel in files:
            content = by_rel.get(rel)
            if content is None:
                continue
            for line in content.active:
                line_tokens = count_tokens(format_line(line, self.cfg), self.cfg)
                if used + line_tokens > budget:
                    self._truncated = True
                    limit_inserted = True
                    break
                out.append((rel, line))
                used += line_tokens
            if limit_inserted:
                break
        return out

    def _render_block(self, lines: List[Tuple[str, MemoryLine]]) -> str:
        mem = self.cfg.memory
        if lines:
            body = "\n".join(f"({rel}) {format_line(entry, self.cfg)}" for rel, entry in lines)
        else:
            body = mem.empty_memory_note
        return LoadPrompt(self.cfg, self.cfg.prompts.recall_block).render(
            memories=body,
            links_mark=mem.superseded_by_prefix,
        )
