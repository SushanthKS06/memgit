"""Transcript -> proposed memory ops, driven by prompt files from the prompts dir."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from memgit.config import Settings
from memgit.core.llm import LLMClient, LLMError, LoadPrompt
from memgit.core.schema import MemoryLine, format_line


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
Role = Literal["user", "assistant"]


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    role: Role
    text: str



class ProposedOp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["add", "supersede", "noop"]
    file: str
    text: str
    quote: str
    turn: int
    supersedes: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class ConsolidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ops: List[ProposedOp]


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    turns: List[Turn]
    known_files: List[str]
    output: ConsolidationOutput


class FewShotFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examples: List[FewShotExample]


def render_transcript(turns: List[Turn], settings: Settings) -> str:
    labels = {ROLE_USER: "USER", ROLE_ASSISTANT: "ASSISTANT"}
    return "\n".join(f"[t{turn.turn}] {labels[turn.role]}: {turn.text}" for turn in turns)


def render_memories(entries: List[Tuple[str, MemoryLine]], settings: Settings) -> str:
    if not entries:
        return "(no memories yet)"
    grouped: Dict[str, List[str]] = {}
    for rel, line in entries:
        grouped.setdefault(rel, []).append(format_line(line, settings))
    blocks = [f"### {path}\n" + "\n".join(lines) for path, lines in grouped.items()]
    return "\n\n".join(blocks)


def render_few_shots(settings: Settings) -> str:
    loader = LoadPrompt(settings, settings.prompts.few_shots)
    try:
        payload = json.loads(loader.text)
    except ValueError as exc:
        raise LLMError(f"few-shot file {loader.path} is not valid JSON: {exc}") from exc
    few = FewShotFile.model_validate(payload)
    blocks: List[str] = []
    for example in few.examples:
        known = ", ".join(example.known_files) if example.known_files else "(none)"
        blocks.append(
            f"### example: {example.name}\n"
            f"known memory files: {known}\n"
            "input transcript:\n"
            f"{render_transcript(example.turns, settings)}\n"
            "expected output:\n"
            f"```json\n{json.dumps(example.output.model_dump(), ensure_ascii=False, indent=2)}\n```"
        )
    return "\n\n".join(blocks)


@dataclass
class ConsolidationResult:
    ops: List[ProposedOp]
    used_retry: bool
    model: str


class Consolidator:
    """Turns a transcript into proposed ops via the consolidate prompt file."""

    def __init__(self, settings: Settings, client: LLMClient) -> None:
        self.cfg = settings
        self.client = client
        self._model: Optional[str] = None

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.client.resolve_model("consolidation")
        return self._model

    def propose(
        self,
        transcript: List[Turn],
        active_entries: List[Tuple[str, MemoryLine]],
        next_id_example: str,
    ) -> ConsolidationResult:
        mem = self.cfg.memory
        ver = self.cfg.verify
        prompt = LoadPrompt(self.cfg, self.cfg.prompts.consolidate).render(
            min_quote_chars=str(ver.min_quote_chars),
            max_ops=str(mem.max_ops_per_session),
            max_path_depth=str(mem.max_path_depth),
            ext=", ".join(mem.allowed_extensions),
            links_mark=mem.superseded_by_prefix,
            id_example=next_id_example,
            existing_memories=render_memories(active_entries, self.cfg),
            turn_count=str(len(transcript)),
            transcript=render_transcript(transcript, self.cfg),
            few_shots=render_few_shots(self.cfg),
        )
        messages = [{"role": "user", "content": prompt}]
        output, used_retry = self.client.chat_json(
            messages,
            schema=ConsolidationOutput,
            purpose="consolidate",
            model=self.model,
            temperature=self.cfg.llm.temperature,
        )
        return ConsolidationResult(ops=output.ops, used_retry=used_retry, model=self.model)
