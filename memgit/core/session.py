"""Session orchestration: transcript -> consolidate -> verify -> apply atomically -> git commit."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from memgit.config import Settings
from memgit.core.consolidate import Consolidator, ProposedOp, Turn
from memgit.core.llm import LLMClient, LLMError, LoadPrompt, render_text
from memgit.core.store import MemoryStore
from memgit.core.schema import MemoryLine
from memgit.core.verify import STATUS_REJECTED, Verifier

logger = logging.getLogger("memgit")


def parse_sections(markdown: str) -> dict:
    sections: dict = {}
    current: Optional[str] = None
    buffer: List[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buffer).strip("\n")
            current = line[3:].strip()
            buffer = []
            continue
        if current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip("\n")
    return sections


def render_commit_message(
    settings: Settings,
    session_id: str,
    added: List[MemoryLine],
    superseded: List[Tuple[MemoryLine, str, MemoryLine]],
    rejected: List[Tuple[ProposedOp, str]],
    noop_count: int,
) -> str:
    sections = parse_sections(LoadPrompt(settings, settings.prompts.commit_message).text)
    required = ("subject", "count_format", "added_line", "superseded_line", "rejected_line", "empty_body", "body")
    missing = [key for key in required if key not in sections]
    if missing:
        raise LLMError(f"commit_message prompt missing sections: {', '.join(missing)}")
    counts = render_text(
        sections["count_format"],
        added=str(len(added)),
        superseded=str(len(superseded)),
        rejected=str(len(rejected)),
        noop=str(noop_count),
    )
    subject = render_text(sections["subject"], session_id=session_id, counts=counts)
    lines: List[str] = []
    for line in added:
        lines.append(
            render_text(
                sections["added_line"],
                id=line.id,
                fact=line.fact,
                turn=str(line.turn),
                session=line.session,
            )
        )
    for old_line, new_id, new_line in superseded:
        lines.append(
            render_text(
                sections["superseded_line"],
                new_id=new_id,
                fact=new_line.fact,
                old_id=old_line.id,
                old_fact=old_line.fact,
                turn=str(new_line.turn),
            )
        )
    for op, reason in rejected:
        lines.append(render_text(sections["rejected_line"], reason=reason))
    operations = "\n".join(lines) if lines else render_text(sections["empty_body"])
    body = render_text(sections["body"], operations=operations)
    return f"{subject}\n\n{body}"


def append_rejections(settings: Settings, session_id: str, rejected: List[Tuple[ProposedOp, str]]) -> None:
    if not rejected:
        return
    path = Path(settings.logging.rejections_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    payload = [
        {
            "ts": ts,
            "session_id": session_id,
            "op": op.model_dump(),
            "reason": reason,
        }
        for op, reason in rejected
    ]
    with path.open("a", encoding="utf-8") as fh:
        for record in payload:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class SessionOutcome:
    ok: bool
    session_id: str
    commit_sha: Optional[str]
    message: str
    added: List[MemoryLine] = field(default_factory=list)
    superseded: List[Tuple[MemoryLine, str, MemoryLine]] = field(default_factory=list)
    rejected: List[Tuple[ProposedOp, str]] = field(default_factory=list)
    noop_count: int = 0
    used_retry: bool = False
    model: str = ""
    error: str = ""


class MemorySession:
    def __init__(self, settings: Settings, store: MemoryStore, client: LLMClient) -> None:
        self.cfg = settings
        self.store = store
        self.client = client
        self.consolidator = Consolidator(settings, client)

    def run(self, transcript: List[Turn]) -> SessionOutcome:
        session_id = self.store.begin_session()
        active_before = self.store.active_entries()
        try:
            result = self.consolidator.propose(
                transcript=transcript,
                active_entries=active_before,
                next_id_example=self.store.next_id_example(),
            )
        except LLMError as exc:
            logger.warning("session not consolidated: %s", exc)
            self.store.finish_session()
            return SessionOutcome(
                ok=False,
                session_id=session_id,
                commit_sha=None,
                message=f"session not consolidated: {exc}",
                error=str(exc),
            )
        verifier = Verifier(self.cfg, turns=transcript, active_entries=active_before)
        summary = verifier.verify_all(result.ops)
        rejected = [
            (verdict.op, verdict.reason)
            for verdict in summary.verdicts
            if verdict.status == STATUS_REJECTED
        ]
        added, superseded = self.store.apply(summary.verdicts)
        noop_count = sum(1 for v in summary.verdicts if v.op.op == "noop")
        message = render_commit_message(
            self.cfg,
            session_id=session_id,
            added=added,
            superseded=superseded,
            rejected=rejected,
            noop_count=noop_count,
        )
        sha = self.store.commit_all(message)
        self.store.finish_session()
        append_rejections(self.cfg, session_id, rejected)
        return SessionOutcome(
            ok=True,
            session_id=session_id,
            commit_sha=sha,
            message=message,
            added=added,
            superseded=superseded,
            rejected=rejected,
            noop_count=noop_count,
            used_retry=result.used_retry,
            model=result.model,
        )
