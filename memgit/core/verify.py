"""Deterministic verification of proposed ops. No LLM involved; all thresholds from config."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from memgit.config import Settings
from memgit.core.consolidate import ROLE_ASSISTANT, ROLE_USER, ProposedOp, Turn
from memgit.core.schema import MemoryLine, PathError, check_rel_path

STATUS_ACCEPTED_ADD = "accepted_add"
STATUS_ACCEPTED_SUPERSEDE = "accepted_supersede"
STATUS_NOOP = "noop"
STATUS_REJECTED = "rejected"


def normalise(text: str, settings: Settings) -> str:
    out = text.strip()
    for rule in settings.verify.normalise_rules:
        if rule == "lowercase":
            out = out.lower()
        elif rule == "collapse_ws":
            out = " ".join(out.split())
        elif rule == "strip_edge_punct":
            out = out.strip(" \t.,;:!?\"'`()[]{}-")
    return out


@dataclass
class Verdict:
    op: ProposedOp
    status: str
    reason: str = ""
    path: str = ""
    corrected_turn: int = 0
    target: Optional[Tuple[str, MemoryLine]] = None


@dataclass
class VerificationSummary:
    verdicts: List[Verdict] = field(default_factory=list)

    def counts(self) -> Tuple[int, int, int]:
        accepted = sum(1 for v in self.verdicts if v.status in (STATUS_ACCEPTED_ADD, STATUS_ACCEPTED_SUPERSEDE))
        rejected = sum(1 for v in self.verdicts if v.status == STATUS_REJECTED)
        noops = sum(1 for v in self.verdicts if v.status == STATUS_NOOP)
        return accepted, rejected, noops


class Verifier:
    """Sequential per-op verification; session-level op quota enforced here."""

    def __init__(
        self,
        settings: Settings,
        turns: Sequence[Turn],
        active_entries: Sequence[Tuple[str, MemoryLine]],
    ) -> None:
        self.cfg = settings
        self.turns = list(turns)
        self.user_turns = [(t.turn, t.text) for t in self.turns if t.role == ROLE_USER]
        self.active_entries = list(active_entries)
        self.active_by_id = {line.id: (rel, line) for rel, line in self.active_entries}
        self._accepted_count = 0
        self._corrected_turn = 0

    def verify_all(self, ops: Sequence[ProposedOp]) -> VerificationSummary:
        summary = VerificationSummary()
        assert self._accepted_count == 0, "Verifier instance must be used for one session only"
        for op in ops:
            summary.verdicts.append(self.verify_op(op))
        return summary

    def verify_op(self, op: ProposedOp) -> Verdict:
        if op.op == "noop":
            return Verdict(op=op, status=STATUS_NOOP)
        try:
            path = check_rel_path(op.file, self.cfg)
        except PathError as exc:
            return Verdict(op=op, status=STATUS_REJECTED, reason=f"bad_path: {exc}")
        quote_fail = self._check_quote(op)
        if quote_fail is not None:
            return Verdict(op=op, status=STATUS_REJECTED, reason=quote_fail, path=path)
        if op.op == "supersede":
            target = self.active_by_id.get(op.supersedes or "")
            if target is None:
                return Verdict(
                    op=op,
                    status=STATUS_REJECTED,
                    reason=f"bad_target: {op.supersedes!r} is not an active memory id",
                    path=path,
                )
            dup_id = self._duplicate_check(op.text, exclude=target[1].id)
        else:
            target = None
            dup_id = self._duplicate_check(op.text)
        if dup_id is not None:
            return Verdict(
                op=op,
                status=STATUS_NOOP,
                reason=f"duplicate_of:{dup_id}",
                path=path,
            )
        if self._accepted_count >= self.cfg.memory.max_ops_per_session:
            return Verdict(
                op=op,
                status=STATUS_REJECTED,
                reason=(
                    f"ops_overflow: session already has {self._accepted_count} accepted ops, "
                    f"memory.max_ops_per_session={self.cfg.memory.max_ops_per_session}"
                ),
                path=path,
            )
        self._accepted_count += 1
        status = STATUS_ACCEPTED_ADD if op.op == "add" else STATUS_ACCEPTED_SUPERSEDE
        return Verdict(
            op=op,
            status=status,
            path=path,
            corrected_turn=self._corrected_turn,
            target=target,
        )

    def _check_quote(self, op: ProposedOp) -> Optional[str]:
        if len(op.quote) < self.cfg.verify.min_quote_chars:
            return (
                f"quote_too_short: quote is {len(op.quote)} chars, "
                f"verify.min_quote_chars={self.cfg.verify.min_quote_chars}"
            )
        norm_quote = normalise(op.quote, self.cfg)
        if not norm_quote:
            return "quote_not_found: quote is empty after normalisation"
        claimed = [(idx, text) for idx, text in self.user_turns if idx == op.turn]
        for idx, text in claimed:
            if norm_quote in normalise(text, self.cfg):
                self._corrected_turn = idx
                return None
        for idx, text in self.user_turns:
            if norm_quote in normalise(text, self.cfg):
                self._corrected_turn = idx
                return None
        # Multi-turn grounding: allow assistant quote if immediately affirmed by the user
        if self.cfg.verify.allow_assistant_quotes:
            turns_by_num = {t.turn: t for t in self.turns}
            for turn_idx, turn in turns_by_num.items():
                if turn.role == ROLE_ASSISTANT and norm_quote in normalise(turn.text, self.cfg):
                    next_turn = turns_by_num.get(turn_idx + 1)
                    if next_turn and next_turn.role == ROLE_USER:
                        norm_user = normalise(next_turn.text, self.cfg)
                        # Negation check: if any negation word appears in the user turn,
                        # reject this grounding (even if an affirmation is also present).
                        negations = self.cfg.verify.negations
                        if negations:
                            for neg in negations:
                                norm_neg = normalise(neg, self.cfg)
                                if norm_neg and re.search(rf"\b{re.escape(norm_neg)}\b", norm_user):
                                    break
                            else:
                                # No negation found; check affirmations
                                for aff in self.cfg.verify.affirmations:
                                    norm_aff = normalise(aff, self.cfg)
                                    if not norm_aff:
                                        continue
                                    if re.search(rf"\b{re.escape(norm_aff)}\b", norm_user):
                                        self._corrected_turn = next_turn.turn
                                        return None
                        else:
                            for aff in self.cfg.verify.affirmations:
                                norm_aff = normalise(aff, self.cfg)
                                if not norm_aff:
                                    continue
                                if re.search(rf"\b{re.escape(norm_aff)}\b", norm_user):
                                    self._corrected_turn = next_turn.turn
                                    return None
        return "quote_not_found: quote is not a verbatim substring of any user turn"

    def _duplicate_check(self, text: str, exclude: Optional[str] = None) -> Optional[str]:
        norm_text = normalise(text, self.cfg)
        if not norm_text:
            return None
        for rel, line in self.active_entries:
            if exclude is not None and line.id == exclude:
                continue
            norm_fact = normalise(line.fact, self.cfg)
            ratio = difflib.SequenceMatcher(None, norm_text, norm_fact).ratio()
            if ratio >= self.cfg.verify.dup_similarity:
                return line.id
        return None
