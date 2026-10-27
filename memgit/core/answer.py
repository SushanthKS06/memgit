"""Answering turn of the chat loop: configured answer prompt + memory block + transcript."""
from __future__ import annotations

from typing import Iterator, List

from memgit.config import Settings
from memgit.core.consolidate import Turn, render_transcript
from memgit.core.llm import LLMClient, LoadPrompt


class Answerer:
    def __init__(self, settings: Settings, client: LLMClient) -> None:
        self.cfg = settings
        self.client = client
        self._model = None

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self.client.resolve_model("consolidation")
        return self._model

    def answer(self, transcript: List[Turn], memory_block: str) -> str:
        prompt = LoadPrompt(self.cfg, self.cfg.prompts.answer).render(
            memory_block=memory_block,
            transcript=render_transcript(transcript, self.cfg),
        )
        reply = self.client.chat(
            [{"role": "user", "content": prompt}],
            purpose="answer",
            model=self.model,
            temperature=self.cfg.llm.temperature,
        )
        return reply.content

    def answer_stream(self, transcript: List[Turn], memory_block: str) -> Iterator[str]:
        prompt = LoadPrompt(self.cfg, self.cfg.prompts.answer).render(
            memory_block=memory_block,
            transcript=render_transcript(transcript, self.cfg),
        )
        return self.client.chat_stream(
            [{"role": "user", "content": prompt}],
            purpose="answer",
            model=self.model,
            temperature=self.cfg.llm.temperature,
        )
