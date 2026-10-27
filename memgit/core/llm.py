"""Config-driven LLM client: model discovery, chat, JSON validation loop.

Every provider-specific detail (base url, endpoint paths, auth header shape,
model names, timeouts, retry/budget numbers) lives in ``settings.llm``.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from memgit.config import Settings

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_PLACEHOLDER_RE_G = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_MONEY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?i:bb|\bb)")
_MILLIONS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?i:mm|\bm)")
_PLAIN_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_SMALL_MARKERS = ("mini", "small", "tiny", "nano", "lite")
_BIG_MARKERS = ("large", "max", "ultra", "405")


class LLMError(RuntimeError):
    pass


class CallBudgetExceeded(LLMError):
    pass


class BudgetCounter:
    """Shared per-run call budget; ``llm.max_calls`` comes from config."""

    def __init__(self) -> None:
        self._used = 0

    @property
    def used(self) -> int:
        return self._used

    def try_call(self, max_calls: int) -> None:
        if self._used >= max_calls:
            raise CallBudgetExceeded(
                f"process LLM call budget exhausted: used {self._used} >= llm.max_calls={max_calls}; "
                "raise [llm].max_calls in memgit.toml if this run is expected to be long"
            )
        self._used += 1


_default_budget = BudgetCounter()


def default_budget() -> BudgetCounter:
    return _default_budget


@dataclass
class ChatReply:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class UsageTotals:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, reply: ChatReply) -> None:
        self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens

    def add_failure(self) -> None:
        self.calls += 1
        self.failures += 1


def render_text(text: str, **placeholders: str) -> str:
    """Fill ``{name}`` placeholders; unknown placeholder names fail loudly."""
    missing: List[str] = []

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in placeholders:
            missing.append(key)
            return match.group(0)
        return str(placeholders[key])

    rendered = _PLACEHOLDER_RE_G.sub(_sub, text)
    if missing:
        raise LLMError(f"template placeholders missing: {', '.join(sorted(set(missing)))}")
    return rendered


def api_key(settings: Settings) -> str:
    name = settings.llm.api_key_env
    value = os.environ.get(name, "")
    if not value.strip():
        raise LLMError(
            f"LLM key missing: environment variable {name!r} is not set. "
            f"Copy .env.example to .env, set {name}=..., or export {name} in your shell."
        )
    return value


def _truncate(text: str, limit: int = 400) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + " [...]"


def extract_json(text: str) -> Optional[str]:
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first = text.find(open_ch)
        last = text.rfind(close_ch)
        if first != -1 and last > first:
            return text[first: last + 1]
    return None


def size_estimate(model_id: str) -> int:
    """Documented, config-independent heuristic: bigger parameter counts -> bigger int.

    Used only to order discovered models when llm.model is unset; the regexes are
    generic size tokens, not provider names.
    """
    score = 1
    for match in _MONEY_RE.finditer(model_id):
        score = max(score, int(float(match.group(1)) * 1_000_000_000) or 1)
    for match in _MILLIONS_RE.finditer(model_id):
        score = max(score, int(float(match.group(1)) * 1_000_000) or 1)
    if not _MONEY_RE.search(model_id) and not _MILLIONS_RE.search(model_id):
        for match in _PLAIN_NUM_RE.finditer(model_id):
            score = max(score, int(float(match.group(1)) * 1_000_000) or 1)
    lowered = model_id.lower()
    if any(marker in lowered for marker in _SMALL_MARKERS):
        score = max(1, score // 100)
    if any(marker in lowered for marker in _BIG_MARKERS):
        score *= 4
    return score


def pick_model(models: Sequence[str], settings: Settings, role: str) -> str:
    """Resolve the model for `role` ("consolidation" or "judge").

    Precedence: configured llm.model / llm.judge_model; then llm.model_preference_patterns
    regex matches (in config order); then the generic size heuristic
    (smallest-looking model for consolidation, largest-looking for judge); then first model.
    """
    explicit = settings.llm.model if role == "consolidation" else settings.llm.judge_model
    if explicit.strip():
        return explicit.strip()
    if not models:
        raise LLMError(
            "model discovery returned no models; set llm.model and llm.judge_model explicitly in memgit.toml"
        )
    patterns = list(settings.llm.model_preference_patterns)
    matched = [m for m in models if any(re.search(p, m) for p in patterns)]
    pool = matched or list(models)
    key = (lambda m: size_estimate(m)) if role == "judge" else (lambda m: -size_estimate(m))
    return max(pool, key=key)


class UsageLog:
    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.logging.usage_log_path)

    def record(
        self,
        purpose: str,
        model: str,
        latency_s: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: str = "",
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "model": model,
            "latency_s": round(latency_s, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "status": "error" if error else "ok",
            "error": error,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise LLMError(f"cannot append usage log {self.path}: {exc}") from exc


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        budget: Optional[BudgetCounter] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.cfg = settings
        self.usage = UsageTotals()
        self.budget = budget if budget is not None else default_budget()
        self._http = http_client or httpx.Client(timeout=settings.llm.timeout_s)
        self._headers: Dict[str, str] = {
            settings.llm.auth_header_name: settings.llm.auth_header_template.format(key=api_key(settings)),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._usage_log = UsageLog(settings)
        self._models: Optional[List[str]] = None

    @property
    def base_url(self) -> str:
        base = self.cfg.llm.base_url.strip().rstrip("/")
        if not base:
            raise LLMError(
                "llm.base_url is empty; set [llm].base_url in memgit.toml "
                "(a full https://host(v1) OpenAI-compatible root)"
            )
        return base

    def discover_models(self, force: bool = False) -> List[str]:
        if self._models is not None and not force:
            return self._models
        url = self.base_url + self.cfg.llm.models_endpoint
        started = time.monotonic()
        try:
            response = self._http.get(url, headers=self._headers)
        except httpx.HTTPError as exc:
            self._log_failure("discover_models", started, str(exc))
            raise LLMError(
                f"model discovery failed: {exc.__class__.__name__}: {exc}; check llm.base_url "
                f"({self.cfg.llm.base_url!r}) and network reachability"
            ) from exc
        latency = time.monotonic() - started
        if response.status_code >= 400:
            self._log_failure("discover_models", latency, f"HTTP {response.status_code}")
            raise LLMError(
                f"model discovery got HTTP {response.status_code} from {url}: {_truncate(response.text)}; "
                "check llm.base_url (no trailing /models in it) and llm.api_key_env"
            )
        self._log_reply("discover_models", "-", latency, 0, 0, "")
        self._models = _parse_models_payload(response.json(), url)
        return self._models

    def resolve_model(self, role: str, force: bool = False) -> str:
        return pick_model(self.discover_models(force=force), self.cfg, role)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        purpose: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatReply:
        if model is None:
            model = self.resolve_model("consolidation")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": self.cfg.llm.temperature if temperature is None else temperature,
        }
        url = self.base_url + self.cfg.llm.chat_endpoint
        started = time.monotonic()
        retries = self.cfg.llm.max_retries
        last_error = ""
        for attempt in range(retries + 1):
            try:
                self.budget.try_call(self.cfg.llm.max_calls)
                response = self._http.post(url, headers=self._headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                if attempt < retries:
                    _sleep_backoff(self.cfg.llm.backoff_s, attempt)
                    continue
                self._log_failure(purpose, time.monotonic() - started, last_error)
                raise LLMError(
                    f"chat transport failed after {attempt + 1} attempt(s) ({last_error}); "
                    "check llm.base_url / llm.timeout_s"
                ) from exc
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {_truncate(response.text)}"
                if attempt < retries:
                    delay = _parse_retry_delay(response)
                    if delay is not None and delay > 0:
                        time.sleep(delay)
                    else:
                        _sleep_backoff(self.cfg.llm.backoff_s, attempt)
                    continue
                self._log_failure(purpose, time.monotonic() - started, last_error)
                raise LLMError(
                    f"chat failed after {attempt + 1} attempt(s) ({last_error}); "
                    "adjust llm.max_retries / llm.backoff_s, or the endpoint may be overloaded"
                )
            latency = time.monotonic() - started
            if response.status_code >= 400:
                self._log_failure(purpose, latency, f"HTTP {response.status_code}")
                raise LLMError(
                    f"chat got HTTP {response.status_code} from {url}: {_truncate(response.text)}"
                )
            try:
                body = response.json()
            except ValueError as exc:
                self._log_failure(purpose, latency, "response is not JSON")
                raise LLMError(
                    f"chat response from {url} is not JSON: {_truncate(response.text)}"
                ) from exc
            try:
                content = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {}) or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                self._log_failure(purpose, latency, "unexpected response shape")
                raise LLMError(
                    f"unexpected chat response shape from {url}: {_truncate(response.text)}"
                ) from exc
            if not isinstance(content, str):
                self._log_failure(purpose, latency, "response content is not a string")
                raise LLMError(f"chat response content is not a string: {_truncate(response.text)}")
            self._log_reply(purpose, model, latency, prompt_tokens, completion_tokens, "")
            reply = ChatReply(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=latency,
            )
            self.usage.add(reply)
            return reply
        raise LLMError(f"chat failed after retries: {last_error}")  # pragma: no cover - defensive

    def chat_stream(
        self,
        messages: Sequence[Dict[str, str]],
        purpose: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream chat completion chunks via server-sent events."""
        if model is None:
            model = self.resolve_model("consolidation")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": self.cfg.llm.temperature if temperature is None else temperature,
            "stream": True,
        }
        url = self.base_url + self.cfg.llm.chat_endpoint
        started = time.monotonic()
        self.budget.try_call(self.cfg.llm.max_calls)
        accumulated: List[str] = []
        try:
            with self._http.stream("POST", url, headers=self._headers, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMError(f"chat stream got HTTP {response.status_code} from {url}: {_truncate(response.text)}")
                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated.append(content)
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except (httpx.HTTPError, LLMError) as exc:
            self._log_failure(purpose, time.monotonic() - started, str(exc))
            raise LLMError(f"chat stream failed: {exc}") from exc
        full_text = "".join(accumulated)
        latency = time.monotonic() - started
        prompt_est = sum(len(m.get("content", "").split()) for m in messages)
        comp_est = len(full_text.split())
        self._log_reply(purpose, model, latency, prompt_est, comp_est, "")
        reply = ChatReply(
            content=full_text,
            model=model,
            prompt_tokens=prompt_est,
            completion_tokens=comp_est,
            latency_s=latency,
        )
        self.usage.add(reply)

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        schema: Type[T],
        purpose: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[T, bool]:
        """Call chat, extract the JSON object, validate with pydantic.

        On validation failure, appends the validation error and asks the model to fix
        it using prompts/fix_json.md. Returns (instance, used_retry).
        """
        convo = [dict(m) for m in messages]
        attempts = self.cfg.llm.json_retries + 1
        used_retry = False
        last_output = ""
        for attempt in range(attempts):
            reply = self.chat(convo, purpose=purpose, model=model, temperature=temperature)
            raw = reply.content
            candidate = extract_json(raw)
            parse_failure = ""
            data: Any = None
            if candidate is None:
                parse_failure = "no JSON object found in the reply"
            else:
                try:
                    data = json.loads(candidate)
                except ValueError as exc:
                    parse_failure = f"JSON parse error: {exc}"
            if parse_failure == "":
                try:
                    return schema.model_validate(data), used_retry
                except ValidationError as exc:
                    parse_failure = _validate_error_text(exc, schema)
            last_output = raw
            used_retry = True
            if attempt >= attempts - 1:
                raise LLMError(
                    f"LLM reply for purpose={purpose!r} failed JSON validation after {attempts} "
                    f"attempt(s): {parse_failure}"
                )
            convo = convo + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": fix_json_prompt(self.cfg, last=last_output, error=parse_failure)},
            ]
        raise LLMError("unreachable: chat_json retry loop exited")  # pragma: no cover

    def totals(self) -> Dict[str, int]:
        return {
            "calls": self.usage.calls,
            "failures": self.usage.failures,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
        }

    def _log_failure(self, purpose: str, latency_s: float, error: str) -> None:
        self._usage_log.record(purpose=purpose, model="-", latency_s=latency_s, error=error)
        self.usage.add_failure()

    def _log_reply(
        self, purpose: str, model: str, latency_s: float, prompt_tokens: int, completion_tokens: int, error: str
    ) -> None:
        self._usage_log.record(
            purpose=purpose,
            model=model,
            latency_s=latency_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
        )


def _sleep_backoff(base_s: float, attempt: int) -> None:
    if base_s > 0:
        time.sleep(base_s * (2 ** attempt))


def _parse_retry_delay(response: httpx.Response) -> Optional[float]:
    if response.status_code != 429:
        return None
    hdr = response.headers.get("retry-after")
    if hdr:
        try:
            return max(0.5, float(hdr))
        except ValueError:
            pass
    m = re.search(r"try again in (\d+(?:\.\d+)?)s", response.text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) + 0.5
        except ValueError:
            pass
    return None



def _parse_models_payload(payload: Any, url: str) -> List[str]:
    ids: List[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        items = payload["data"]
        for item in items:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
    if not ids:
        raise LLMError(
            f"model discovery response from {url} has no model ids (shape: {_truncate(json.dumps(payload, default=str))}); "
            "if this provider differs from OpenAI conventions, adjust llm.models_endpoint or llm.base_url"
        )
    seen: List[str] = []
    for model_id in ids:
        if model_id not in seen:
            seen.append(model_id)
    return seen


def _validate_error_text(exc: ValidationError, schema: Type[BaseModel]) -> str:
    parts = []
    for err in exc.errors()[:6]:
        loc = ".".join(str(item) for item in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']} (type={err['type']})")
    return f"expects: {schema.__name__}; issues: " + "; ".join(parts)


def fix_json_prompt(settings: Settings, last: str, error: str) -> str:
    loaded = LoadPrompt(settings, settings.prompts.fix_json)
    return render_text(loaded.text, last=last, error=error)


class LoadPrompt:
    """Loads a prompt file from the configured prompts dir with its {placeholders}."""

    _PLACEHOLDER_RE = _PLACEHOLDER_RE_G

    def __init__(self, settings: Settings, filename: str) -> None:
        from memgit.config import prompts_dir

        self.path = prompts_dir(settings) / filename
        try:
            self.text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMError(f"prompt file missing: {self.path} ({exc}); check [memory].prompts_dir") from exc

    def render(self, **placeholders: str) -> str:
        return render_text(self.text, **placeholders)
