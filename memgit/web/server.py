"""Stackless web UI for memgit: stdlib http.server, server-rendered HTML.

Pages:
  GET  /            history (paginated) with one-click revert buttons
  GET  /commit/<sha>  commit message + patch + revert button
  GET  /memory        current memory files and their active entries
  GET  /rejections    audit log of rejected consolidation ops
  POST /revert        perform the revert for a submitted commit

Presentation chrome (labels/css) is code-local; every data value comes from
memory/config state. [web] host/port/page_size drive the endpoint; port 0
binds a free ephemeral port (used by tests).
"""
from __future__ import annotations

import html
import json
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from memgit.config import Settings
from memgit.core.store import CommitInfo, MemoryStore, StoreError

logger = logging.getLogger("memgit.web")

STYLE = (
    "body{font-family:ui-monospace,Consolas,monospace;margin:1.5rem;color:#111}"
    "nav a{margin-right:1rem}"
    "table{border-collapse:collapse;width:100%}"
    "td,th{border:1px solid #ccc;padding:0.25rem 0.5rem;text-align:left;font-size:0.9rem}"
    "pre{background:#f6f6f6;padding:0.75rem;overflow-x:auto;white-space:pre-wrap}"
    "button{font:inherit}"
    ".flash{background:#fdecea;border:1px solid #e05252;padding:0.5rem 0.75rem}"
)

PAGE_OK = "page"
STORE_MISSING = "store"
REVERT = "revert"


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)} — memgit</title>"
        f"<style>{STYLE}</style></head><body>"
        "<nav><a href='/'>history</a><a href='/memory'>memory</a>"
        "<a href='/rejections'>rejections</a></nav>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>"
    )


class WebApp:
    """Read-model + revert actions over one MemoryStore; formatted to HTML."""

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings

    def store(self) -> MemoryStore:
        return MemoryStore(self.cfg)

    def commits(self, limit: int) -> List[CommitInfo]:
        return self.store().log(limit)

    def commit_view(self, commit: str) -> Tuple[CommitInfo, str]:
        store = self.store()
        matching = [c for c in store.log(limit=50_000) if c.sha.startswith(commit)]
        if len(matching) != 1:
            raise LookupError(f"no unique commit for {commit!r} ({len(matching)} matches)")
        info = matching[0]
        return info, store.diff_against(info.sha)

    def revert(self, commit: str) -> str:
        return self.store().revert(commit)

    def memory_overview(self) -> List[Tuple[str, int, int, List[Any]]]:
        from memgit.core.schema import iter_memory_files

        rows: List[Tuple[str, int, int, List[Any]]] = []
        for rel, content in iter_memory_files(Path(self.cfg.memory.dir), self.cfg):
            rows.append((rel, len(content.active), len(content.superseded), list(content.active)))
        return rows

    def rejections(self) -> List[Dict[str, Any]]:
        path = Path(self.cfg.logging.rejections_log_path)
        if not path.is_file():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("skipping malformed rejections row: %s", exc)
        return rows[-self.cfg.web.rejections_rows :]


def render_history(app: WebApp, page_number: int) -> str:
    page_size = app.cfg.web.page_size
    start = max(0, page_number - 1) * page_size
    fetched = app.commits(start + page_size + 1)
    commits = fetched[start : start + page_size]
    rows = []
    for commit in commits:
        rows.append(
            "<tr>"
            f"<td>{html.escape(commit.date)}</td>"
            f"<td><a href='/commit/{html.escape(commit.short_sha)}'>{html.escape(commit.short_sha)}</a></td>"
            f"<td>{html.escape(commit.subject)}</td>"
            "<td>"
            f"<form method='post' action='/revert'>"
            f"<input type='hidden' name='commit' value='{html.escape(commit.sha)}'>"
            "<button type='submit'>revert</button>"
            "</form>"
            "</td>"
            "</tr>"
        )
    table = f"<table><tr><th>date</th><th>sha</th><th>subject</th><th></th></tr>{''.join(rows)}</table>"
    has_more = len(fetched) >= start + page_size + 1
    links = []
    if page_number > 1:
        links.append(f"<a href='/?page={page_number - 1}'>&larr; newer</a>")
    if has_more:
        links.append(f"<a href='/?page={page_number + 1}'>older &rarr;</a>")
    nav = " ".join(links) if links else "(single page)"
    return page("history", f"{table}<p>{nav}</p>")


def render_commit(app: WebApp, commit: str) -> str:
    info, diff = app.commit_view(commit)
    body = (
        "<pre>" + html.escape(info.full_message()) + "</pre>"
        "<h2>patch</h2><pre>" + html.escape(diff) + "</pre>"
        f"<form method='post' action='/revert'>"
        f"<input type='hidden' name='commit' value='{html.escape(info.sha)}'>"
        "<button type='submit'>revert this session</button>"
        "</form>"
    )
    return page(f"commit {info.short_sha}", body)


def render_memory(app: WebApp) -> str:
    sections = []
    for rel, active_n, superseded_n, active in app.memory_overview():
        bullets = "".join(
            f"<li><code>{html.escape(line.id)}</code> {html.escape(line.fact)} "
            f"<small>(s:{html.escape(line.session)} t:{line.turn})</small></li>"
            for line in active
        )
        sections.append(
            f"<h2>{html.escape(rel)}</h2>"
            f"<p>{active_n} active, {superseded_n} superseded</p><ul>{bullets}</ul>"
        )
    if not sections:
        sections.append("<p>(no memory files yet)</p>")
    return page("memory", "".join(sections))


def render_rejections(app: WebApp) -> str:
    rows = app.rejections()
    if not rows:
        return page("rejections", "<p>(no rejections logged yet)</p>")
    trs = []
    for row in rows:
        op = row.get("op", {})
        trs.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('ts', '')))}</td>"
            f"<td>{html.escape(str(row.get('session_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('reason', '')))}</td>"
            f"<td><code>{html.escape(json.dumps(op, ensure_ascii=False))}</code></td>"
            "</tr>"
        )
    note = "" if len(rows) < app.cfg.web.rejections_rows else f"<p>(capped at {app.cfg.web.rejections_rows} rows)</p>"
    return page(
        "rejections",
        f"<small>{note}</small><table><tr><th>ts</th><th>session</th><th>reason</th><th>op</th></tr>{''.join(trs)}</table>",
    )


def render_error(message: str) -> Tuple[int, str]:
    return 409, page("error", f"<p class='flash'>{html.escape(message)}</p><p><a href='/'>&larr; history</a></p>")


def not_found(path: str) -> Tuple[int, str]:
    return 404, page("not found", f"<p class='flash'>no route: {html.escape(path)}</p><p><a href='/'>&larr; history</a></p>")


def make_handler(app: WebApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "memgit-web"

        def log_message(self, fmt: str, *args: Any) -> None:  # default noise -> stderr logging
            logger.debug(fmt, *args)

        def _send_html(self, status: int, markup: str) -> None:
            payload = markup.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                query = parse_qs(parsed.query)
                page_number = 1
                if "page" in query:
                    raw = query["page"][0]
                    if raw.isdigit() and int(raw) >= 1:
                        page_number = int(raw)
                self._send_html(200, render_history(app, page_number))
            elif path == "/memory":
                self._send_html(200, render_memory(app))
            elif path == "/rejections":
                self._send_html(200, render_rejections(app))
            elif path.startswith("/commit/"):
                commit = path[len("/commit/") :].strip("/")
                try:
                    self._send_html(200, render_commit(app, commit))
                except LookupError as exc:
                    status, markup = render_error(str(exc))
                    self._send_html(status, markup)
            else:
                status, markup = not_found(path)
                self._send_html(status, markup)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/revert":
                status, markup = not_found(parsed.path)
                self._send_html(status, markup)
                return
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            commit = (form.get("commit") or [""])[0]
            if not commit:
                status, markup = render_error("no commit given; nothing to revert")
                self._send_html(status, markup)
                return
            try:
                app.revert(commit)
            except StoreError as exc:
                status, markup = render_error(str(exc))
                self._send_html(status, markup)
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    return Handler


def build_server(settings: Settings) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((settings.web.host, settings.web.port), make_handler(WebApp(settings)))


def serve(settings: Settings) -> None:
    server = build_server(settings)
    host, port = server.server_address[:2]
    display_host = host if host else "localhost"
    logger.info("serving memgit web UI on http://%s:%d", display_host, port)
    runner = threading.current_thread()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down (port %d)", port)
    finally:
        if runner is threading.main_thread():
            pass
        server.server_close()
