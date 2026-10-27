"""Modern, sleek web UI and REST API for memgit: stdlib http.server, server-rendered HTML.

Pages:
  GET  /              history (paginated) with diff links and one-click revert buttons
  GET  /commit/<sha>  commit message + syntax-colored patch + revert button
  GET  /memory        current memory files, search filter, and active/superseded entries
  GET  /rejections    audit log of rejected consolidation ops with search filter
  POST /revert        perform the revert for a submitted commit

REST API:
  GET  /api/history       list commits as JSON
  GET  /api/commit/<sha>  commit details and patch as JSON
  GET  /api/memories      all memory files and lines as JSON
  GET  /api/rejections    rejection log events as JSON
  GET  /api/stats         aggregate metrics as JSON
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

STYLE = """
:root {
  --bg: #0b0f19;
  --surface: #111827;
  --card: #1f293d;
  --card-hover: #243048;
  --border: rgba(255, 255, 255, 0.08);
  --border-focus: rgba(56, 189, 248, 0.5);
  --text: #f8fafc;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --accent: #38bdf8;
  --accent-glow: rgba(56, 189, 248, 0.2);
  --success: #34d399;
  --success-bg: rgba(52, 211, 153, 0.12);
  --danger: #f87171;
  --danger-bg: rgba(248, 113, 113, 0.12);
  --badge-bg: rgba(255, 255, 255, 0.05);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
  padding: 0 1.5rem 3rem 1.5rem;
}
.header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.25rem 0;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  gap: 1rem;
}
.brand-group {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.brand-logo {
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.live-dot {
  width: 8px;
  height: 8px;
  background: var(--success);
  border-radius: 50%;
  display: inline-block;
  box-shadow: 0 0 10px var(--success);
}
.version-tag {
  font-size: 0.75rem;
  background: var(--badge-bg);
  color: var(--accent);
  padding: 0.15rem 0.5rem;
  border-radius: 9999px;
  border: 1px solid var(--border);
}
nav {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
nav a {
  color: var(--text-muted);
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 500;
  padding: 0.4rem 0.875rem;
  border-radius: 6px;
  transition: all 0.15s ease;
  border: 1px solid transparent;
}
nav a:hover {
  color: var(--text);
  background: rgba(255, 255, 255, 0.04);
}
nav a.active {
  color: var(--accent);
  background: var(--accent-glow);
  border-color: rgba(56, 189, 248, 0.3);
}
.main-container {
  max-width: 1100px;
  margin: 0 auto;
}
h1 {
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 1.5rem;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
h2 {
  font-size: 1.2rem;
  font-weight: 600;
  margin: 1.5rem 0 0.75rem 0;
  color: var(--text);
}
.search-box {
  width: 100%;
  padding: 0.65rem 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-size: 0.9rem;
  margin-bottom: 1.25rem;
  outline: none;
  transition: border-color 0.15s ease;
}
.search-box:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  margin-bottom: 1rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  background: var(--surface);
}
th, td {
  padding: 0.75rem 1rem;
  text-align: left;
  font-size: 0.875rem;
  border-bottom: 1px solid var(--border);
}
th {
  background: rgba(255, 255, 255, 0.02);
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
}
tr:last-child td {
  border-bottom: none;
}
tr:hover td {
  background: rgba(255, 255, 255, 0.02);
}
pre {
  background: #080c14;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  overflow-x: auto;
  white-space: pre-wrap;
  font-family: ui-monospace, Consolas, "JetBrains Mono", monospace;
  font-size: 0.875rem;
  line-height: 1.5;
  color: #e2e8f0;
  margin-bottom: 1.25rem;
}
code {
  font-family: ui-monospace, Consolas, monospace;
  font-size: 0.875rem;
  color: var(--accent);
  background: rgba(56, 189, 248, 0.08);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
}
button {
  font-family: inherit;
  font-size: 0.825rem;
  font-weight: 500;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  cursor: pointer;
  background: var(--danger-bg);
  color: var(--danger);
  border: 1px solid rgba(248, 113, 113, 0.3);
  transition: all 0.15s ease;
}
button:hover {
  background: rgba(248, 113, 113, 0.25);
  border-color: var(--danger);
}
.btn-primary {
  background: var(--accent-glow);
  color: var(--accent);
  border-color: rgba(56, 189, 248, 0.3);
}
.btn-primary:hover {
  background: rgba(56, 189, 248, 0.3);
}
.flash {
  background: var(--danger-bg);
  border: 1px solid var(--danger);
  color: var(--danger);
  padding: 0.75rem 1rem;
  border-radius: 8px;
  margin-bottom: 1.25rem;
}
ul {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin: 0.75rem 0;
}
li {
  padding: 0.5rem 0.75rem;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 0.875rem;
}
.diff-viewer {
  padding: 0;
  background: #080c14;
}
.diff-line { display: block; padding: 2px 12px; }
.diff-header { color: #38bdf8; font-weight: 600; background: rgba(56, 189, 248, 0.05); }
.diff-hunk { color: #c084fc; background: rgba(192, 132, 252, 0.08); }
.diff-add { color: #4ade80; background: rgba(74, 222, 128, 0.12); }
.diff-del { color: #f87171; background: rgba(248, 113, 113, 0.12); }
.diff-meta { color: #94a3b8; }
.badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  font-size: 0.75rem;
  font-weight: 600;
  border-radius: 9999px;
  background: var(--badge-bg);
  color: var(--text-muted);
  border: 1px solid var(--border);
}
.badge-active { background: var(--success-bg); color: var(--success); border-color: rgba(52, 211, 153, 0.3); }
.badge-superseded { background: rgba(245, 158, 11, 0.1); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }
.pagination { display: flex; gap: 0.75rem; align-items: center; margin-top: 1rem; }
.pagination a { color: var(--accent); text-decoration: none; font-size: 0.9rem; font-weight: 500; }
.pagination a:hover { text-decoration: underline; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
"""

SCRIPT = """
function filterRows(inputId, tableSelector) {
  const input = document.getElementById(inputId);
  if (!input) return;
  const q = input.value.toLowerCase().trim();
  const rows = document.querySelectorAll(tableSelector + ' tbody tr');
  rows.forEach(r => {
    const text = r.textContent.toLowerCase();
    r.style.display = (!q || text.includes(q)) ? '' : 'none';
  });
}
function filterSections(inputId, sectionSelector) {
  const input = document.getElementById(inputId);
  if (!input) return;
  const q = input.value.toLowerCase().trim();
  const cards = document.querySelectorAll(sectionSelector);
  cards.forEach(c => {
    const text = c.textContent.toLowerCase();
    c.style.display = (!q || text.includes(q)) ? '' : 'none';
  });
}
"""


def page(title: str, body: str, active_tab: str = "") -> str:
    nav_history = " class='active'" if active_tab == "history" else ""
    nav_memory = " class='active'" if active_tab == "memory" else ""
    nav_rejections = " class='active'" if active_tab == "rejections" else ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)} — memgit</title>"
        f"<style>{STYLE}</style>"
        f"<script>{SCRIPT}</script>"
        "</head><body><div class='main-container'>"
        "<div class='header-bar'>"
        "<div class='brand-group'>"
        "<a href='/' class='brand-logo'><span class='live-dot'></span>memgit</a>"
        "<span class='version-tag'>v0.6.0</span>"
        "</div>"
        f"<nav><a href='/'{nav_history}>history</a>"
        f"<a href='/memory'{nav_memory}>memory</a>"
        f"<a href='/rejections'{nav_rejections}>rejections</a></nav>"
        "</div>"
        f"<h1>{html.escape(title)}</h1>{body}</div></body></html>"
    )


class WebApp:
    """Read-model + revert actions over one MemoryStore; formatted to HTML and JSON."""

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


def colorize_diff(diff: str) -> str:
    lines = []
    for line in diff.splitlines():
        esc = html.escape(line)
        if line.startswith("diff --git"):
            lines.append(f"<span class='diff-line diff-header'>{esc}</span>")
        elif line.startswith("@@"):
            lines.append(f"<span class='diff-line diff-hunk'>{esc}</span>")
        elif line.startswith("+") and not line.startswith("+++"):
            lines.append(f"<span class='diff-line diff-add'>{esc}</span>")
        elif line.startswith("-") and not line.startswith("---"):
            lines.append(f"<span class='diff-line diff-del'>{esc}</span>")
        elif line.startswith("+++") or line.startswith("---") or line.startswith("index "):
            lines.append(f"<span class='diff-line diff-meta'>{esc}</span>")
        else:
            lines.append(f"<span class='diff-line'>{esc}</span>")
    return "".join(lines)


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
            f"<td><a href='/commit/{html.escape(commit.short_sha)}'><code>{html.escape(commit.short_sha)}</code></a></td>"
            f"<td>{html.escape(commit.subject)}</td>"
            "<td>"
            f"<form method='post' action='/revert'>"
            f"<input type='hidden' name='commit' value='{html.escape(commit.sha)}'>"
            "<button type='submit'>revert</button>"
            "</form>"
            "</td>"
            "</tr>"
        )
    search_input = (
        "<input type='text' id='historySearch' class='search-box' title='Filter commits by subject or SHA' "
        "oninput=\"filterRows('historySearch', '#historyTable')\">"
    )
    table = (
        f"<table id='historyTable'><thead><tr><th>date</th><th>sha</th><th>subject</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    has_more = len(fetched) >= start + page_size + 1
    links = []
    if page_number > 1:
        links.append(f"<a href='/?page={page_number - 1}'>&larr; newer</a>")
    if has_more:
        links.append(f"<a href='/?page={page_number + 1}'>older &rarr;</a>")
    nav = " ".join(links) if links else "(single page)"
    return page("history", f"{search_input}{table}<p class='pagination'>{nav}</p>", active_tab="history")


def render_commit(app: WebApp, commit: str) -> str:
    info, diff = app.commit_view(commit)
    colored_diff = colorize_diff(diff)
    body = (
        f"<div class='card'><pre>{html.escape(info.full_message())}</pre></div>"
        f"<h2>patch</h2><pre class='diff-viewer'>{colored_diff}</pre>"
        f"<form method='post' action='/revert'>"
        f"<input type='hidden' name='commit' value='{html.escape(info.sha)}'>"
        "<button type='submit'>revert this session</button>"
        "</form>"
    )
    return page(f"commit {info.short_sha}", body, active_tab="history")


def render_memory(app: WebApp) -> str:
    search_input = (
        "<input type='text' id='memorySearch' class='search-box' placeholder='Filter memory cards by fact, id, or file...' "
        "oninput=\"filterSections('memorySearch', '.memory-card')\">"
    )
    sections = []
    for rel, active_n, superseded_n, active in app.memory_overview():
        bullets = "".join(
            f"<li><code>{html.escape(line.id)}</code> {html.escape(line.fact)} "
            f"<small style='color:var(--text-dim)'>(s:{html.escape(line.session)} t:{line.turn})</small></li>"
            for line in active
        )
        sections.append(
            f"<div class='card memory-card'>"
            f"<h2>{html.escape(rel)}</h2>"
            f"<p><span class='badge badge-active'>{active_n} active</span> "
            f"<span class='badge badge-superseded'>{superseded_n} superseded</span></p>"
            f"<ul>{bullets}</ul>"
            f"</div>"
        )
    if not sections:
        sections.append("<div class='card'><p>(no memory files yet)</p></div>")
    return page("memory", f"{search_input}{''.join(sections)}", active_tab="memory")


def render_rejections(app: WebApp) -> str:
    rows = app.rejections()
    if not rows:
        return page("rejections", "<div class='card'><p>(no rejections logged yet)</p></div>", active_tab="rejections")
    trs = []
    for row in rows:
        op = row.get("op", {})
        trs.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('ts', '')))}</td>"
            f"<td><code>{html.escape(str(row.get('session_id', '')))}</code></td>"
            f"<td><span class='badge' style='background:var(--danger-bg);color:var(--danger)'>{html.escape(str(row.get('reason', '')))}</span></td>"
            f"<td><code>{html.escape(json.dumps(op, ensure_ascii=False))}</code></td>"
            "</tr>"
        )
    note = "" if len(rows) < app.cfg.web.rejections_rows else f"<small>(capped at {app.cfg.web.rejections_rows} rows)</small>"
    search_input = (
        "<input type='text' id='rejSearch' class='search-box' placeholder='Filter rejections by reason or session...' "
        "oninput=\"filterRows('rejSearch', '#rejTable')\">"
    )
    table = (
        f"{note}<table id='rejTable'><thead><tr><th>ts</th><th>session</th><th>reason</th><th>op</th></tr></thead>"
        f"<tbody>{''.join(trs)}</tbody></table>"
    )
    return page("rejections", f"{search_input}{table}", active_tab="rejections")


def render_error(message: str) -> Tuple[int, str]:
    return 409, page("error", f"<p class='flash'>{html.escape(message)}</p><p><a href='/'>&larr; history</a></p>")


def not_found(path: str) -> Tuple[int, str]:
    return 404, page("not found", f"<p class='flash'>no route: {html.escape(path)}</p><p><a href='/'>&larr; history</a></p>")


def make_handler(app: WebApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "memgit-web"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug(fmt, *args)

        def _send_html(self, status: int, markup: str) -> None:
            payload = markup.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: int, payload: Any) -> None:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            # REST API endpoints
            if path == "/api/history":
                commits = [
                    {"sha": c.sha, "short_sha": c.short_sha, "author": c.author, "date": c.date, "subject": c.subject}
                    for c in app.commits(app.cfg.git.log_limit)
                ]
                self._send_json(200, commits)
                return
            if path == "/api/memories":
                overview = [
                    {
                        "file": rel,
                        "active_count": act_n,
                        "superseded_count": sup_n,
                        "active": [line.model_dump() for line in active],
                    }
                    for rel, act_n, sup_n, active in app.memory_overview()
                ]
                self._send_json(200, overview)
                return
            if path == "/api/rejections":
                self._send_json(200, app.rejections())
                return
            if path.startswith("/api/commit/"):
                commit = path[len("/api/commit/") :].strip("/")
                try:
                    info, diff = app.commit_view(commit)
                    self._send_json(200, {
                        "sha": info.sha,
                        "short_sha": info.short_sha,
                        "author": info.author,
                        "date": info.date,
                        "subject": info.subject,
                        "body": info.body,
                        "diff": diff,
                    })
                except LookupError as exc:
                    self._send_json(404, {"error": str(exc)})
                return

            # HTML pages
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
