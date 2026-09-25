"""Local, read-only dashboard for the transformer experiment sweep.

Serves a single static page plus a small JSON API over Python's standard
library `http.server`. Deliberately has no new third-party dependency:
`pyproject.toml` lists none of a web framework, and this dashboard doesn't
need one to poll a handful of JSON files on disk.

This script never writes to `results/`. It only reads
`results/logs/transformer_sweep.json`, `results/logs/<model>/<fold>.json`,
and `results/metrics/<model>/<fold>.json`.

Usage:
    uv run python scripts/run_dashboard.py [--results-dir results] [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboard import state as dashboard_state  # noqa: E402

INDEX_HTML_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "static" / "index.html"


class StateStore:
    """Holds the last-good dashboard state, refreshed on a timer.

    Reading the manifest can transiently fail while the sweep process is
    mid-write; `dashboard_state.build_dashboard_state` already handles that
    by returning the previous state with a `manifest_stale` flag, so this
    store just needs to keep calling it on an interval and hold the result
    behind a lock for the HTTP handler to read.
    """

    def __init__(self, results_dir: Path, poll_interval: float):
        self._results_dir = results_dir
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._state: dict = dashboard_state.build_dashboard_state(results_dir)
        self._stop = threading.Event()

    def get(self) -> dict:
        with self._lock:
            return self._state

    def refresh_once(self) -> None:
        with self._lock:
            previous = self._state
        new_state = dashboard_state.build_dashboard_state(self._results_dir, previous_state=previous)
        with self._lock:
            self._state = new_state

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self._poll_interval)

    def stop(self) -> None:
        self._stop.set()

    def model_detail(self, model: str) -> dict | None:
        state = self.get()
        manifest = {"models": state.get("models", []), "folds": state.get("folds", [])}
        if model not in manifest["models"]:
            return None
        return dashboard_state.build_model_detail(self._results_dir, manifest, model)

    def run_detail(self, model: str, fold: str) -> dict:
        return dashboard_state.build_run_detail(self._results_dir, model, fold)


def make_handler(store: StateStore, poll_interval: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter default logging
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (stdlib method name)
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                try:
                    body = INDEX_HTML_PATH.read_bytes()
                except FileNotFoundError:
                    self.send_error(500, "Dashboard UI file missing")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/state":
                state = store.get()
                state = dict(state)
                state["poll_interval_seconds"] = poll_interval
                self._send_json(state)
                return

            if path == "/api/model":
                qs = parse_qs(parsed.query)
                model = (qs.get("model") or [None])[0]
                if not model:
                    self._send_json({"error": "missing 'model' query param"}, status=400)
                    return
                detail = store.model_detail(model)
                if detail is None:
                    self._send_json({"error": f"unknown model {model!r}"}, status=404)
                    return
                self._send_json(detail)
                return

            if path == "/api/run":
                qs = parse_qs(parsed.query)
                model = (qs.get("model") or [None])[0]
                fold = (qs.get("fold") or [None])[0]
                if not model or not fold:
                    self._send_json({"error": "missing 'model'/'fold' query params"}, status=400)
                    return
                self._send_json(store.run_detail(model, fold))
                return

            self.send_error(404, "Not found")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to the results/ directory to read (read-only). Default: ./results",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between manifest re-reads (default: 5.0)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    store = StateStore(results_dir, args.poll_interval)

    refresh_thread = threading.Thread(target=store.run_forever, daemon=True)
    refresh_thread.start()

    handler_cls = make_handler(store, args.poll_interval)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)

    print(f"Transformer sweep dashboard (read-only) serving on http://127.0.0.1:{args.port}")
    print(f"Reading results from: {results_dir}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        store.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
