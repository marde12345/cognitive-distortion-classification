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
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboard import analysis_loader  # noqa: E402
from dashboard import state as dashboard_state  # noqa: E402

INDEX_HTML_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "static" / "index.html"


def local_lan_ip() -> str | None:
    """Best-effort discovery of this machine's LAN IP address.

    Opens a UDP socket to a public address without sending any packet
    (UDP ``connect`` just picks a local route) purely to ask the OS which
    local interface/address would be used — a standard, side-effect-free
    trick. Returns None if it fails for any reason (e.g. no network),
    since this is informational only and must never crash startup.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


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
        self._demo_model = None
        self._demo_model_lock = threading.Lock()
        self._demo_model_error: str | None = None

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

    def infer(self, text: str) -> dict:
        """Run inference through the first available demo checkpoint.

        Loads the checkpoint at most once (cached for the life of the
        server process) and reuses it for every request. Raises the same
        exceptions ``DemoModel``/``DemoModel.predict`` raise; the HTTP
        handler is responsible for turning those into a JSON error
        response — this method never fabricates a result.
        """
        with self._demo_model_lock:
            if self._demo_model is None and self._demo_model_error is None:
                live = self.get().get("live_demo", {})
                checkpoints = live.get("checkpoints") or []
                if not checkpoints:
                    self._demo_model_error = "no demo checkpoint available"
                else:
                    from dashboard.inference import DemoModel

                    checkpoint_dir = self._results_dir.parent / checkpoints[0]["checkpoint_dir"]
                    try:
                        self._demo_model = DemoModel(checkpoint_dir)
                    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not swallowed
                        self._demo_model_error = str(exc)
            if self._demo_model is None:
                raise RuntimeError(self._demo_model_error or "demo model unavailable")
            model = self._demo_model
        return model.predict(text)


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

            if path == "/api/analysis/plots/":
                # Trailing-slash listing form; not a file request.
                self._send_json({"available_plots": analysis_loader.available_plots(store._results_dir)})
                return

            if path.startswith("/api/analysis/plots/"):
                # Path-traversal defense: the requested name is checked
                # against a fixed allowlist (KNOWN_PLOTS) inside
                # resolve_plot_path — nothing derived from the URL is ever
                # joined onto a directory and opened without that check
                # passing first, so "../" or an absolute path is rejected
                # by simple non-membership in the allowlist, not by string
                # sanitization.
                name = path[len("/api/analysis/plots/"):]
                plot_path = analysis_loader.resolve_plot_path(store._results_dir, name)
                if plot_path is None:
                    self.send_error(404, "Unknown plot")
                    return
                body = plot_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(404, "Not found")

        def do_POST(self):  # noqa: N802 (stdlib method name)
            if self.path != "/api/infer":
                self.send_error(404, "Not found")
                return

            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0 or length > 20_000:
                self._send_json({"error": "missing or oversized request body"}, status=400)
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json({"error": "request body must be JSON"}, status=400)
                return

            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                self._send_json({"error": "'text' must be a non-empty string"}, status=400)
                return

            try:
                result = store.infer(text)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:  # noqa: BLE001 - reported to caller, not swallowed
                self._send_json({"error": f"inference unavailable: {exc}"}, status=503)
                return
            self._send_json(result)

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
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Interface to bind to. Default 127.0.0.1 (localhost only, safest). "
            "Use 0.0.0.0 to accept connections from other devices on your "
            "local network (LAN). Binding to 0.0.0.0 does NOT make the "
            "dashboard reachable from the public internet — see printed "
            "guidance at startup."
        ),
    )
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
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print("Transformer sweep dashboard (read-only)")
    print(f"Reading results from: {results_dir}")
    print(f"Bound to: {args.host}:{args.port}")
    print()
    print("Local:")
    print(f"  http://127.0.0.1:{args.port}/")
    if args.host == "0.0.0.0":
        lan_ip = local_lan_ip()
        print()
        print("LAN (other devices on the same Wi-Fi/router):")
        if lan_ip:
            print(f"  http://{lan_ip}:{args.port}/")
        else:
            print("  (could not determine this machine's LAN IP automatically;")
            print("   check your OS network settings)")
        print()
        print("NOTE: a private LAN address (e.g. 192.168.x.x or 10.x.x.x) is")
        print("NOT a public internet address. A device on mobile data/GSM or")
        print("a different network cannot reach it.")
        print()
        print("To let someone outside your LAN view this dashboard, use a")
        print("secure tunnel (recommended) rather than exposing this machine")
        print("directly to the internet. Example, in a second terminal:")
        print(f"  cloudflared tunnel --url http://127.0.0.1:{args.port}")
        print("  (or: ngrok http " + str(args.port) + " / tailscale funnel " + str(args.port) + ")")
        print("The tunnel tool prints the externally-reachable URL; this")
        print("script does not install, launch, or depend on any tunnel tool.")
        print()
        print("WARNING: public access makes this dashboard reachable by people")
        print("outside your local network. Do not expose sensitive files or")
        print("credentials. This server only serves the dashboard UI and the")
        print("read-only experiment artifacts under --results-dir; it is not")
        print("a general-purpose file server.")
    else:
        print()
        print("Bound to localhost only. Re-run with --host 0.0.0.0 to allow")
        print("other devices on your local network to connect.")
    print()
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
