"""Web dashboard: runs the paper trader in the background and serves a live page.

    python -m updown dashboard            # live data, http://localhost:8050
    python -m updown dashboard --demo     # offline simulation
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("static") / "dashboard.html"


def serve(trader, host: str = "127.0.0.1", port: int = 8050, poll: float = 2.0):
    stop = threading.Event()
    worker = threading.Thread(target=trader.run, kwargs={"poll": poll, "verbose": False, "stop": stop},
                              daemon=True)
    worker.start()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                self._send(200, json.dumps(trader.snapshot(), default=str).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):  # keep the console quiet
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
