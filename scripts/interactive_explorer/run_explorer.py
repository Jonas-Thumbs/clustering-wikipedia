"""CLI launcher for the interactive Wikipedia semantic cluster explorer.

Starts a local HTTP server and automatically opens the user's default web browser.
"""
from __future__ import annotations

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8050
WEB_DIR = Path(__file__).parent / "web"


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        # Keep terminal clean
        pass


def main() -> None:
    html_file = WEB_DIR / "index.html"
    if not html_file.exists():
        print(f"Building explorer bundle first...")
        from build_explorer import main as build
        build()

    url = f"http://localhost:{PORT}/index.html"
    print(f"===========================================================")
    print(f"  Wikipedia Semantic Cluster Explorer is running!")
    print(f"  URL: {url}")
    print(f"  Local file: {html_file.resolve()}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"===========================================================")

    # Open directly in browser
    webbrowser.open(url)

    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()

