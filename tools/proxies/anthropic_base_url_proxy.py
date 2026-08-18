#!/usr/bin/env python3
"""Small Anthropic-compatible proxy for Claude Code endpoint experiments.

Claude Code/claude-agent-sdk talks to an Anthropic Messages-style endpoint.
Some OpenAI-compatible gateways expose `/v1/messages` and `/v1/chat/completions`
but do not behave exactly like the official Anthropic SDK expects. This proxy
keeps Claude Code pointed at a local Anthropic-shaped base URL while forwarding
the raw request to an upstream gateway.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def normalize_base_url(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and "](" in value and value.endswith(")"):
        value = value.split("](", 1)[1][:-1]
    return value.rstrip("/")


def redact(value: str) -> str:
    if len(value) <= 10:
        return "***"
    return f"{value[:5]}...{value[-4:]}"


def split_proxy_path(path: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(path)
    route = parsed.path
    while route.startswith("/v1/v1/"):
        route = route[3:]
    if route.startswith("/v1/"):
        route = route[3:]
    if route == "/v1":
        route = "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return route, query


def rewrite_route(route: str) -> str:
    if route in ("", "/"):
        return "/"
    if route == "/health":
        return route
    if route == "/messages/count_tokens":
        return route
    if route.startswith("/messages"):
        return route
    if route.startswith("/models"):
        return route
    if route.startswith("/chat/completions"):
        return route
    if route.startswith("/responses"):
        return route
    return route


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "AnthropicBaseURLProxy/0.1"

    def do_GET(self) -> None:  # noqa: N802
        route, query = split_proxy_path(self.path)
        if route in ("", "/", "/health"):
            self.send_json(200, {"ok": True, "upstream_base_url": self.server.upstream_base_url})
            return
        self.forward(method="GET", route=rewrite_route(route), query=query, body=None)

    def do_HEAD(self) -> None:  # noqa: N802
        route, _query = split_proxy_path(self.path)
        status = 200 if route in ("", "/", "/health") else 404
        self.send_response(status)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        route, query = split_proxy_path(self.path)
        length = int(self.headers.get("content-length") or "0")
        body = self.rfile.read(length) if length else b""
        self.forward(method="POST", route=rewrite_route(route), query=query, body=body)

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(fmt, *args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def forward(self, *, method: str, route: str, query: str, body: bytes | None) -> None:
        upstream_url = f"{self.server.upstream_base_url}{route}{query}"
        self.server.log(f"{method} {self.path} -> {upstream_url}")
        headers = self.forward_headers()
        if self.server.api_key:
            headers["x-api-key"] = self.server.api_key
            headers["authorization"] = f"Bearer {self.server.api_key}"
        headers.setdefault("anthropic-version", self.server.anthropic_version)

        try:
            request = urllib.request.Request(upstream_url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=self.server.timeout) as response:
                self.server.log(f"{method} {self.path} <- {response.status}")
                self.relay_response(response.status, response.headers, response)
        except urllib.error.HTTPError as exc:
            self.server.log(f"{method} {self.path} <- {exc.code}")
            self.relay_response(exc.code, exc.headers, exc)
        except Exception as exc:  # noqa: BLE001
            self.server.log(f"{method} {self.path} proxy error: {type(exc).__name__}: {exc}")
            self.send_json(
                502,
                {
                    "error": {
                        "type": "proxy_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                },
            )

    def forward_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS:
                continue
            headers[key] = value
        return headers

    def relay_response(self, status: int, headers: Any, stream: Any) -> None:
        self.send_response(status)
        for key, value in headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS:
                continue
            self.send_header(key, value)
        self.end_headers()

        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except BrokenPipeError:
                self.server.log(f"{self.command} {self.path} client disconnected")
                break


class ProxyServer(ThreadingHTTPServer):
    upstream_base_url: str
    api_key: str
    anthropic_version: str
    timeout: int
    verbose: bool

    def log(self, message: str) -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Anthropic-compatible forwarding proxy.")
    parser.add_argument("--listen-host", default=os.environ.get("PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", "38445")))
    parser.add_argument("--upstream-base-url", default=os.environ.get("UPSTREAM_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("UPSTREAM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--anthropic-version", default=os.environ.get("ANTHROPIC_VERSION", "2023-06-01"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("PROXY_TIMEOUT", "600")))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.upstream_base_url:
        print("Set --upstream-base-url or UPSTREAM_BASE_URL.", file=sys.stderr)
        return 2
    if not args.api_key:
        print("Set --api-key or UPSTREAM_API_KEY.", file=sys.stderr)
        return 2

    server = ProxyServer((args.listen_host, args.port), ProxyHandler)
    server.upstream_base_url = normalize_base_url(args.upstream_base_url)
    server.api_key = args.api_key
    server.anthropic_version = args.anthropic_version
    server.timeout = args.timeout
    server.verbose = args.verbose

    print(
        "anthropic proxy listening on "
        f"http://{args.listen_host}:{args.port}/v1 -> {server.upstream_base_url} "
        f"(api_key={redact(args.api_key)})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping proxy", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
