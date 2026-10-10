"""Call the API in-process, without a server, for the smoke check."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI


async def _call(app: FastAPI, path: str, query: str) -> tuple[int, bytes]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 80),
    }
    await app(scope, receive, send)
    status = next(item["status"] for item in sent if item["type"] == "http.response.start")
    body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
    return status, body


def get_json(app: FastAPI, path: str, query: str = "") -> tuple[int, Any]:
    """Send a GET request to the application and return the status code and the JSON body."""
    status, body = asyncio.run(_call(app, path, query))
    return status, json.loads(body)
