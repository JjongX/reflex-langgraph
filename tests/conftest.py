"""Shared test helpers (the scripted chat model is in ``scripted.py``)."""

import json
import time

import httpx
import pytest

RUNNING = ("starting", "fingerprinting", "running")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def sse_events(client, url, body, headers=None):
    """POST to an SSE route and return ``[(event, data), ...]`` once the stream ends."""
    out = []
    async with client.stream("POST", url, json=body, headers=headers or {}) as resp:
        assert resp.status_code == 200, await resp.aread()
        event = None
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                out.append((event, json.loads(line.split(":", 1)[1])))
    return out


async def collect(stream):
    """Drain a ``BackendClient`` event stream."""
    return [(e, d) async for e, d in stream]


def wait(runner, job_id, done=None, timeout=20):
    """Poll a job until its status is in ``done`` (default: anything but running)."""
    end = time.time() + timeout
    while time.time() < end:
        s = runner.summary(job_id, public=False)
        if (s["status"] in done) if done else (s["status"] not in RUNNING):
            return s
        time.sleep(0.1)
    raise AssertionError(runner.summary(job_id, public=False))
