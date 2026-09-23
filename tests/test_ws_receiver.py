"""Tests for the Home Assistant WebSocket receiver."""

from __future__ import annotations

import logging
import socket
import sys
import threading
from typing import Iterator

import pytest

from plugins.voice_stack import ws_receiver
import plugins.voice_stack as voice_stack


@pytest.fixture(autouse=True)
def _reset_assist_handler():
    ws_receiver.set_assist_query_handler(None)
    yield
    ws_receiver.set_assist_query_handler(None)


@pytest.mark.asyncio
async def test_assist_query_returns_response_from_configured_handler():
    """assist_query should produce assist_response instead of timing out."""

    async def handler(payload: dict) -> dict:
        assert payload["text"] == "Hello Hermes"
        return {"text": "Hello from Hermes", "provider": "test-provider"}

    ws_receiver.set_assist_query_handler(handler)

    response = await ws_receiver.handle_ha_ws_payload_async(
        {
            "id": "req-1",
            "type": "assist_query",
            "text": "Hello Hermes",
            "conversation_id": "conv-1",
            "language": "en",
        }
    )

    assert response["id"] == "req-1"
    assert response["type"] == "assist_response"
    assert response["ok"] is True
    assert response["text"] == "Hello from Hermes"
    assert response["conversation_id"] == "conv-1"
    assert response["speech"]["plain"]["speech"] == "Hello from Hermes"
    assert response["provider"] == "test-provider"


@pytest.mark.asyncio
async def test_assist_query_without_handler_returns_spoken_fallback():
    """Missing handler should resolve HA's pending future with a fallback."""

    response = await ws_receiver.handle_ha_ws_payload_async(
        {
            "type": "assist_query",
            "text": "Are you there?",
            "conversation_id": "conv-2",
        }
    )

    assert response["type"] == "assist_response"
    assert response["ok"] is False
    assert response["conversation_id"] == "conv-2"
    assert "not available" in response["text"]
    assert response["speech"]["plain"]["speech"] == response["text"]


@pytest.mark.asyncio
async def test_assist_query_handler_exception_returns_spoken_error():
    """Handler exceptions should not fall back to unsupported-message errors."""

    def handler(_payload: dict) -> dict:
        raise RuntimeError("boom")

    ws_receiver.set_assist_query_handler(handler)

    response = await ws_receiver.handle_ha_ws_payload_async(
        {
            "id": "req-3",
            "type": "assist_query",
            "text": "break",
            "conversation_id": "conv-3",
        }
    )

    assert response["id"] == "req-3"
    assert response["type"] == "assist_response"
    assert response["ok"] is False
    assert response["error"] == "boom"
    assert response["conversation_id"] == "conv-3"


def test_sync_payload_handler_still_reports_unsupported_for_unknown_types():
    """Existing synchronous handler semantics are preserved."""

    response = ws_receiver.handle_ha_ws_payload({"id": "x", "type": "banana"})

    assert response == {
        "id": "x",
        "type": "error",
        "ok": False,
        "error": "Unsupported message type: banana",
    }


@pytest.mark.asyncio
async def test_voice_stack_assist_handler_uses_ctx_llm():
    """The registered voice-stack handler should call Hermes plugin LLM access."""

    class _Result:
        text = "LLM reply"
        provider = "provider-x"
        model = "model-y"

    class _Llm:
        def __init__(self):
            self.calls = []

        async def acomplete(self, **kwargs):
            self.calls.append(kwargs)
            return _Result()

    class _Ctx:
        def __init__(self):
            self.llm = _Llm()

    ctx = _Ctx()

    result = await voice_stack._handle_assist_query_with_llm(
        ctx,
        {
            "text": "What is the weather?",
            "language": "en-AU",
            "conversation_id": "conv-4",
        },
    )

    assert result == {
        "ok": True,
        "text": "LLM reply",
        "conversation_id": "conv-4",
        "provider": "provider-x",
        "model": "model-y",
    }
    assert ctx.llm.calls[0]["purpose"] == "voice_stack.assist_query"
    assert "What is the weather?" in ctx.llm.calls[0]["messages"][1]["content"]


# --------------------------------------------------------------------------- #
# Bind idempotency                                                            #
#                                                                             #
# Every Hermes process that loads this plugin (gateway, CLI session,          #
# dashboard) re-imports the module with fresh globals and calls register(),    #
# so several processes legitimately race for the same port. Only the first     #
# one may bind; the others must stay quiet instead of warning about a port     #
# they are not supposed to own.                                               #
# --------------------------------------------------------------------------- #

requires_aiohttp = pytest.mark.skipif(
    not ws_receiver.AIOHTTP_AVAILABLE,
    reason="aiohttp is not a dev extra; the receiver only serves with it installed",
)


class _FakeListener:
    """Minimal TCP server answering every connection with one HTTP status line.

    Stands in for the two things that can already hold the port: another Hermes
    process (which answers the receiver's path) and an unrelated service (which
    does not).
    """

    def __init__(self, status_line: bytes) -> None:
        self._status_line = status_line
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = int(self._sock.getsockname()[1])
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stopped.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:  # pragma: no cover - closed while accepting
                return
            with conn:
                try:
                    conn.recv(1024)
                    conn.sendall(self._status_line + b"\r\nContent-Length: 0\r\n\r\n")
                except OSError:  # pragma: no cover - client went away
                    pass

    def close(self) -> None:
        self._stopped.set()
        try:
            self._sock.close()
        except OSError:  # pragma: no cover - already closed
            pass
        self._thread.join(timeout=5)


@pytest.fixture
def free_port() -> int:
    """A port nothing is listening on."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture(autouse=True)
def _clean_receiver() -> Iterator[None]:
    """Never leak a started receiver (or its sys marker) into another test."""
    yield
    marker = getattr(sys, ws_receiver._PROC_SINGLETON_ATTR, None)
    if marker is not None:
        marker.stop()
        if getattr(sys, ws_receiver._PROC_SINGLETON_ATTR, None) is marker:
            delattr(sys, ws_receiver._PROC_SINGLETON_ATTR)
    ws_receiver._WS_SERVER = None


def test_probe_recognises_another_hermes_process() -> None:
    """A 401 on our path means the receiver is already served — do not bind."""
    listener = _FakeListener(b"HTTP/1.1 401 Unauthorized")
    try:
        assert ws_receiver._probe_receiver(
            "127.0.0.1", listener.port, ws_receiver.DEFAULT_WS_PATH
        )
    finally:
        listener.close()


def test_probe_ignores_a_foreign_service() -> None:
    """A 404 is somebody else's service: a real conflict must still surface."""
    listener = _FakeListener(b"HTTP/1.1 404 Not Found")
    try:
        assert not ws_receiver._probe_receiver(
            "127.0.0.1", listener.port, ws_receiver.DEFAULT_WS_PATH
        )
    finally:
        listener.close()


def test_probe_is_false_when_nothing_listens(free_port: int) -> None:
    assert not ws_receiver._probe_receiver(
        "127.0.0.1", free_port, ws_receiver.DEFAULT_WS_PATH
    )


def test_is_port_conflict_matches_both_errno_and_message() -> None:
    assert ws_receiver._is_port_conflict(OSError(98, "Address already in use"))
    assert ws_receiver._is_port_conflict(
        OSError("error while attempting to bind on address ('0.0.0.0', 7860): "
                "address already in use")
    )
    assert not ws_receiver._is_port_conflict(OSError(99, "Cannot assign requested address"))


def test_process_receiver_is_duck_typed() -> None:
    """The instance found on ``sys`` belongs to another module object.

    A class-identity check would reject a perfectly live receiver after the
    plugin is re-imported, so the lookup must only require ``running``.
    """

    class _ReimportedReceiver:
        running = True

    fake = _ReimportedReceiver()
    setattr(sys, ws_receiver._PROC_SINGLETON_ATTR, fake)
    try:
        assert ws_receiver._process_receiver() is fake
    finally:
        delattr(sys, ws_receiver._PROC_SINGLETON_ATTR)

    assert ws_receiver._process_receiver() is None


@requires_aiohttp
def test_running_requires_a_completed_bind() -> None:
    """``_started`` alone must never report a live receiver.

    A failed bind sets ``_started`` too (the caller must not wait forever), so
    judging on it handed back a dead server while the thread was still cleaning
    up.
    """
    server = ws_receiver.HermesHAWebSocketServer(
        "127.0.0.1", 0, ws_receiver.DEFAULT_WS_PATH
    )
    assert not server.running
    server._started.set()  # startup attempt finished — it failed
    assert not server.running
    server._ready.set()  # bound... but no live thread either
    assert not server.running


@requires_aiohttp
def test_start_is_quiet_when_another_process_serves_the_port(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HA_WS_ENABLED", "1")
    listener = _FakeListener(b"HTTP/1.1 401 Unauthorized")
    try:
        with caplog.at_level(logging.INFO, logger=ws_receiver.__name__):
            started = ws_receiver.start_ws_receiver(
                host="127.0.0.1", port=listener.port, path=ws_receiver.DEFAULT_WS_PATH
            )
    finally:
        listener.close()

    assert started is None
    assert "already served" in caplog.text
    assert "failed to start" not in caplog.text
    assert getattr(sys, ws_receiver._PROC_SINGLETON_ATTR, None) is None


@requires_aiohttp
def test_start_warns_when_a_foreign_service_holds_the_port(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HA_WS_ENABLED", "1")
    listener = _FakeListener(b"HTTP/1.1 404 Not Found")
    try:
        with caplog.at_level(logging.INFO, logger=ws_receiver.__name__):
            started = ws_receiver.start_ws_receiver(
                host="127.0.0.1", port=listener.port, path=ws_receiver.DEFAULT_WS_PATH
            )
    finally:
        listener.close()

    assert not started
    assert "failed to start" in caplog.text


@requires_aiohttp
def test_start_reuses_the_receiver_across_reimports(
    free_port: int, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HA_WS_ENABLED", "1")
    first = ws_receiver.start_ws_receiver(
        host="127.0.0.1", port=free_port, path=ws_receiver.DEFAULT_WS_PATH
    )
    assert first is not None and first.running
    try:
        assert getattr(sys, ws_receiver._PROC_SINGLETON_ATTR, None) is first

        # A fresh import wipes the module globals: the sys marker is all that is left.
        ws_receiver._WS_SERVER = None
        with caplog.at_level(logging.INFO, logger=ws_receiver.__name__):
            second = ws_receiver.start_ws_receiver(
                host="127.0.0.1", port=free_port, path=ws_receiver.DEFAULT_WS_PATH
            )

        assert second is first
        assert "failed to start" not in caplog.text
    finally:
        first.stop()
