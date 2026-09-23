"""Bind idempotency, receiver identity and owner boundaries.

Every Hermes process that loads this plugin (the gateway, every CLI session, the
dashboard) re-imports the module with fresh globals and calls ``register()``, so
several of them legitimately race for the same port. Only the first one may bind;
the others must stay quiet about a port a Hermes receiver already serves — and
must still shout about a port somebody else's service holds.

Identity is decided by the receiver's own unauthenticated ``/health`` payload
(``service == "hermes-ha-ws"``), never by an HTTP status: a protected foreign API
answers 401, an unrelated upgrade endpoint answers 426, generic validation
answers 400.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pytest

from plugins.voice_stack import ws_receiver

MODULE_PATH = Path(ws_receiver.__file__).resolve()
WS_PATH = ws_receiver.DEFAULT_WS_PATH

requires_aiohttp = pytest.mark.skipif(
    not ws_receiver.AIOHTTP_AVAILABLE,
    reason="aiohttp is not a dev extra; the receiver only serves with it installed",
)


# --------------------------------------------------------------------------- #
# Doubles                                                                     #
# --------------------------------------------------------------------------- #


class _FakeService:
    """A minimal HTTP service holding a port, with a scriptable responder.

    ``responder(target) -> (status, payload)`` decides what each request gets,
    so a test can model both a foreign API and another Hermes receiver.
    """

    def __init__(
        self,
        responder: Callable[[str], tuple[int, Optional[dict[str, Any]]]],
        *,
        family: int = socket.AF_INET,
        host: str = "127.0.0.1",
    ) -> None:
        self._responder = responder
        self._sock = socket.socket(family, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(8)
        bound = self._sock.getsockname()
        self.host, self.port = str(bound[0]), int(bound[1])
        self._stopped = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stopped:
            try:
                conn, _ = self._sock.accept()
            except OSError:  # pragma: no cover - closed while accepting
                return
            with conn:
                try:
                    request = conn.recv(4096).decode("latin-1", "replace")
                    parts = request.split(" ", 2)
                    target = parts[1] if len(parts) > 1 else "/"
                    status, payload = self._responder(target)
                    if payload is None:
                        conn.sendall(
                            f"HTTP/1.1 {status} X\r\nContent-Length: 0\r\n"
                            "Connection: close\r\n\r\n".encode()
                        )
                    else:
                        body = json.dumps(payload).encode()
                        conn.sendall(
                            (
                                f"HTTP/1.1 {status} X\r\n"
                                "Content-Type: application/json\r\n"
                                f"Content-Length: {len(body)}\r\n"
                                "Connection: close\r\n\r\n"
                            ).encode()
                            + body
                        )
                except OSError:  # pragma: no cover - client went away
                    pass

    def close(self) -> None:
        self._stopped = True
        try:
            self._sock.close()
        except OSError:  # pragma: no cover - already closed
            pass
        self._thread.join(timeout=5)


def _identity_responder(path: str = WS_PATH) -> Callable[[str], tuple[int, Optional[dict[str, Any]]]]:
    """Answers like this receiver's /health route."""

    def responder(target: str) -> tuple[int, Optional[dict[str, Any]]]:
        if target.split("?")[0] != ws_receiver.DEFAULT_HEALTH_PATH:
            return 404, None
        return 200, {
            "type": "status",
            "service": ws_receiver.SERVICE_ID,
            "running": True,
            "host": "0.0.0.0",
            "port": 0,
            "path": path,
        }

    return responder


def _protected_api_responder(target: str) -> tuple[int, Optional[dict[str, Any]]]:
    """A foreign, token-protected API: 401 on everything, including /health."""
    return 401, None


def _foreign_health_responder(target: str) -> tuple[int, Optional[dict[str, Any]]]:
    """Somebody else's service that does expose a /health of its own."""
    return 200, {"service": "not-hermes", "path": WS_PATH}


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture(autouse=True)
def _clean_receiver() -> Iterator[None]:
    """Never leak a started receiver (or its sys record) into another test."""
    yield
    record = getattr(sys, ws_receiver._PROC_SINGLETON_ATTR, None)
    if isinstance(record, dict):
        server = record.get("server")
        if server is not None:
            server.stop()
        delattr(sys, ws_receiver._PROC_SINGLETON_ATTR)
    ws_receiver._WS_SERVER = None


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient receiver config leaks into a test's expectations."""
    monkeypatch.setenv("HERMES_HA_WS_ENABLED", "1")
    monkeypatch.delenv("HERMES_HA_WS_TOKEN", raising=False)


def _load_module(name: str, path: Path = MODULE_PATH):
    """Load the receiver module again, as Hermes does for every process."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Identity: the /health marker decides, not the HTTP status                    #
# --------------------------------------------------------------------------- #


def test_probe_requires_the_service_marker() -> None:
    service = _FakeService(_identity_responder())
    try:
        identity = ws_receiver._probe_receiver("127.0.0.1", service.port)
    finally:
        service.close()

    assert identity is not None
    assert identity["service"] == ws_receiver.SERVICE_ID


def test_probe_ignores_a_protected_foreign_api() -> None:
    """A 401 is not identity: a foreign protected API answers 401 too."""
    service = _FakeService(_protected_api_responder)
    try:
        assert ws_receiver._probe_receiver("127.0.0.1", service.port) is None
    finally:
        service.close()


def test_probe_ignores_a_foreign_health_endpoint() -> None:
    service = _FakeService(_foreign_health_responder)
    try:
        assert ws_receiver._probe_receiver("127.0.0.1", service.port) is None
    finally:
        service.close()


def test_probe_is_none_when_nothing_listens(free_port: int) -> None:
    assert ws_receiver._probe_receiver("127.0.0.1", free_port) is None


def test_probe_hosts_covers_both_loopbacks_for_a_wildcard() -> None:
    assert ws_receiver._probe_hosts("0.0.0.0") == ["127.0.0.1", "::1"]
    assert ws_receiver._probe_hosts("::") == ["127.0.0.1", "::1"]
    assert ws_receiver._probe_hosts("192.168.31.5") == ["192.168.31.5"]


def test_probe_finds_an_ipv6_only_receiver_on_a_wildcard_bind() -> None:
    try:
        service = _FakeService(_identity_responder(), family=socket.AF_INET6, host="::1")
    except OSError:  # pragma: no cover - no IPv6 in this environment
        pytest.skip("IPv6 loopback is unavailable")
    try:
        identity = ws_receiver._probe_receiver("::", service.port)
    finally:
        service.close()

    assert identity is not None
    assert identity["service"] == ws_receiver.SERVICE_ID


def test_is_port_conflict_matches_both_errno_and_message() -> None:
    assert ws_receiver._is_port_conflict(OSError(98, "Address already in use"))
    assert ws_receiver._is_port_conflict(
        OSError(
            "error while attempting to bind on address ('0.0.0.0', 7860): "
            "address already in use"
        )
    )
    assert not ws_receiver._is_port_conflict(OSError(99, "Cannot assign requested address"))


# --------------------------------------------------------------------------- #
# Liveness                                                                    #
# --------------------------------------------------------------------------- #


@requires_aiohttp
def test_running_requires_a_completed_bind() -> None:
    """``_started`` alone must never report a live receiver.

    A failed bind sets ``_started`` too (the caller must not wait forever), so
    judging on it handed back a dead server while the thread was still cleaning
    up.
    """
    server = ws_receiver.HermesHAWebSocketServer("127.0.0.1", 0, WS_PATH)
    assert not server.running
    server._started.set()  # the startup attempt finished — it failed
    assert not server.running
    server._ready.set()  # bound... but there is no live thread either
    assert not server.running


# --------------------------------------------------------------------------- #
# Owner boundaries: reuse, reload and other profiles                          #
# --------------------------------------------------------------------------- #


@requires_aiohttp
def test_start_reuses_the_receiver_across_reimports(
    free_port: int, caplog: pytest.LogCaptureFixture
) -> None:
    first_module = _load_module("ws_receiver_first")
    first = first_module.start_ws_receiver(host="127.0.0.1", port=free_port, path=WS_PATH)
    assert first is not None and first.running
    try:
        second_module = _load_module("ws_receiver_second")
        with caplog.at_level(logging.INFO, logger=second_module.__name__):
            second = second_module.start_ws_receiver(
                host="127.0.0.1", port=free_port, path=WS_PATH
            )

        assert second is first
        assert "failed to start" not in caplog.text
    finally:
        first.stop()


@requires_aiohttp
def test_reload_with_changed_code_rebinds(
    free_port: int, caplog: pytest.LogCaptureFixture
) -> None:
    """A reload whose file changed must serve the new code, not the old object.

    Otherwise the reloaded module keeps answering through a receiver still bound
    to the previous module's globals and assist handler.
    """
    first_module = _load_module("ws_receiver_stale_first")
    first = first_module.start_ws_receiver(host="127.0.0.1", port=free_port, path=WS_PATH)
    assert first is not None and first.running

    record = getattr(sys, first_module._PROC_SINGLETON_ATTR)
    path, mtime, size = record["fingerprint"]
    record["fingerprint"] = (path, mtime + 60, size)
    second = None

    try:
        second_module = _load_module("ws_receiver_stale_second")
        with caplog.at_level(logging.INFO, logger=second_module.__name__):
            second = second_module.start_ws_receiver(
                host="127.0.0.1", port=free_port, path=WS_PATH
            )

        assert second is not None and second is not first
        assert second.running
        assert not first.running
        assert "rebinding" in caplog.text
    finally:
        first.stop()
        if second is not None:
            second.stop()


@requires_aiohttp
def test_another_profile_module_does_not_adopt_the_receiver(
    tmp_path: Path, free_port: int, caplog: pytest.LogCaptureFixture
) -> None:
    """A plugin loaded from another profile keeps its hands off.

    It must not reuse — or stop — a receiver another profile's module started.
    """
    first_module = _load_module("ws_receiver_profile_a")
    first = first_module.start_ws_receiver(host="127.0.0.1", port=free_port, path=WS_PATH)
    assert first is not None and first.running

    other_dir = tmp_path / "profiles" / "other" / "plugins" / "voice_stack"
    other_dir.mkdir(parents=True)
    other_file = other_dir / "ws_receiver.py"
    other_file.write_bytes(MODULE_PATH.read_bytes())

    try:
        other_module = _load_module("ws_receiver_profile_b", other_file)
        with caplog.at_level(logging.INFO, logger=other_module.__name__):
            adopted = other_module.start_ws_receiver(
                host="127.0.0.1", port=free_port, path=WS_PATH
            )

        assert adopted is None
        assert first.running
        assert "not reusing the receiver" in caplog.text
    finally:
        first.stop()


@requires_aiohttp
def test_start_is_quiet_when_another_process_serves_the_port(
    free_port: int, caplog: pytest.LogCaptureFixture
) -> None:
    serving_module = _load_module("ws_receiver_other_process")
    serving = serving_module.start_ws_receiver(host="127.0.0.1", port=free_port, path=WS_PATH)
    assert serving is not None and serving.running
    # Another process's record is not on our sys, so drop ours to model that.
    delattr(sys, ws_receiver._PROC_SINGLETON_ATTR)

    try:
        other_module = _load_module("ws_receiver_third")
        with caplog.at_level(logging.INFO, logger=other_module.__name__):
            started = other_module.start_ws_receiver(
                host="127.0.0.1", port=free_port, path=WS_PATH
            )

        assert started is None
        assert "already served" in caplog.text
        assert "failed to start" not in caplog.text
    finally:
        serving.stop()


@requires_aiohttp
def test_start_warns_when_a_foreign_service_holds_the_port(
    free_port: int, caplog: pytest.LogCaptureFixture
) -> None:
    service = _FakeService(_protected_api_responder)
    try:
        with caplog.at_level(logging.INFO, logger=ws_receiver.__name__):
            started = ws_receiver.start_ws_receiver(
                host="127.0.0.1", port=service.port, path=WS_PATH
            )
    finally:
        service.close()

    assert not started
    assert "failed to start" in caplog.text


@requires_aiohttp
def test_start_warns_when_a_hermes_receiver_serves_another_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same receiver, different config: quiet is wrong, the port is not ours to use."""
    service = _FakeService(_identity_responder(path="/api/other/ws"))
    try:
        with caplog.at_level(logging.INFO, logger=ws_receiver.__name__):
            started = ws_receiver.start_ws_receiver(
                host="127.0.0.1", port=service.port, path=WS_PATH
            )
    finally:
        service.close()

    assert started is None
    assert "instead of" in caplog.text


@requires_aiohttp
@pytest.mark.parametrize("token", [None, "secret-token"])
def test_probe_reads_identity_in_both_token_modes(
    token: Optional[str], free_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The health route is the identity source and is not token protected."""
    if token is None:
        monkeypatch.delenv("HERMES_HA_WS_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HERMES_HA_WS_TOKEN", token)

    module = _load_module(f"ws_receiver_token_{token is not None}")
    server = module.start_ws_receiver(host="127.0.0.1", port=free_port, path=WS_PATH)
    assert server is not None and server.running
    try:
        identity = module._probe_receiver("127.0.0.1", free_port)
        assert identity is not None
        assert identity["service"] == module.SERVICE_ID
        assert identity["auth_required"] is (token is not None)
    finally:
        server.stop()