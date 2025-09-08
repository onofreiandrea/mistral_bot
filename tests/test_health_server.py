import socket
import time

import bot


def _read_http_response(sock):
    data = b""
    sock.settimeout(2)
    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data:
                break
        except Exception:
            break
    return data


def test_health_endpoints_start(monkeypatch):
    # Bind ephemeral port for health server
    monkeypatch.setenv("HEALTH_PORT", "18080")

    # Start minimal application loop that triggers post_init
    class FakeBot:
        async def get_me(self):
            return type("Me", (), {"username": "testbot"})()

    class FakeApp:
        bot = FakeBot()

    # Run post_init to start health server
    import asyncio

    asyncio.get_event_loop().run_until_complete(bot.post_init(FakeApp()))
    time.sleep(0.2)

    # Connect to /ready
    s = socket.create_connection(("127.0.0.1", 18080), timeout=2)
    s.sendall(b"GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n")
    resp = _read_http_response(s)
    s.close()
    assert b" 200" in resp or b" 503" in resp

    # Connect to /healthz
    s = socket.create_connection(("127.0.0.1", 18080), timeout=2)
    s.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
    resp = _read_http_response(s)
    s.close()
    assert b" 200" in resp
    assert b"application/json" in resp
