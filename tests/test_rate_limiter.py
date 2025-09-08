from bot import InMemoryRateLimiter


def test_rate_limiter_basic(monkeypatch):
    rl = InMemoryRateLimiter()
    now = 1000.0

    def fake_time():
        return fake_time.t

    fake_time.t = now
    monkeypatch.setattr("bot.time.time", fake_time)

    user = 1
    chat = "c1"
    # within window allow a few
    assert rl.allow(user, chat)
    assert rl.allow(user, chat)
    # advance beyond user window to reset
    fake_time.t = now + 1000
    assert rl.allow(user, chat)
