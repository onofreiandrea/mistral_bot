from types import SimpleNamespace

from prometheus_client import generate_latest

import bot


def test_mistral_success_records_latency(monkeypatch):
    # Arrange: fake successful response
    fake_msg = SimpleNamespace(content="ok")
    fake_choice = SimpleNamespace(message=fake_msg)
    fake_resp = SimpleNamespace(choices=[fake_choice])

    def fake_complete(**kwargs):
        return fake_resp

    monkeypatch.setattr(
        bot,
        "mistral_client",
        SimpleNamespace(chat=SimpleNamespace(complete=fake_complete)),
    )

    # Act
    out = bot.get_mistral_response([{"role": "user", "content": "hi"}], max_tokens=5)

    # Assert
    assert out == "ok"
    metrics = generate_latest(bot.registry).decode("utf-8", errors="ignore")
    assert "mistral_latency_seconds_bucket" in metrics


def test_mistral_error_increments_counter(monkeypatch):
    # Arrange: raise typical 429 message
    def fake_complete(**kwargs):
        raise Exception("API error occurred: Status 429 capacity")

    monkeypatch.setattr(
        bot,
        "mistral_client",
        SimpleNamespace(chat=SimpleNamespace(complete=fake_complete)),
    )

    # Act
    _ = bot.get_mistral_response([{"role": "user", "content": "hi"}], max_tokens=5)

    # Assert
    metrics = generate_latest(bot.registry).decode("utf-8", errors="ignore")
    assert "mistral_errors_total" in metrics
    assert 'code="429"' in metrics
