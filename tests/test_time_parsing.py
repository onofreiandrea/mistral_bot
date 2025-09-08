from datetime import datetime, timedelta

import pytest

from bot import _parse_time_spec


def _freeze_utc(monkeypatch, ts: datetime):
    class _DT:
        @staticmethod
        def utcnow():
            return ts

    monkeypatch.setattr("bot.datetime", _DT)


def test_parse_simple_seconds(monkeypatch):
    now = datetime(2025, 1, 1, 12, 0, 0)
    _freeze_utc(monkeypatch, now)
    when, text = _parse_time_spec(["15", "s", "Call", "Bella"])
    assert when == now + timedelta(seconds=15)
    assert text == "Call Bella"


def test_parse_simple_minutes(monkeypatch):
    now = datetime(2025, 1, 1, 12, 0, 0)
    _freeze_utc(monkeypatch, now)
    when, text = _parse_time_spec(["10m", "Standup"])
    assert when == now + timedelta(minutes=10)
    assert text == "Standup"


def test_parse_mixed_duration(monkeypatch):
    now = datetime(2025, 1, 1, 12, 0, 0)
    _freeze_utc(monkeypatch, now)
    when, text = _parse_time_spec(["1h30m", "Water", "plants"])
    assert when == now + timedelta(hours=1, minutes=30)
    assert text == "Water plants"


def test_parse_today_time(monkeypatch):
    now = datetime(2025, 1, 1, 12, 0, 0)
    _freeze_utc(monkeypatch, now)
    when, _ = _parse_time_spec(["14:30", "Meet"])
    assert when == datetime(2025, 1, 1, 14, 30)


def test_parse_tomorrow_time(monkeypatch):
    now = datetime(2025, 1, 1, 23, 50, 0)
    _freeze_utc(monkeypatch, now)
    when, _ = _parse_time_spec(["14:30", "Meet"])
    assert when.date() == datetime(2025, 1, 2).date()


def test_parse_tomorrow_keyword(monkeypatch):
    now = datetime(2025, 1, 1, 12, 0, 0)
    _freeze_utc(monkeypatch, now)
    when, _ = _parse_time_spec(["tomorrow", "09:00", "Flight"])
    assert when == datetime(2025, 1, 2, 9, 0)


def test_parse_invalid():
    with pytest.raises(ValueError):
        _parse_time_spec(["later"])
