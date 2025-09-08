from bot import _format_duration


def test_format_duration_zero():
    assert _format_duration(0) == "0s"


def test_format_duration_seconds():
    assert _format_duration(45) == "45s"


def test_format_duration_minutes():
    assert _format_duration(60) == "1m"
    assert _format_duration(125) == "2m"


def test_format_duration_hours_minutes():
    assert _format_duration(3700).startswith("1h")
