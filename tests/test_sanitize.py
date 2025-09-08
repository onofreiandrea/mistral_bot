from bot import sanitize_output


def test_sanitize_removes_bold_and_trims():
    assert sanitize_output(" **Hello** ") == "Hello"
    assert sanitize_output("no change") == "no change"
