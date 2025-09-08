def test_import_bot_module():
    import importlib

    m = importlib.import_module("bot")
    assert hasattr(m, "main")
