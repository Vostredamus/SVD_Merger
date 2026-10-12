from pathlib import Path

from llmtraits import paths


def test_relocated_path_is_a_passthrough():
    original = Path(r"D:\work\compressed\CurrentModel\manifest.json")

    assert paths.relocated_path(original) == original


def test_relocated_path_accepts_strings():
    assert paths.relocated_path("some/relative/dir") == Path("some/relative/dir")


def test_relocated_str_returns_a_string():
    result = paths.relocated_str(r"D:\work\compressed\CurrentModel")

    assert result == str(Path(r"D:\work\compressed\CurrentModel"))
