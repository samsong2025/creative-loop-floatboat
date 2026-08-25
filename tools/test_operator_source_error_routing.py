"""Regression checks for source-acquisition versus language-routing errors.

Run from the repository root:
    python tools/test_operator_source_error_routing.py
"""
from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "api" / "app" / "branding_v09.py"


def _function_source(name: str) -> str:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function not found: {name}")


def test_media_acquisition_failure_precedes_language_unresolved():
    """No media URL means Whisper cannot have run, regardless of metadata."""
    source = _function_source("_operator_run_crawl")
    acquisition = source.index('"error": "detail_or_media_download_failed"')
    language = source.index('"error": (\n                        "source_language_unresolved"')
    assert acquisition < language
    assert 'issue.get("reason") == "no_downloadable_media_url"' in source
    assert "The source video was not downloaded" in source


def test_console_has_media_acquisition_message():
    source = SOURCE.read_text(encoding="utf-8")
    assert "code === 'detail_or_media_download_failed'" in source
    assert "媒体下载地址" in source


if __name__ == "__main__":
    test_media_acquisition_failure_precedes_language_unresolved()
    test_console_has_media_acquisition_message()
    print("operator source error routing regressions passed")
