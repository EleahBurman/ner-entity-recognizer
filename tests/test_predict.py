"""
Unit tests for format_entity_results in src/predict.py.

Uses hand-built fake pipeline output instead of a real model, so this runs
instantly with no GPU, no download, and no trained model required.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import format_entity_results


def test_empty_text_returns_message_without_crashing():
    lines = format_entity_results("", [])
    assert lines == ["No input text provided."]


def test_whitespace_only_text_treated_as_empty():
    # " " passes a truthy check in Python but should still be treated as
    # "no real input" — guards against a bug where only exact "" is caught.
    lines = format_entity_results("   ", [])
    assert lines == ["No input text provided."]


def test_no_entities_found_reports_clearly():
    lines = format_entity_results("The weather is nice today.", [])
    assert lines == ["No entities found."]


def test_single_entity_formats_correctly():
    results = [{"word": "Brussels", "entity_group": "LOC", "score": 0.998}]
    lines = format_entity_results("I live in Brussels.", results)

    assert len(lines) == 1
    assert "Brussels" in lines[0]
    assert "LOC" in lines[0]
    assert "0.998" in lines[0]


def test_multiple_entities_each_get_their_own_line():
    results = [
        {"word": "Satya Nadella", "entity_group": "PER", "score": 0.99},
        {"word": "Brussels", "entity_group": "LOC", "score": 0.95},
    ]
    lines = format_entity_results("...", results)

    assert len(lines) == 2
    assert "Satya Nadella" in lines[0]
    assert "Brussels" in lines[1]


def test_long_entity_word_does_not_break_formatting():
    # A word longer than the 20-character padding width should still
    # produce a valid, readable line rather than crashing or truncating.
    long_word = "Supercalifragilisticexpialidocious"
    results = [{"word": long_word, "entity_group": "MISC", "score": 0.5}]
    lines = format_entity_results("...", results)

    assert long_word in lines[0]


def test_low_confidence_score_still_displays_correctly():
    # A near-zero score is a legitimate, if unlikely, result — confirm it
    # formats cleanly rather than e.g. erroring on an edge-case float.
    results = [{"word": "Maybe", "entity_group": "ORG", "score": 0.001}]
    lines = format_entity_results("...", results)

    assert "0.001" in lines[0]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))