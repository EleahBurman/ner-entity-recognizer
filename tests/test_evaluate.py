"""
Unit tests for extract_predictions_and_labels in src/evaluate_model.py.

Uses hand-built fake logits/labels instead of a real model, so this runs
instantly with no GPU, no dataset download, and no trained model required.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from evaluate_model import extract_predictions_and_labels

LABEL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]


def test_filters_out_ignored_positions():
    # Two tokens of "real" content, surrounded by -100 (special/subword
    # tokens) that should be dropped entirely from the output.
    logits = np.array(
        [
            [
                [10, 0, 0, 0, 0],  # confidently predicts label 0 ("O")
                [0, 10, 0, 0, 0],  # confidently predicts label 1 ("B-PER")
                [0, 0, 0, 0, 0],  # doesn't matter, label is -100 here
            ]
        ]
    )
    labels = np.array([[0, 1, -100]])

    predictions, true = extract_predictions_and_labels(logits, labels, LABEL_NAMES)

    assert predictions == [["O", "B-PER"]]
    assert true == [["O", "B-PER"]]


def test_mismatched_prediction_still_reported_correctly():
    # The model predicts "B-ORG" but the true label is "B-PER" — this
    # should show up as a genuine mismatch, not silently hidden.
    logits = np.array([[[0, 0, 0, 10, 0]]])  # predicts index 3 = "B-ORG"
    labels = np.array([[1]])  # true label index 1 = "B-PER"

    predictions, true = extract_predictions_and_labels(logits, labels, LABEL_NAMES)

    assert predictions == [["B-ORG"]]
    assert true == [["B-PER"]]


def test_every_token_perfectly_predicted():
    # A full, realistic sentence-length example where every prediction
    # matches the true label exactly — the "everything went right" case.
    logits = np.array(
        [
            [
                [10, 0, 0, 0, 0],  # O
                [0, 10, 0, 0, 0],  # B-PER
                [0, 0, 10, 0, 0],  # I-PER
                [10, 0, 0, 0, 0],  # O
            ]
        ]
    )
    labels = np.array([[0, 1, 2, 0]])

    predictions, true = extract_predictions_and_labels(logits, labels, LABEL_NAMES)

    assert predictions == true == [["O", "B-PER", "I-PER", "O"]]


def test_all_tokens_ignored_returns_empty_lists():
    # An edge case: what if a whole example is just special tokens
    # (e.g. an empty or truncated-to-nothing input)? Should produce an
    # empty list for that example, not crash or produce garbage.
    logits = np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    labels = np.array([[-100, -100]])

    predictions, true = extract_predictions_and_labels(logits, labels, LABEL_NAMES)

    assert predictions == [[]]
    assert true == [[]]


def test_handles_multiple_examples_in_one_batch():
    # Real batches contain more than one example at a time — confirm each
    # example's predictions/labels stay correctly separated from the others
    # rather than getting merged or mixed up across the batch dimension.
    logits = np.array(
        [
            [[10, 0, 0, 0, 0], [0, 10, 0, 0, 0]],  # example 1: O, B-PER
            [[0, 0, 0, 10, 0], [0, 0, 0, 0, 10]],  # example 2: B-ORG, I-ORG
        ]
    )
    labels = np.array([[0, 1], [3, 4]])

    predictions, true = extract_predictions_and_labels(logits, labels, LABEL_NAMES)

    assert predictions == [["O", "B-PER"], ["B-ORG", "I-ORG"]]
    assert true == [["O", "B-PER"], ["B-ORG", "I-ORG"]]


def test_tie_in_logits_picks_lowest_index_deterministically():
    # When two labels are exactly tied for highest score, numpy's argmax
    # always picks the first (lowest-index) one — confirming that behavior
    # explicitly here means a future numpy/code change that broke this
    # silently would get caught by a test, not discovered during training.
    logits = np.array([[[5, 5, 0, 0, 0]]])  # indices 0 and 1 are tied
    labels = np.array([[0]])

    predictions, _ = extract_predictions_and_labels(logits, labels, LABEL_NAMES)

    assert predictions == [["O"]]  # index 0 ("O") wins the tie


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))