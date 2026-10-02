"""
Unit tests for extract_predictions_and_labels in src/evaluate.py.

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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))