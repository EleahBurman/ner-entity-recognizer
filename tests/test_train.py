"""
Smoke tests for the label-alignment logic in src/train.py.

These don't run actual model training (too slow/heavy for a quick check) —
they verify the trickiest, easiest-to-get-wrong part of the pipeline: lining
up word-level NER labels with subword-level tokens after tokenization.

Run with:
    pytest tests/test_train.py
"""

import os
import sys

# Make src/ importable as a module without needing an __init__.py or
# installing this project as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer
from train import tokenize_and_align_labels, MODEL_CHECKPOINT


def test_single_token_words_keep_their_labels():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    examples = {
        "tokens": [["EU", "rejects", "call"]],
        "ner_tags": [[3, 0, 0]],
    }

    result = tokenize_and_align_labels(examples, tokenizer)
    labels = result["labels"][0]
    input_ids = result["input_ids"][0]

    # Same number of labels as tokens (including special tokens).
    assert len(labels) == len(input_ids)

    # The first and last positions are the special [CLS]/[SEP] tokens,
    # which should always be ignored (-100) since they aren't real words.
    assert labels[0] == -100
    assert labels[-1] == -100


def test_subword_pieces_only_label_the_first_piece():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # "Lollapalooza" is unusual enough that DistilBERT's tokenizer will
    # almost certainly split it into multiple subword pieces, which is
    # exactly the case this alignment logic needs to handle correctly.
    examples = {
        "tokens": [["Lollapalooza", "starts", "today"]],
        "ner_tags": [[5, 0, 0]],
    }

    result = tokenize_and_align_labels(examples, tokenizer)
    labels = result["labels"][0]
    word_ids = tokenizer(
        examples["tokens"][0], truncation=True, is_split_into_words=True
    ).word_ids()

    # For each word, only its first subword token should carry the real
    # label; every later subword of the same word should be -100.
    seen_words = set()
    for label, word_id in zip(labels, word_ids):
        if word_id is None:
            assert label == -100
            continue
        if word_id not in seen_words:
            seen_words.add(word_id)
            assert label == examples["ner_tags"][0][word_id]
        else:
            assert label == -100


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))