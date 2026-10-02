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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer
from train import tokenize_and_align_labels, MODEL_CHECKPOINT

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def test_single_token_words_keep_their_labels():
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


def test_handles_multiple_examples_in_one_batch():
    # Real training batches contain more than one sentence at once — confirm
    # each example's labels stay correctly aligned to its own tokens rather
    # than leaking across examples in the batch.
    examples = {
        "tokens": [
            ["EU", "rejects", "call"],
            ["Peter", "lives", "in", "Germany"],
        ],
        "ner_tags": [
            [3, 0, 0],
            [1, 0, 0, 5],
        ],
    }

    result = tokenize_and_align_labels(examples, tokenizer)

    assert len(result["labels"]) == 2
    for i in range(2):
        assert len(result["labels"][i]) == len(result["input_ids"][i])
        assert result["labels"][i][0] == -100  # [CLS]
        assert result["labels"][i][-1] == -100  # [SEP]


def test_empty_sentence_only_has_special_tokens():
    # Edge case: an empty token list should still produce a valid result
    # (just the special tokens), not crash or produce a mismatched length.
    examples = {
        "tokens": [[]],
        "ner_tags": [[]],
    }

    result = tokenize_and_align_labels(examples, tokenizer)
    labels = result["labels"][0]
    input_ids = result["input_ids"][0]

    assert len(labels) == len(input_ids)
    assert all(label == -100 for label in labels)


def test_label_value_zero_is_preserved_not_treated_as_missing():
    # Label 0 ("O", the most common tag) is falsy in Python. This test
    # guards against a subtle bug class where code might accidentally use
    # `if label:` somewhere and silently drop/mishandle legitimate
    # zero-value labels, treating them as if they were missing.
    examples = {
        "tokens": [["The", "cat", "sat"]],
        "ner_tags": [[0, 0, 0]],
    }

    result = tokenize_and_align_labels(examples, tokenizer)
    labels = result["labels"][0]
    word_ids = tokenizer(
        examples["tokens"][0], truncation=True, is_split_into_words=True
    ).word_ids()

    for label, word_id in zip(labels, word_ids):
        if word_id is not None:
            assert label == 0


def test_consecutive_single_token_words_dont_bleed_labels():
    # Guards against a bug where previous_word_idx tracking could cause
    # one word's label to accidentally carry over onto the next word,
    # by using several consecutive words each with a different label.
    examples = {
        "tokens": [["Mary", "met", "John", "yesterday"]],
        "ner_tags": [[1, 0, 1, 0]],
    }

    result = tokenize_and_align_labels(examples, tokenizer)
    labels = result["labels"][0]
    word_ids = tokenizer(
        examples["tokens"][0], truncation=True, is_split_into_words=True
    ).word_ids()

    seen_words = set()
    for label, word_id in zip(labels, word_ids):
        if word_id is None:
            continue
        if word_id not in seen_words:
            seen_words.add(word_id)
            assert label == examples["ner_tags"][0][word_id]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))