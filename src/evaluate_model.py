"""
Evaluate the fine-tuned NER model on the CoNLL-2003 test split.

Usage:
    python src/evaluate_model.py
    python src/evaluate_model.py --model_dir ner-model/final
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
import evaluate as evaluate_lib

from train import tokenize_and_align_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="ner-model/final")
    return parser.parse_args()


def extract_predictions_and_labels(logits, labels, label_names):
    """
    Converts raw model output (logits) and raw label IDs into the
    string-label format seqeval expects, filtering out the -100
    "ignore this token" positions (special tokens, subword continuations)
    that tokenize_and_align_labels introduced during preprocessing.

    Pulled out as its own function (separate from main()) specifically so
    it can be unit-tested with plain made-up arrays, without needing an
    actual trained model or tokenizer.
    """
    predictions = np.argmax(logits, axis=-1)

    true_labels = [
        [label_names[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_predictions, true_labels


def main():
    args = parse_args()

    print("Loading test split of CoNLL-2003...")
    raw_datasets = load_dataset("conll2003", trust_remote_code=True)
    label_names = raw_datasets["train"].features["ner_tags"].feature.names

    print(f"Loading fine-tuned model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    tokenized_test = raw_datasets["test"].map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    trainer = Trainer(model=model, tokenizer=tokenizer)
    logits, labels, _ = trainer.predict(tokenized_test)
    true_predictions, true_labels = extract_predictions_and_labels(
        logits, labels, label_names
    )

    seqeval = evaluate_lib.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    print("\n=== Test set results ===")
    for entity_type, metrics in results.items():
        if isinstance(metrics, dict):
            print(
                f"{entity_type:10s} precision={metrics['precision']:.3f}  "
                f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}"
            )
    print(f"\nOverall precision: {results['overall_precision']:.3f}")
    print(f"Overall recall:    {results['overall_recall']:.3f}")
    print(f"Overall F1:        {results['overall_f1']:.3f}")
    print(f"Overall accuracy:  {results['overall_accuracy']:.3f}")


if __name__ == "__main__":
    main()