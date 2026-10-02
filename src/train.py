"""
Fine-tune a DistilBERT model for Named Entity Recognition (token classification)
on the CoNLL-2003 dataset.

Usage:
    python src/train.py
    python src/train.py --max_samples 500   # smoke-test on a small subset (e.g. CPU/local)
    python src/train.py --epochs 5 --batch_size 16
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR = "ner-model"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, truncate train/val sets to this many examples (useful for smoke tests on CPU).",
    )
    return parser.parse_args()


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizing splits words into subword pieces, so we need to re-align the
    original word-level NER labels to the new subword-level tokens.

    Rule: the first subword of a word gets the real label; subsequent
    subwords of the same word get -100 (ignored in the loss).
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def main():
    args = parse_args()

    print("Loading CoNLL-2003...")
    raw_datasets = load_dataset("conll2003", trust_remote_code=True)
    label_names = raw_datasets["train"].features["ner_tags"].feature.names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    if args.max_samples:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(args.max_samples, len(raw_datasets["train"])))
        )
        raw_datasets["validation"] = raw_datasets["validation"].select(
            range(min(args.max_samples, len(raw_datasets["validation"])))
        )

    print(f"Loading tokenizer/model: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    print("Tokenizing + aligning labels...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels = [
            [label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("Done. Run `python src/evaluate.py` to see test-set metrics.")


if __name__ == "__main__":
    main()