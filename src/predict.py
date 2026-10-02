"""
Run the fine-tuned NER model on a sentence of your choice.

Usage:
    python src/predict.py --text "Satya Nadella met with regulators in Brussels on Tuesday."
    python src/predict.py --model_dir ner-model/final --text "..."
"""

import argparse
from dataclasses import dataclass

from transformers import pipeline


@dataclass
class PredictArgs:
    """
    Typed container for command-line arguments, same pattern as
    TrainArgs/EvalArgs — keeps args.model_dir / args.text recognized by
    editors/type-checkers instead of showing up as unresolved attributes
    on a generic argparse Namespace.
    """

    model_dir: str
    text: str


def parse_args() -> PredictArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="ner-model/final")
    parser.add_argument("--text", type=str, required=True)
    namespace = parser.parse_args()
    return PredictArgs(model_dir=namespace.model_dir, text=namespace.text)


def format_entity_results(text: str, results: list) -> list[str]:
    """
    Turns the raw list of entity dicts the HF pipeline returns into
    human-readable display lines. Pulled out as its own function (instead
    of printing directly inside main()) specifically so it's unit-testable
    with made-up fake results, with no model/pipeline required at all.

    `results` is a list of dicts shaped like:
        {"word": "Brussels", "entity_group": "LOC", "score": 0.998}
    """
    if not text.strip():
        return ["No input text provided."]

    if not results:
        return ["No entities found."]

    lines = []
    for entity in results:
        lines.append(
            f"  {entity['word']:20s}  {entity['entity_group']:6s}  "
            f"score={entity['score']:.3f}"
        )
    return lines


def main():
    args = parse_args()

    ner_pipeline = pipeline(
        "token-classification",
        model=args.model_dir,
        tokenizer=args.model_dir,
        aggregation_strategy="simple",  # merges subword pieces into whole entities
    )

    results = ner_pipeline(args.text)

    print(f"\nInput: {args.text}\n")
    for line in format_entity_results(args.text, results):
        print(line)


if __name__ == "__main__":
    main()