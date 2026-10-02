# NER Fine-Tuning Project

Fine-tuning a transformer (DistilBERT) for Named Entity Recognition using
PyTorch + Hugging Face `transformers` and `datasets`.

## Why this project

- Demonstrates end-to-end ML pipeline skills: data loading, tokenization
  with label alignment, fine-tuning, evaluation, inference.
- Portfolio-relevant for both general SWE/ML roles and quant/finance-adjacent
  roles — the architecture here is domain-agnostic, and Phase 2 swaps in a
  financial-text NER dataset (tickers, company names, monetary amounts).

## Phase 1 (today): Baseline on CoNLL-2003

We start with CoNLL-2003 (standard PER/ORG/LOC/MISC entities) because it's
a clean, well-documented dataset with no download friction — the point today
is to get the full pipeline running end-to-end, not to pick the perfect
dataset first.

### Run in Google Colab (recommended for today)

1. Open a new Colab notebook, enable GPU: `Runtime > Change runtime type > T4 GPU`
2. Upload this whole `ner-entity-recognizer` folder (or just `src/` and
   `requirements.txt`) via the Colab file browser, or `git clone` if you
   push this to GitHub first.
3. In a Colab cell:
   ```
   !pip install -r requirements.txt
   !python src/train.py
   ```
4. Evaluate:
   ```
   !python src/evaluate.py
   ```
5. Try inference on your own sentence:
   ```
   !python src/predict.py --text "Satya Nadella met with regulators in Brussels on Tuesday."
   ```

### Run locally

Same commands, minus the `!`. Training will be slow/CPU-bound without a GPU —
fine for smoke-testing the pipeline on a small subset (see `--max_samples` in
`train.py`), but do the real training run in Colab or AWS.

## Phase 2 (next session): Move to AWS + swap in financial NER data

- Swap `load_dataset("conll2003")` in `train.py` for a financial NER dataset
  (e.g. FiNER-139, or a scraped/labeled set of earnings-call transcripts /
  SEC filings).
- Move training to an AWS SageMaker training job or an EC2 GPU instance —
  this is the resume-worthy step: "productionized training pipeline from
  notebook prototype to cloud training job."
- Add a small FastAPI inference endpoint so the model is servable, not just
  a script.

## Project structure

```
ner-entity-recognizer/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py       # fine-tunes DistilBERT for token classification
│   ├── evaluate.py    # computes precision/recall/F1 (seqeval) on test set
│   └── predict.py     # run inference on arbitrary text
└── data/              # (empty for now — Phase 1 pulls data via `datasets`)
```