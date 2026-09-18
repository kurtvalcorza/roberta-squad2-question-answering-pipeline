"""SQuAD-style per-pair helpers, corpus-level extractive-QA metrics and two non-neural baselines.

The per-pair helpers (`normalize_answer`, `exact_match`, `f1`) follow the official SQuAD 2.0 evaluation
script's normalisation (re-exported by `pipeline.py`). This module averages them over a dataset and adds two
baselines a fine-tuned reader must beat: **always-null** (the empty answer for every question — it scores
exactly the unanswerable fraction of the set) and **lexical overlap** (return the passage sentence sharing
the most normalised tokens with the question — a bag-of-words reader with no model).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lower-case, drop punctuation and the articles a/an/the, collapse spaces."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    """1.0 if the normalised prediction equals any normalised gold answer (an empty gold list means
    unanswerable and matches only the empty prediction), else 0.0."""
    golds = [normalize_answer(g) for g in gold_answers] or [""]
    return 1.0 if normalize_answer(prediction) in golds else 0.0


def f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Best token-overlap F1 against any gold answer, SQuAD 2.0 style: for an unanswerable gold (empty
    list or empty string) the score is 1.0 only when the prediction is empty too."""
    golds = list(gold_answers) or [""]
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            score = float(pred_tokens == gold_tokens)
        else:
            common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
            if common == 0:
                score = 0.0
            else:
                precision, recall = common / len(pred_tokens), common / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


METRIC_DEFINITIONS = {
    "exact_match": (
        "mean over questions of 1[normalised prediction equals any normalised gold]; SQuAD normalisation "
        "(lower-case, strip punctuation and a/an/the, collapse spaces); an unanswerable gold matches only "
        "the empty prediction; reported in percent"
    ),
    "f1": (
        "mean over questions of the best token-overlap F1 against any gold (1.0 for an empty prediction on "
        "an unanswerable gold, else 0.0 when either side is empty); reported in percent"
    ),
}


def qa_metrics(predictions: Sequence[str], golds: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Corpus exact-match and F1 in percent over parallel predictions and SQuAD-style gold lists."""
    if len(predictions) != len(golds):
        raise ValueError(f"{len(predictions)} predictions but {len(golds)} gold lists")
    if not predictions:
        raise ValueError("no predictions to score")
    em = [exact_match(p, g) for p, g in zip(predictions, golds, strict=True)]
    f1s = [f1(p, g) for p, g in zip(predictions, golds, strict=True)]
    return {
        "n": len(predictions),
        "exact_match": 100.0 * sum(em) / len(em),
        "f1": 100.0 * sum(f1s) / len(f1s),
        "answered_rate": 100.0 * sum(1 for p in predictions if p) / len(predictions),
        "unanswerable_gold": sum(1 for g in golds if not g),
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _golds(records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    return [[a["text"] for a in r["answers"]] for r in records]


def null_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The empty answer for every question: scores the unanswerable fraction of the set and nothing else."""
    result = qa_metrics([""] * len(records), _golds(records))
    result["baseline"] = "always the empty (no-answer) prediction"
    return result


def lexical_overlap_answer(question: str, context: str) -> str:
    """The passage sentence with the most normalised tokens in common with the question (first on ties)."""
    q_tokens = set(normalize_answer(question).split())
    best, best_overlap = "", -1
    for sentence in _SENTENCE_RE.split(context.strip()):
        overlap = len(q_tokens & set(normalize_answer(sentence).split()))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best


def lexical_overlap_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A bag-of-words reader with no model: the best-overlapping sentence as the whole answer."""
    predictions = [lexical_overlap_answer(r["question"], r["context"]) for r in records]
    result = qa_metrics(predictions, _golds(records))
    result["baseline"] = "lexical overlap (passage sentence sharing the most question tokens)"
    return result
