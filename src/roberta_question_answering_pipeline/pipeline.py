"""Extractive question answering with the pinned ``deepset/roberta-base-squad2`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/roberta-base-squad2/``) or,
when explicitly allowed, from the Hugging Face Hub at the pinned revision. One task method, ``answer``:
a question and a context in, one context span (or the SQuAD 2.0 empty "no answer") out, decided with
the same null-vs-span rule the upstream ``transformers`` question-answering pipeline applies.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "deepset/roberta-base-squad2"
MODEL_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
MODEL_LICENSE = "cc-by-4.0"
MODEL_KEY = "roberta-base-squad2"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Ceilings. Hard ceilings, no sliding window: a question/context pair that does not fit is rejected, never
# split or truncated. 64 and 384 are the upstream fine-tuning settings (`max_query_length=64`,
# `max_seq_len=386` with `doc_stride=128` in the pinned README); 64 + 384 + 4 special tokens = 452 <= the
# 512-token `model_max_length` in the pinned tokenizer_config.json, so an accepted pair fits in one pass.
MAX_QUESTION_TOKENS = 64
MAX_CONTEXT_TOKENS = 384
MAX_QUESTION_CHARS = 500  # pre-tokenisation guard; ~4 chars per BPE token on English text
MAX_CONTEXT_CHARS = 3_000
MAX_ANSWER_TOKENS = 64  # ceiling on the span length a caller may request
DEFAULT_MAX_ANSWER_TOKENS = 15  # the upstream pipeline's `max_answer_len` default
NULL_MASK_VALUE = -10000.0  # the upstream pipeline's fill for non-context positions before the softmax
DECISION_RULE = (
    "softmax start and end logits over the context tokens plus <s>; no_answer_score = P_start(<s>) * "
    "P_end(<s>); best span = argmax P_start(i) * P_end(j) over i <= j < i + max_answer_tokens inside the "
    "context; the empty answer is returned when no_answer_score > best span score (upstream "
    "handle_impossible_answer=True rule); scores are products of softmax masses, not calibrated probabilities"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one (question, context) pair of non-empty str; the answer is a span of the context or empty",
    "question_chars": [1, MAX_QUESTION_CHARS],
    "context_chars": [1, MAX_CONTEXT_CHARS],
    "question_tokens": [1, MAX_QUESTION_TOKENS],
    "context_tokens": [1, MAX_CONTEXT_TOKENS],
    "max_answer_tokens": [1, MAX_ANSWER_TOKENS],
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "byte-level BPE (vocab.json + merges.txt, no lower-casing) of the pair "
        "<s> question </s></s> context </s>; hard ceilings, no sliding window: a question over "
        "MAX_QUESTION_TOKENS or a context over MAX_CONTEXT_TOKENS is rejected with a ValueError naming the "
        "count, never truncated or chunked"
    ),
}


def _check_text(text: Any, name: str, max_chars: int) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError(f"{name} is empty")
    if len(text) > max_chars:
        raise ValueError(f"{name} has {len(text)} chars; ceiling is MAX_{name.upper()}_CHARS={max_chars}")
    return text


def _check_inputs(question: Any, context: Any, max_answer_tokens: Any) -> tuple[str, str]:
    """Raise TypeError/ValueError naming the first violated ceiling; return (question, context).

    The token ceilings are not checked here because they need the loaded tokenizer;
    ``_check_token_counts`` applies them inside the pipeline once the counts are known.
    """
    question = _check_text(question, "question", MAX_QUESTION_CHARS)
    context = _check_text(context, "context", MAX_CONTEXT_CHARS)
    if isinstance(max_answer_tokens, bool) or not isinstance(max_answer_tokens, int):
        raise TypeError("max_answer_tokens must be an int")
    if not 1 <= max_answer_tokens <= MAX_ANSWER_TOKENS:
        raise ValueError(
            f"max_answer_tokens must be between 1 and {MAX_ANSWER_TOKENS}, got {max_answer_tokens}"
        )
    return question, context


def _check_token_counts(n_question: int, n_context: int) -> tuple[int, int]:
    """The token ceilings, applied once the tokenizer has counted (no special tokens)."""
    if n_question > MAX_QUESTION_TOKENS:
        raise ValueError(
            f"question is {n_question} tokens; ceiling is MAX_QUESTION_TOKENS={MAX_QUESTION_TOKENS}"
        )
    if n_context > MAX_CONTEXT_TOKENS:
        raise ValueError(f"context is {n_context} tokens; ceiling is MAX_CONTEXT_TOKENS={MAX_CONTEXT_TOKENS}")
    return n_question, n_context


def validate_inputs(
    questions: Sequence[str],
    contexts: Sequence[str],
    *,
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-pair observations, verdict).

    Rejection is reported by raising exactly as ``answer`` would: both route through ``_check_inputs``.
    ``answer`` takes one pair per call, so ``questions``/``contexts`` are the parallel batch the notebook
    will loop over. The token ceilings (``MAX_QUESTION_TOKENS``, ``MAX_CONTEXT_TOKENS``) need the loaded
    tokenizer and are enforced inside ``answer``, which reports both counts in every result.
    """
    for label, value in (("questions", questions), ("contexts", contexts)):
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise TypeError(f"{label} must be a sequence of str, not a single string")
    if not questions:
        raise ValueError("questions must hold at least one item")
    if len(questions) != len(contexts):
        raise ValueError("questions and contexts must have the same length")
    checked = [_check_inputs(q, c, max_answer_tokens) for q, c in zip(questions, contexts, strict=True)]
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per pair")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"pair{i:02d}", "question_chars": len(q), "context_chars": len(c)}
            for i, (q, c) in enumerate(checked)
        ],
        "max_answer_tokens": max_answer_tokens,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


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


def evaluation_report(
    result: Mapping[str, Any], gold_answers: Sequence[str] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report for one ``answer`` result.

    With ``gold_answers`` (the SQuAD convention: a list of acceptable strings; an empty list means the
    question is unanswerable) the report carries ``exact_match`` and ``f1`` for that single pair and the
    verdict ``sample-sanity`` — one pair is a plumbing check, not an accuracy. Without gold the verdict is
    ``not-measurable``.
    """
    prediction = str(result.get("answer", ""))
    supplied = gold_answers is not None
    metrics = []
    if supplied:
        metrics = [
            {
                "id": "exact_match",
                "value": exact_match(prediction, gold_answers),
                "estimation": "single pair",
            },
            {"id": "f1", "value": f1(prediction, gold_answers), "estimation": "single pair"},
        ]
    return {
        "task": "extractive question answering with the SQuAD 2.0 unanswerable case",
        "decision_rule": DECISION_RULE,
        "sample_kind": sample_kind,
        "n_pairs": 1,
        "answerable": bool(prediction),
        "metrics": metrics,
        "baselines": [],
        "verdict": "sample-sanity" if supplied else "not-measurable",
        "reason": (
            "exact_match and f1 are computed for one (question, context, gold) triple with the repository's "
            "SQuAD-style normalisation; a single pair states no dispersion and is not an accuracy"
            if supplied
            else "no gold answer was supplied, so exact_match and f1 cannot be computed"
        ),
        "needs": (
            "gold answer spans (SQuAD 2.0 format: a list of acceptable strings per question, empty for "
            "unanswerable) over enough held-out pairs from the deployment domain to state a dispersion, "
            "scored with the repository's exact_match and f1 helpers"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class RoBERTaQuestionAnsweringPipeline:
    """``_runner(question, context)`` -> ``(start_logits (T,), end_logits (T,), offsets (T, 2),
    context_mask (T,))`` for the encoded pair; ``_count_tokens(text)`` -> BPE token count without special
    tokens. Both injectable so tests run offline."""

    _runner: Callable[[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    _count_tokens: Callable[[str], int]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> RoBERTaQuestionAnsweringPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), dict(local_files_only=True), "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, dict(revision=MODEL_REVISION), "hf-hub"
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoTokenizer, RobertaForQuestionAnswering

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(location, trust_remote_code=False, **kwargs)
        model = RobertaForQuestionAnswering.from_pretrained(
            location, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        def count_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

        def runner(question: str, context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            enc = tokenizer(
                question, context, return_tensors="pt", return_offsets_mapping=True, truncation=False
            )
            offsets = enc.pop("offset_mapping")[0].numpy()
            context_mask = np.array([sid == 1 for sid in enc.sequence_ids(0)], dtype=bool)
            with torch.inference_mode():
                out = model(**enc.to(resolved_device))
            return (
                out.start_logits[0].float().cpu().numpy(),
                out.end_logits[0].float().cpu().numpy(),
                offsets,
                context_mask,
            )

        return cls(runner, count_tokens, resolved_device, source)

    def _validate(self, question: Any, context: Any, max_answer_tokens: Any) -> tuple[int, int]:
        question, context = _check_inputs(question, context, max_answer_tokens)
        return _check_token_counts(self._count_tokens(question), self._count_tokens(context))

    def answer(
        self, question: str, context: str, *, max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    ) -> dict[str, Any]:
        """Extract the best context span or the SQuAD 2.0 empty answer (upstream null-vs-span rule)."""
        n_question, n_context = self._validate(question, context, max_answer_tokens)
        start, end, offsets, context_mask = self._runner(question, context)
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        context_mask = np.asarray(context_mask, dtype=bool)
        offsets = np.asarray(offsets)
        if not (start.shape == end.shape == context_mask.shape) or offsets.shape != (start.shape[0], 2):
            raise RuntimeError(
                f"runner returned inconsistent shapes {start.shape} {end.shape} {offsets.shape}"
            )
        allowed = context_mask.copy()
        allowed[0] = True  # <s> stays eligible so the null answer has a score
        start = np.where(allowed, start, NULL_MASK_VALUE)
        end = np.where(allowed, end, NULL_MASK_VALUE)
        p_start = np.exp(start - start.max())
        p_start /= p_start.sum()
        p_end = np.exp(end - end.max())
        p_end /= p_end.sum()
        no_answer_score = float(p_start[0] * p_end[0])
        p_start[0] = p_end[0] = 0.0
        outer = np.tril(np.triu(np.outer(p_start, p_end)), max_answer_tokens - 1)
        outer[~context_mask, :] = 0.0
        outer[:, ~context_mask] = 0.0
        best_index = int(np.argmax(outer))
        span_start, span_end = np.unravel_index(best_index, outer.shape)
        best_span_score = float(outer[span_start, span_end])
        if no_answer_score > best_span_score:
            answer_text, char_start, char_end, score = "", 0, 0, no_answer_score
        else:
            char_start, char_end = int(offsets[span_start][0]), int(offsets[span_end][1])
            answer_text, score = context[char_start:char_end], best_span_score
        return {
            "answer": answer_text,
            "score": score,
            "start": char_start,
            "end": char_end,
            "no_answer_score": no_answer_score,
            "best_span_score": best_span_score,
            "answerable": bool(answer_text),
            "question_tokens": n_question,
            "context_tokens": n_context,
            "max_answer_tokens": max_answer_tokens,
            "decision_rule": DECISION_RULE,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
