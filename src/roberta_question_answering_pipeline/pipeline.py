"""Extractive question answering with the pinned ``deepset/roberta-base-squad2`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/roberta-base-squad2/``) or,
when explicitly allowed, from the Hugging Face Hub at the pinned revision. One task method, ``answer``:
a question and a context in, one context span (or the SQuAD 2.0 empty "no answer") out, decided with
the same null-vs-span rule the upstream ``transformers`` question-answering pipeline applies.

The adaptation contract (``evaluate``, ``adapt``, ``save_artifact``, ``from_artifact``) fine-tunes the last
encoder blocks plus the span head on a validated ``{id, question, context, answers}`` dataset with
validation-F1 epoch selection and exports the trained tensors as a safetensors adapter bound to the pinned
base weights. The inference contract above is unchanged by it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import exact_match, f1, normalize_answer  # noqa: F401 -- re-exported public helpers

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
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 124_056_578
ENCODER_LAYERS = 12  # config.json num_hidden_layers
DEFAULT_TRAINABLE_ENCODER_LAYERS = 2  # the last two encoder blocks plus the span head (14,177,282 parameters)
MAX_EVAL_RECORDS = 2_000
MAX_RECORDS_FIT = 20_000  # check_fit accepts a whole dataset before it is split
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.roberta-base-squad2.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


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
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

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

        return cls(runner, count_tokens, resolved_device, source, _model=model, _tokenizer=tokenizer)

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

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def check_fit(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Split validated records into those within the token ceilings and those `answer` would refuse."""
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_RECORDS_FIT)["records"]
        fitting, dropped = [], []
        for record in checked:
            if (
                self._count_tokens(record["question"]) <= MAX_QUESTION_TOKENS
                and self._count_tokens(record["context"]) <= MAX_CONTEXT_TOKENS
            ):
                fitting.append(record)
            else:
                dropped.append(record["id"])
        return {"fitting": fitting, "dropped": dropped, "n_fitting": len(fitting), "n_dropped": len(dropped)}

    def evaluate(
        self, records: Sequence[Mapping[str, Any]], *, max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    ) -> dict[str, Any]:
        """Answer every record and score the predictions against its gold list (corpus exact-match and F1)."""
        from .metrics import qa_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        predictions = []
        for record in checked:
            predictions.append(
                self.answer(record["question"], record["context"], max_answer_tokens=max_answer_tokens)[
                    "answer"
                ]
            )
        metrics = qa_metrics(predictions, [[a["text"] for a in r["answers"]] for r in checked])
        metrics.update(
            {
                "max_answer_tokens": max_answer_tokens,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_encoder_layers: int) -> list[str]:
        if (
            not isinstance(trainable_encoder_layers, int)
            or not 1 <= trainable_encoder_layers <= ENCODER_LAYERS
        ):
            raise ValueError(f"trainable_encoder_layers must be an int in 1..{ENCODER_LAYERS}")
        model, _ = self._require_model()
        first = ENCODER_LAYERS - trainable_encoder_layers
        prefixes = tuple(f"roberta.encoder.layer.{k}." for k in range(first, ENCODER_LAYERS)) + (
            "qa_outputs.",
        )
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def _span_positions(self, encoded: Any, index: int, record: Mapping[str, Any]) -> tuple[int, int]:
        """Token start/end of the first gold answer inside the context segment; (0, 0) for unanswerable."""
        if not record["answers"]:
            return 0, 0
        answer = record["answers"][0]
        char_start, char_end = int(answer["answer_start"]), int(answer["answer_start"]) + len(answer["text"])
        offsets = encoded["offset_mapping"][index].tolist()
        sequence_ids = encoded.sequence_ids(index)
        start = end = None
        for pos, (sid, (o_start, o_end)) in enumerate(zip(sequence_ids, offsets, strict=True)):
            if sid != 1 or o_end <= o_start:
                continue
            if start is None and o_end > char_start:
                start = pos
            if o_start < char_end:
                end = pos
        if start is None or end is None or end < start:
            raise ValueError(f"record {record['id']}: gold span does not map onto context tokens")
        return start, end

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        lr: float = 3e-5,
        batch_size: int = 16,
        trainable_encoder_layers: int = DEFAULT_TRAINABLE_ENCODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning on a validated QA dataset.

        Only the last `trainable_encoder_layers` encoder blocks and the span head train (2 blocks by
        default: 14,177,282 of 124,056,578 parameters; the embeddings and the earlier blocks stay frozen).
        Start/end cross-entropy on the first gold span (position 0, `<s>`, for an unanswerable record —
        the SQuAD 2.0 convention), AdamW at a fixed learning rate with gradient clipping at 1.0, no
        truncation: every training record must already fit the ceilings (`check_fit`). Epoch 0 records the
        frozen model's validation F1; the epoch with the highest validation F1 is kept."""
        from .samples import validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        names = self._trainable_names(trainable_encoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        for record in train_checked:
            self._validate(record["question"], record["context"], DEFAULT_MAX_ANSWER_TOKENS)
        import torch

        torch.manual_seed(seed)
        model, tokenizer = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked).items()
                if k in ("exact_match", "f1", "n", "answered_rate")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_f1 = entry["val"]["f1"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_checked), generator=generator).tolist()
            losses = []
            for start in range(0, len(order), batch_size):
                batch = [train_checked[i] for i in order[start : start + batch_size]]
                encoded = tokenizer(
                    [r["question"] for r in batch],
                    [r["context"] for r in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_offsets_mapping=True,
                )
                positions = [self._span_positions(encoded, i, r) for i, r in enumerate(batch)]
                encoded.pop("offset_mapping")
                out = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    start_positions=torch.tensor([s for s, _e in positions], device=device),
                    end_positions=torch.tensor([e for _s, e in positions], device=device),
                )
                optimiser.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                losses.append(float(out.loss.detach()))
            model.eval()
            entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
            history.append(entry)
            if progress:
                progress(entry)
            current = entry["val"]["f1"] if entry["val"] else math.inf
            if current > best_f1 or not entry["val"]:
                best_f1 = current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                best_epoch = epoch
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_encoder_layers": trainable_encoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation F1" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted encoder-block and span-head tensors as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest, then overwrite exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not (
                key.startswith("roberta.encoder.layer.") or key.startswith("qa_outputs.")
            ):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable encoder or span-head tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> RoBERTaQuestionAnsweringPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
