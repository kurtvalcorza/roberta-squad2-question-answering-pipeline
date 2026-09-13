import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from roberta_question_answering_pipeline import (
    DECISION_RULE,
    DEFAULT_MAX_ANSWER_TOKENS,
    DEFAULT_WEIGHTS_DIR,
    MAX_ANSWER_TOKENS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TOKENS,
    MAX_QUESTION_CHARS,
    MAX_QUESTION_TOKENS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    RoBERTaQuestionAnsweringPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
QUESTION = "Where is the tower?"
CONTEXT = "The tower stands in Paris near the river."


def _fake_count(text: str) -> int:
    return len(text.split())  # one token per word, stands in for the BPE count


def _make_runner(peak: str | None, null_logit: float = 0.0):
    """Word-level fake encoder: <s> q... </s></s> c... </s>. ``peak`` names the context word(s) to favour;
    ``None`` favours <s> so the null answer wins."""

    def runner(question: str, context: str):
        q_words, c_words = question.split(), context.split()
        n = 1 + len(q_words) + 2 + len(c_words) + 1
        start = np.full(n, -5.0)
        end = np.full(n, -5.0)
        offsets = np.zeros((n, 2), dtype=int)
        context_mask = np.zeros(n, dtype=bool)
        first_context = 1 + len(q_words) + 2
        cursor = 0
        for i, word in enumerate(c_words):
            pos = first_context + i
            begin = context.index(word, cursor)
            offsets[pos] = (begin, begin + len(word))
            cursor = begin + len(word)
            context_mask[pos] = True
        start[0] = end[0] = null_logit
        if peak is not None:
            words = peak.split()
            first = c_words.index(words[0])
            start[first_context + first] = 5.0
            end[first_context + first + len(words) - 1] = 5.0
        return start, end, offsets, context_mask

    return runner


def _pipeline(peak: str | None = "Paris", null_logit: float = 0.0) -> RoBERTaQuestionAnsweringPipeline:
    return RoBERTaQuestionAnsweringPipeline(_make_runner(peak, null_logit), _fake_count, "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "deepset/roberta-base-squad2"
    assert MODEL_LICENSE == "cc-by-4.0"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert (MAX_QUESTION_TOKENS, MAX_CONTEXT_TOKENS) == (64, 384)
    assert MAX_QUESTION_TOKENS + MAX_CONTEXT_TOKENS + 4 <= 512  # one pass always fits model_max_length
    assert 1 <= DEFAULT_MAX_ANSWER_TOKENS <= MAX_ANSWER_TOKENS


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_ceilings_fit_local_config_when_present():
    config_path = DEFAULT_WEIGHTS_DIR / "config.json"
    tok_path = DEFAULT_WEIGHTS_DIR / "tokenizer_config.json"
    if not (config_path.is_file() and tok_path.is_file()):
        pytest.skip("local snapshot config not staged")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    assert config["architectures"] == ["RobertaForQuestionAnswering"]
    assert (
        MAX_QUESTION_TOKENS + MAX_CONTEXT_TOKENS + 4
        <= tok["model_max_length"]
        <= config["max_position_embeddings"]
    )


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_bytes_and_missing_file(tmp_path: Path):
    _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        RoBERTaQuestionAnsweringPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path: Path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path: Path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_answer_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError, match="question must be str"):
        pipe.answer(b"bytes", CONTEXT)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="context must be str"):
        pipe.answer(QUESTION, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="question is empty"):
        pipe.answer("   ", CONTEXT)
    with pytest.raises(ValueError, match="context is empty"):
        pipe.answer(QUESTION, "")
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        pipe.answer("x" * (MAX_QUESTION_CHARS + 1), CONTEXT)
    with pytest.raises(ValueError, match="MAX_CONTEXT_CHARS"):
        pipe.answer(QUESTION, "x" * (MAX_CONTEXT_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_QUESTION_TOKENS"):
        pipe.answer(" ".join(["w"] * (MAX_QUESTION_TOKENS + 1)), CONTEXT)
    with pytest.raises(ValueError, match="MAX_CONTEXT_TOKENS"):
        pipe.answer(QUESTION, " ".join(["w"] * (MAX_CONTEXT_TOKENS + 1)))
    with pytest.raises(ValueError, match="max_answer_tokens"):
        pipe.answer(QUESTION, CONTEXT, max_answer_tokens=0)
    with pytest.raises(ValueError, match="max_answer_tokens"):
        pipe.answer(QUESTION, CONTEXT, max_answer_tokens=MAX_ANSWER_TOKENS + 1)
    with pytest.raises(TypeError):
        pipe.answer(QUESTION, CONTEXT, max_answer_tokens=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pipe.answer(QUESTION, CONTEXT, max_answer_tokens=True)  # type: ignore[arg-type]


def test_answer_extracts_span_with_char_offsets():
    pipe = _pipeline("Paris")
    result = pipe.answer(QUESTION, CONTEXT)
    assert result["answer"] == "Paris"
    assert CONTEXT[result["start"] : result["end"]] == "Paris"
    assert result["answerable"] is True
    assert result["score"] == result["best_span_score"] > result["no_answer_score"]
    assert 0.0 < result["score"] <= 1.0
    assert (result["question_tokens"], result["context_tokens"]) == (4, 8)
    assert result["max_answer_tokens"] == DEFAULT_MAX_ANSWER_TOKENS
    assert result["decision_rule"] == DECISION_RULE
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["source"]) == ("cpu", "injected")


def test_answer_returns_empty_when_null_score_wins():
    pipe = _pipeline(peak=None, null_logit=5.0)
    result = pipe.answer(QUESTION, CONTEXT)
    assert result["answer"] == ""
    assert (result["start"], result["end"]) == (0, 0)
    assert result["answerable"] is False
    assert result["score"] == result["no_answer_score"] > result["best_span_score"]


def test_answer_span_length_is_capped_by_max_answer_tokens():
    pipe = _pipeline("in Paris near")  # a three-token span is favoured by the fake logits
    assert pipe.answer(QUESTION, CONTEXT, max_answer_tokens=3)["answer"] == "in Paris near"
    capped = pipe.answer(QUESTION, CONTEXT, max_answer_tokens=2)
    assert capped["answer"] != "in Paris near"
    assert len(capped["answer"].split()) <= 2


def test_answer_accepts_ceiling_boundaries():
    pipe = _pipeline("Paris")
    long_context = " ".join(["Paris"] + ["w"] * (MAX_CONTEXT_TOKENS - 1))
    result = pipe.answer(
        " ".join(["w"] * MAX_QUESTION_TOKENS), long_context, max_answer_tokens=MAX_ANSWER_TOKENS
    )
    assert (result["question_tokens"], result["context_tokens"]) == (MAX_QUESTION_TOKENS, MAX_CONTEXT_TOKENS)
    assert result["answer"] == "Paris"
