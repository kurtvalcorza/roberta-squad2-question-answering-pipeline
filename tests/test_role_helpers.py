"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from roberta_question_answering_pipeline import (
    DECISION_RULE,
    DEFAULT_MAX_ANSWER_TOKENS,
    INPUT_SCHEMA,
    MAX_ANSWER_TOKENS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TOKENS,
    MAX_QUESTION_CHARS,
    MAX_QUESTION_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    exact_match,
    f1,
    normalize_answer,
    validate_inputs,
)

QUESTION = "Where is the tower?"
CONTEXT = "The tower stands in Paris near the river."
UNANSWERABLE = "Who built the river?"


def _result(answer: str = "Paris", score: float = 0.9, null: float = 0.05) -> dict:
    return {
        "answer": answer,
        "score": score,
        "start": CONTEXT.find(answer) if answer else 0,
        "end": CONTEXT.find(answer) + len(answer) if answer else 0,
        "no_answer_score": null,
        "best_span_score": score if answer else 0.01,
        "answerable": bool(answer),
        "question_tokens": 5,
        "context_tokens": 10,
        "max_answer_tokens": DEFAULT_MAX_ANSWER_TOKENS,
        "decision_rule": DECISION_RULE,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([QUESTION, UNANSWERABLE], [CONTEXT, CONTEXT], names=["pair00", "pair01"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["question_chars"] == [1, MAX_QUESTION_CHARS]
    assert manifest["schema"]["context_chars"] == [1, MAX_CONTEXT_CHARS]
    assert manifest["schema"]["question_tokens"] == [1, MAX_QUESTION_TOKENS]
    assert manifest["schema"]["context_tokens"] == [1, MAX_CONTEXT_TOKENS]
    assert manifest["schema"]["max_answer_tokens"] == [1, MAX_ANSWER_TOKENS]
    assert manifest["schema"]["decision_rule"] == DECISION_RULE
    assert manifest["inputs"] == [
        {"id": "pair00", "question_chars": len(QUESTION), "context_chars": len(CONTEXT)},
        {"id": "pair01", "question_chars": len(UNANSWERABLE), "context_chars": len(CONTEXT)},
    ]
    assert manifest["max_answer_tokens"] == DEFAULT_MAX_ANSWER_TOKENS
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_settings() -> None:
    manifest = validate_inputs([QUESTION], [CONTEXT], max_answer_tokens=30)
    assert [entry["id"] for entry in manifest["inputs"]] == ["pair00"]
    assert manifest["max_answer_tokens"] == 30


def test_validate_inputs_rejects_like_answer() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(QUESTION, [CONTEXT])
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs([QUESTION], CONTEXT)
    with pytest.raises(ValueError, match="at least one item"):
        validate_inputs([], [])
    with pytest.raises(ValueError, match="same length"):
        validate_inputs([QUESTION], [CONTEXT, CONTEXT])
    with pytest.raises(TypeError, match="question must be str"):
        validate_inputs([None], [CONTEXT])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="context is empty"):
        validate_inputs([QUESTION], ["   "])
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        validate_inputs(["x" * (MAX_QUESTION_CHARS + 1)], [CONTEXT])
    with pytest.raises(ValueError, match="MAX_CONTEXT_CHARS"):
        validate_inputs([QUESTION], ["x" * (MAX_CONTEXT_CHARS + 1)])
    with pytest.raises(TypeError, match="max_answer_tokens must be an int"):
        validate_inputs([QUESTION], [CONTEXT], max_answer_tokens=True)
    with pytest.raises(ValueError, match="max_answer_tokens must be between"):
        validate_inputs([QUESTION], [CONTEXT], max_answer_tokens=MAX_ANSWER_TOKENS + 1)
    with pytest.raises(ValueError, match="names must have one entry per pair"):
        validate_inputs([QUESTION], [CONTEXT], names=["a", "b"])


def test_metric_helpers_follow_squad_normalisation() -> None:
    assert normalize_answer("The Eiffel Tower, in Paris!") == "eiffel tower in paris"
    assert exact_match("the Paris", ["Paris"]) == 1.0
    assert exact_match("London", ["Paris", "the city of Paris"]) == 0.0
    assert exact_match("", []) == 1.0 and exact_match("Paris", []) == 0.0
    assert f1("Paris near the river", ["the river"]) == pytest.approx(0.5)  # 3 tokens vs 1, 1 shared
    assert f1("", [""]) == 1.0 and f1("", ["Paris"]) == 0.0 and f1("London", ["Paris"]) == 0.0
    assert f1("in Paris", ["Paris", "London"]) == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_evaluation_report_is_not_measurable_without_gold() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["answerable"] is True
    assert report["sample_kind"] == "synthetic"
    assert "no gold answer" in report["reason"]
    assert "exact_match and f1" in report["needs"]
    assert report["decision_rule"] == DECISION_RULE
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_is_sample_sanity_with_gold() -> None:
    report = evaluation_report(_result("Paris"), ["Paris"], sample_kind="BYOD upload")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "BYOD upload"
    assert [m["id"] for m in report["metrics"]] == ["exact_match", "f1"]
    assert [m["value"] for m in report["metrics"]] == [1.0, 1.0]
    assert "single pair" in report["reason"]
    unanswerable = evaluation_report(_result("", 0.8, 0.8), [])
    assert unanswerable["answerable"] is False
    assert [m["value"] for m in unanswerable["metrics"]] == [1.0, 1.0]
    wrong = evaluation_report(_result("Paris"), [])
    assert [m["value"] for m in wrong["metrics"]] == [0.0, 0.0]
