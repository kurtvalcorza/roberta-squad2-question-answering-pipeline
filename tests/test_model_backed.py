"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a
referenced evaluation, a one-epoch adaptation of the last encoder block on a dozen pairs, and the
artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from roberta_question_answering_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    WEIGHT_FILE,
    RoBERTaQuestionAnsweringPipeline,
)

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

PASSAGES = {
    "Paris": "The tower stands in Paris near the river. It was finished in 1889 by Gustave Eiffel.",
    "Rome": "Rome is the capital of Italy. The Colosseum was completed in 80 AD under Titus.",
    "Cairo": "Cairo lies on the Nile. The Great Pyramid was built for the pharaoh Khufu.",
    "Lima": "Lima is on the Pacific coast. It was founded in 1535 by Francisco Pizarro.",
}
QUESTIONS = [
    ("Paris", "Where does the tower stand?", "Paris"),
    ("Paris", "Who finished the tower?", "Gustave Eiffel"),
    ("Paris", "When was the tower finished?", "1889"),
    ("Rome", "What is the capital of Italy?", "Rome"),
    ("Rome", "When was the Colosseum completed?", "80 AD"),
    ("Rome", "Under whom was the Colosseum completed?", "Titus"),
    ("Cairo", "Which river does Cairo lie on?", "Nile"),
    ("Cairo", "For whom was the Great Pyramid built?", "Khufu"),
    ("Cairo", "Where does Cairo lie?", "on the Nile"),
    ("Lima", "Who founded Lima?", "Francisco Pizarro"),
    ("Lima", "When was Lima founded?", "1535"),
    ("Lima", "Which coast is Lima on?", "Pacific"),
]
RECORDS = [
    {
        "id": f"p{i:02d}",
        "question": q,
        "context": PASSAGES[t],
        "answers": [{"text": a, "answer_start": PASSAGES[t].index(a)}],
    }
    for i, (t, q, a) in enumerate(QUESTIONS)
]


@pytest.fixture(scope="module")
def pipe():
    return RoBERTaQuestionAnsweringPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_gold_answers(pipe):
    metrics = pipe.evaluate(RECORDS[:4])
    assert metrics["n"] == 4 and metrics["f1"] > 50.0 and metrics["adapted"] is False


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:8], RECORDS[8:], epochs=1, trainable_encoder_layers=1, batch_size=4)
    assert result["n_trainable"] == 7_089_410 and result["history"][0]["note"] == "frozen model"
    assert all(
        name.startswith("roberta.encoder.layer.11.") or name.startswith("qa_outputs.")
        for name in result["trainable_names"]
    )
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = RoBERTaQuestionAnsweringPipeline.from_artifact(artifact, device="cpu")
    a = [pipe.answer(r["question"], r["context"])["answer"] for r in RECORDS[:3]]
    b = [reloaded.answer(r["question"], r["context"])["answer"] for r in RECORDS[:3]]
    assert a == b
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
