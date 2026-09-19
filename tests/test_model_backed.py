"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a
referenced evaluation, a one-epoch adaptation of the last encoder block on a dozen pairs, and the
artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

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


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, tmp_path):
    """Without a validation split the recorded policy is "final epoch": the state after the last of three
    epochs is what stays in memory and what the artifact carries."""
    import torch

    result = pipe.adapt(RECORDS[:8], None, epochs=3, trainable_encoder_layers=1, batch_size=4)
    assert result["best_epoch"] == 3 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 4
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = RoBERTaQuestionAnsweringPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 3 and reloaded.adapter["trainable_encoder_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, tmp_path):
    """The loader derives the exact tensor set from the recorded layer count: a manifest listing fewer,
    more or other tensors — or one whose safetensors payload differs from its list — is refused."""
    import json as _json
    import shutil

    from safetensors.torch import load_file, save_file

    pipe.adapt(RECORDS[:8], None, epochs=1, trainable_encoder_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = _json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(_json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        RoBERTaQuestionAnsweringPipeline.from_artifact(fewer, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    size = (extra / "adapter.safetensors").stat().st_size
    files = [{**manifest["files"][0], "bytes": size, "sha256": digest}]
    (extra / "manifest.json").write_text(_json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        RoBERTaQuestionAnsweringPipeline.from_artifact(extra, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_encoder_layers": 2}
    (other_layers / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        RoBERTaQuestionAnsweringPipeline.from_artifact(other_layers, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    """A failure inside training leaves the base exactly as it was, frozen, with no adapter attached."""
    import torch

    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(RECORDS[:8], None, epochs=2, trainable_encoder_layers=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())
