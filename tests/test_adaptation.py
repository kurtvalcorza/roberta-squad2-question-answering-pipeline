"""Offline tests for the QA-dataset contract, the pinned corpus reader, the corpus metrics and baselines,
BYOD loaders, CSV export, artifact-manifest rejections and adapt() argument validation. Nothing here
imports torch or transformers; the corpus is a crafted zip served through an injected fetcher."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from roberta_question_answering_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_SHA256,
    ENCODER_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    RoBERTaQuestionAnsweringPipeline,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    fetch_corpus,
    fetch_sample_dataset,
    filter_records,
    lexical_overlap_baseline,
    load_byod_dataset,
    null_baseline,
    qa_metrics,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from roberta_question_answering_pipeline import pipeline as pl
from roberta_question_answering_pipeline import samples as sm

PASSAGES = {
    "Paris": "The tower stands in Paris near the river. It was finished in 1889 by Gustave Eiffel.",
    "Rome": "Rome is the capital of Italy. The Colosseum was completed in 80 AD under Titus.",
    "Cairo": "Cairo lies on the Nile. The Great Pyramid was built for the pharaoh Khufu.",
    "Lima": "Lima is on the Pacific coast. It was founded in 1535 by Francisco Pizarro.",
    "Tokyo": "Tokyo is the capital of Japan. The city hosted the Summer Olympics in 1964.",
    "Delhi": "Delhi sits on the Yamuna river. The Red Fort was commissioned by Shah Jahan.",
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
    ("Tokyo", "What is the capital of Japan?", "Tokyo"),
    ("Tokyo", "When did Tokyo host the Summer Olympics?", "1964"),
    ("Tokyo", "Which games did Tokyo host?", "Summer Olympics"),
    ("Delhi", "Which river does Delhi sit on?", "Yamuna"),
    ("Delhi", "Who commissioned the Red Fort?", "Shah Jahan"),
    ("Delhi", "What did Shah Jahan commission?", "Red Fort"),
]


def _records(questions=QUESTIONS, prefix="r"):
    out = []
    for i, (title, question, answer) in enumerate(questions):
        context = PASSAGES[title]
        out.append(
            {
                "id": f"{prefix}{i:03d}",
                "question": question,
                "context": context,
                "answers": [{"text": answer, "answer_start": context.index(answer)}],
                "title": title,
            }
        )
    return out


def _squad(records):
    articles: dict[str, dict] = {}
    for r in records:
        art = articles.setdefault(r["title"], {"title": r["title"], "paragraphs": []})
        para = next((p for p in art["paragraphs"] if p["context"] == r["context"]), None)
        if para is None:
            para = {"context": r["context"], "qas": []}
            art["paragraphs"].append(para)
        para["qas"].append({"id": r["id"], "question": r["question"], "answers": r["answers"]})
    return {"version": "test", "data": list(articles.values())}


def _corpus_zip(train=None, dev=None):
    train = _records(prefix="t") if train is None else train
    dev = _records(prefix="d") if dev is None else dev
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("README.md", "# AdversarialQA\n")
        archive.writestr("3_droberta/train.json", json.dumps(_squad(train)))
        archive.writestr("3_droberta/dev.json", json.dumps(_squad(dev)))
    return buffer.getvalue()


def _fake_count(text: str) -> int:
    return len(text.split())


def _echo_runner(question: str, context: str):
    """Word-level fake reader that always favours the first context word."""
    q, c = question.split(), context.split()
    n = 1 + len(q) + 2 + len(c) + 1
    start = np.full(n, -5.0)
    end = np.full(n, -5.0)
    first = 1 + len(q) + 2
    start[first] = end[first] = 5.0
    offsets = np.zeros((n, 2), dtype=int)
    pos = 0
    for i, word in enumerate(c):
        offsets[first + i] = (pos, pos + len(word))
        pos += len(word) + 1
    mask = np.zeros(n, dtype=bool)
    mask[first : first + len(c)] = True
    return start, end, offsets, mask


def _pipeline_without_model():
    return RoBERTaQuestionAnsweringPipeline(_echo_runner, _fake_count)


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_URL == "https://adversarialqa.github.io/data/aqa_v1.0.zip"
    assert len(CORPUS_SHA256) == 64 and sm.CORPUS_BYTES == 9_018_914
    assert sum(SAMPLE_SPLIT.values()) == 1_600 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_digest_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    payload = _corpus_zip()
    monkeypatch.setattr(sm, "CORPUS_BYTES", len(payload))
    monkeypatch.setattr(sm, "CORPUS_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def fetcher(url):
        calls.append(url)
        return payload

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == payload
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == payload
    assert calls == [sm.CORPUS_URL]
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_flattens_squad_and_checks_counts(monkeypatch, forbid_model_imports):
    monkeypatch.setattr(sm, "CORPUS_QUESTIONS", {"train": 18, "dev": 18})
    corpus = read_corpus(_corpus_zip())
    assert len(corpus["train"]) == 18 and corpus["train"][0]["id"] == "train-t000"
    assert corpus["dev"][3]["title"] == "Rome" and corpus["dev"][3]["answers"][0]["text"] == "Rome"
    monkeypatch.setattr(sm, "CORPUS_QUESTIONS", {"train": 99, "dev": 18})
    with pytest.raises(ValueError, match="expected 99"):
        read_corpus(_corpus_zip())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("3_droberta/train.json", "{}")
    with pytest.raises(ValueError, match="missing member"):
        read_corpus(buffer.getvalue())


def test_filter_and_article_split_are_seeded_and_disjoint(monkeypatch, forbid_model_imports):
    records = _records()
    noisy = [
        *records,
        {**records[0], "id": "dup", "question": records[0]["question"].upper()},  # duplicate pair
        {**records[1], "id": "bad", "answers": [{"text": "nowhere", "answer_start": 0}]},  # offset wrong
        {**records[2], "id": "none", "answers": []},  # unanswerable
        {**records[3], "id": "long", "context": "x " * 800},  # over the sample filter
    ]
    assert [r["id"] for r in filter_records(noisy)] == [r["id"] for r in records]
    sizes = {"train": 6, "validation": 3, "test": 3}
    corpus = {"train": _records(prefix="t"), "dev": records}
    splits = build_sample_dataset(corpus, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert splits["train"][0]["id"] == "train-0000"
    val_titles = {r["title"] for r in splits["validation"]}
    test_titles = {r["title"] for r in splits["test"]}
    assert not (val_titles & test_titles)
    assert build_sample_dataset(corpus, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(corpus, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 100, "validation": 1, "test": 1})
    with pytest.raises(ValueError, match="dev pool"):
        build_sample_dataset(corpus, sizes={"train": 1, "validation": 12, "test": 9})


def test_check_split_disjoint_catches_a_shared_passage(forbid_model_imports):
    records = _records()
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:2], "test": records[2:3]})
    assert check_split_disjoint({"train": records[:3], "test": records[3:6]}) == {"train": 3, "test": 3}


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    payload = _corpus_zip()
    monkeypatch.setattr(sm, "CORPUS_BYTES", len(payload))
    monkeypatch.setattr(sm, "CORPUS_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(sm, "CORPUS_QUESTIONS", {"train": 18, "dev": 18})
    splits = fetch_sample_dataset(
        cache_dir=tmp_path, fetcher=lambda url: payload, sizes={"train": 8, "validation": 3, "test": 3}
    )
    assert validate_dataset(splits["train"])["n_records"] == 8


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    report = validate_dataset(_records())
    assert report["n_records"] == 18 and report["unique_contexts"] == 6 and report["unanswerable"] == 0
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    assert report["records"][0]["title"] == "Paris"
    good = _records()
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "question": ""}, *good[1:]], "question is empty"),
        ([{**good[0], "context": "x" * 3_001}, *good[1:]], "MAX_CONTEXT_CHARS"),
        ([{**good[0], "answers": "Paris"}, *good[1:]], "answers must be a list"),
        ([{**good[0], "answers": [{"text": "Paris"}]}, *good[1:]], "text and answer_start"),
        ([{**good[0], "answers": [{"text": "Paris", "answer_start": 3}]}, *good[1:]], "not found at offset"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)
    unanswerable = validate_dataset([{**r, "answers": []} for r in good])
    assert unanswerable["unanswerable"] == 18


def test_split_dataset_keeps_passages_together_and_is_seeded(forbid_model_imports):
    records = _records()
    splits = split_dataset(records, val_fraction=0.15, test_fraction=0.15, seed=3)
    assert sum(len(v) for v in splits.values()) == 18 and len(splits["train"]) == 12
    assert check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.15, test_fraction=0.15, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)


# --- metrics and baselines ----------------------------------------------------------------------------


def test_qa_metrics_and_baselines(forbid_model_imports):
    records = _records()
    golds = [[a["text"] for a in r["answers"]] for r in records]
    perfect = qa_metrics([g[0] for g in golds], golds)
    assert perfect["exact_match"] == 100.0 and perfect["f1"] == 100.0 and perfect["answered_rate"] == 100.0
    empty = qa_metrics([""] * 18, golds)
    assert empty["exact_match"] == 0.0 and empty["unanswerable_gold"] == 0
    with pytest.raises(ValueError, match="gold lists"):
        qa_metrics(["a"], [["a"], ["b"]])
    null = null_baseline([{**r, "answers": []} for r in records[:4]] + records[4:])
    assert null["f1"] == pytest.approx(100.0 * 4 / 18) and "empty" in null["baseline"]
    lexical = lexical_overlap_baseline(records)
    assert 0.0 < lexical["f1"] < 100.0 and lexical["exact_match"] == 0.0


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_csv_json_jsonl_squad_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = [{k: v for k, v in r.items() if k != "title"} for r in _records()]
    csv_path = write_dataset_csv(records, tmp_path / "data.csv")
    assert load_byod_dataset(csv_path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "squad.json").write_text(json.dumps(_squad(_records())), encoding="utf-8")
    assert [r["id"] for r in load_byod_dataset(tmp_path / "squad.json")] == [r["id"] for r in records]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.jsonl") == records
    (tmp_path / "bad.csv").write_text("id,question\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv.bak").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="csv, .json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv.bak")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.csv")
    unanswerable = write_dataset_csv([{**records[0], "answers": []}], tmp_path / "null.csv")
    assert load_byod_dataset(unanswerable)[0]["answers"] == []


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(_records(), batch_size=0)
    with pytest.raises(ValueError, match="trainable_encoder_layers"):
        pipe.adapt(_records(), trainable_encoder_layers=ENCODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    # evaluate() and check_fit() only need the answer path, so they work with an injected runner
    metrics = pipe.evaluate(_records())
    assert metrics["n"] == 18 and metrics["verdict"] == "measured-small-sample"
    assert metrics["adapted"] is False
    assert metrics["answered_rate"] == 100.0 and 0.0 <= metrics["f1"] <= 100.0
    huge = {
        **_records()[0],
        "id": "huge",
        "context": "Paris " * 400,
        "answers": [{"text": "Paris", "answer_start": 0}],
    }
    fit = pipe.check_fit([*_records(), huge])
    assert fit["n_fitting"] == 18 and fit["dropped"] == ["huge"]


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["qa_outputs.weight"],
        "adapter": {"trainable_encoder_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_encoder_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_encoder_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
