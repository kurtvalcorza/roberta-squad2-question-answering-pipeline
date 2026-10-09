"""Regression tests for the Notebook Review Framework v1 findings on roberta_question_answering_colab.ipynb
(RQA-M2..M6, RQA-m2, RQA-m3, RQA-m5, RQA-S1). RQA-M1, RQA-m1 and RQA-m4 are covered by tests/test_sweep_fixes.py
(SWP-R, SWP-B, SWP-A) and tests/test_worker_colab_stubs.py.

Only CI's dependencies are used: the notebook's own cell sources run against the carried modules with a word-level
fake runner or stand-ins. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import csv
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest

import roberta_question_answering_pipeline as pkg
from roberta_question_answering_pipeline import RoBERTaQuestionAnsweringPipeline, qa_metrics, samples

with contextlib.suppress(ImportError):  # Windows DLL load-order trap: import torch before any NumPy matmul
    import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "roberta_question_answering_colab.ipynb"
TEMPLATE = ROOT / "tools" / "notebook_template.py"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _runner(question: str, context: str):
    """Word-level fake encoder whose null score always beats a moderately peaked span on the word 'Paris'."""
    q_words, c_words = question.split(), context.split()
    n = 1 + len(q_words) + 2 + len(c_words) + 1
    start, end = np.full(n, -5.0), np.full(n, -5.0)
    offsets, mask = np.zeros((n, 2), dtype=int), np.zeros(n, dtype=bool)
    first, cursor = 1 + len(q_words) + 2, 0
    for i, word in enumerate(c_words):
        begin = context.index(word, cursor)
        offsets[first + i], cursor, mask[first + i] = (begin, begin + len(word)), begin + len(word), True
    start[0] = end[0] = 3.0
    if "Paris" in c_words:
        start[first + c_words.index("Paris")] = end[first + c_words.index("Paris")] = 2.0
    return start, end, offsets, mask


def _pipe() -> RoBERTaQuestionAnsweringPipeline:
    return RoBERTaQuestionAnsweringPipeline(_runner, lambda t: len(t.split()), "cpu", "injected")


def _record(i: int, answerable: bool = True, passage: int | None = None) -> dict:
    k = i if passage is None else passage
    context = f"Passage {k} says the tower stands in Paris near river number {k}."
    answers = [{"text": "Paris", "answer_start": context.index("Paris")}] if answerable else []
    return {"id": f"r{i:03d}", "question": f"Where is tower {i}?", "context": context, "answers": answers}


# --- RQA-M2: the forced-span baseline -----------------------------------------------------------------------------


def test_rqa_m2_forced_span_returns_the_best_span_where_the_null_rule_abstains():
    pipe = _pipe()
    record = _record(0)
    default = pipe.answer(record["question"], record["context"])
    forced = pipe.answer(record["question"], record["context"], allow_null=False)
    assert default["answer"] == "" and default["no_answer_score"] > default["best_span_score"]
    assert forced["answer"] == "Paris" and forced["allow_null"] is False and forced["decision_rule"] == pkg.FORCED_SPAN_RULE
    assert forced["no_answer_score"] == default["no_answer_score"]
    with pytest.raises(ValueError, match="allow_null must be a bool"):
        pipe.answer(record["question"], record["context"], allow_null=0)


def test_rqa_m2_evaluate_scores_the_forced_span_reading_and_f1_on_answered():
    records = [_record(i) for i in range(4)]
    frozen = _pipe().evaluate(records)
    forced = _pipe().evaluate(records, allow_null=False)
    assert (frozen["f1"], frozen["answered_rate"], frozen["f1_answered"], frozen["allow_null"]) == (0.0, 0.0, None, True)
    assert (forced["f1"], forced["answered_rate"], forced["f1_answered"], forced["allow_null"]) == (100.0, 100.0, 100.0, False)
    metrics = qa_metrics(["Paris", "", "London"], [["Paris"], ["Paris"], ["Paris"]])
    assert metrics["f1_answered"] == pytest.approx(50.0) and metrics["f1"] == pytest.approx(100 / 3)


def _m(f1: float, answered: float = 100.0) -> dict:
    return {"exact_match": f1 / 2, "f1": f1, "answered_rate": answered, "n": 4}


@pytest.mark.parametrize(("adapted", "reading"), [(25.81, "within 1 F1 (not distinguishable without a dispersion estimate)"), (30.0, "more than 1 F1 above"), (20.0, "more than 1 F1 below")])
def test_rqa_m2_section_8_splits_the_gain_and_reports_the_forced_span_row(notebook, tmp_path, monkeypatch, capsys, adapted, reading):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = types.SimpleNamespace(evaluate=lambda records, **kw: _m(adapted), answer=lambda q, c, **kw: {"answer": "x"}, adapter={"best_epoch": 1})
    namespace = {
        "json": json, "pipe": pipe, "test_records": [], "val_records": [], "ANSWER_MAX_TOKENS": 15, "gold_texts": lambda r: [],
        "baseline_null": _m(0.25, 0.0), "baseline_lexical": _m(8.58), "frozen_test": _m(12.38, 43.5), "frozen_forced_test": _m(25.28),
        "frozen_vs_null": "above", "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "stand-in",
        "dataset_manifests": {}, "disjoint": {}, "fit": {}, "adapt_result": {"history": [], "best_epoch": 1}, "adapt_seconds": 0.0,
    }
    exec(compile(_cell(notebook, "delta_f1 = "), "<section 8>", "exec"), namespace)
    comparison = namespace["comparison"]
    assert comparison["f1"]["frozen_forced_span"] == 25.28 and comparison["answered_rate"]["frozen_forced_span"] == 100.0
    assert comparison["f1_gain_parts"] == {"null_suppression": round(25.28 - 12.38, 2), "better_reading": round(adapted - 25.28, 2)}
    assert comparison["verdict"]["adapted_vs_frozen_forced_span_f1"] == reading
    report = json.loads((tmp_path / "outputs" / "roberta_question_answering_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["frozen_forced_span_test"]["f1"] == 25.28


def test_rqa_m2_prose_no_longer_credits_the_whole_rise_to_reading(notebook):
    text = _markdown(notebook)
    assert "**forced-span baseline**" in text and "null suppression" in text.lower()
    assert "Almost all of that F1 rise is null suppression, not better reading" in text
    assert "it is not evidence that the adapted model reads adversarial questions better" in text
    assert "allow_null=False" in _cell(notebook, "frozen_forced_test = pipe.evaluate(")


# --- RQA-M3: re-runs never mislabel the adapted model as frozen -----------------------------------------------------


class _Recorder(types.SimpleNamespace):
    pass


def test_rqa_m3_section_6_restores_the_pinned_base_before_measuring(notebook, capsys):
    source = _cell(notebook, "frozen_forced_test = pipe.evaluate(")
    calls: list[str] = []

    def restore_base():
        calls.append("restore")
        return ["roberta.encoder.layer.10.output.dense.weight"]

    def evaluate(records, **kw):
        calls.append(f"evaluate allow_null={kw.get('allow_null', True)}")
        return {**_m(12.38, 43.5), "verdict": "measured", "definitions": {}, "f1_answered": 28.4}

    pipe = _Recorder(restore_base=restore_base, evaluate=evaluate, answer=lambda q, c, **kw: {"answer": ""})
    namespace = {"pipe": pipe, "test_records": [], "ANSWER_MAX_TOKENS": 15, "time": __import__("time"), "gold_texts": lambda r: [],
                 "null_baseline": lambda r: _m(0.25, 0.0), "lexical_overlap_baseline": lambda r: _m(8.58)}
    exec(compile(source, "<section 6>", "exec"), namespace)
    assert calls[0] == "restore" and calls[1:] == ["evaluate allow_null=True", "evaluate allow_null=False"]
    assert "restored_pinned_base" in capsys.readouterr().out


@pytest.mark.parametrize("marker", ["delta_f1 = ", "result_payload = {"])
def test_rqa_m3_sections_8_and_9_refuse_the_pinned_base(notebook, marker):
    source = _cell(notebook, marker)
    namespace = {"pipe": types.SimpleNamespace(adapter=None), "USE_BYOD": False}
    with pytest.raises(RuntimeError, match="holds the pinned base"):
        exec(compile(source, "<guard>", "exec"), namespace)


def test_rqa_m3_activity_reruns_from_section_6(notebook):
    text = _markdown(notebook)
    assert "## Activity: does a smaller adapter still beat the forced-span baseline?" in text
    for step in ("1. **Predict.**", "2. **Change.**", "3. **Run.** Run Sections 6, 7, 8 and 9", "4. **Observe.**", "5. **Explain.**"):
        assert step in text


def test_rqa_m3_restore_base_undoes_an_earlier_two_block_adaptation():
    """A real (tiny, CPU) RoBERTa QA model: default adapt, then a layers=1 re-run exports an artifact that reloads onto the
    clean base with identical logits — the probe's 25/40 parity failure before restore_base."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.RobertaConfig(vocab_size=64, hidden_size=16, num_hidden_layers=12, num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    torch.manual_seed(0)
    model = transformers.RobertaForQuestionAnswering(config).eval()
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pipe = RoBERTaQuestionAnsweringPipeline(_runner, lambda t: len(t.split()), "cpu", "injected", _model=model, _tokenizer=object())
    names_two = pipe._trainable_names(2)
    pkg_pipeline = __import__("roberta_question_answering_pipeline.pipeline", fromlist=["_remember_base"])
    pkg_pipeline._remember_base(pipe._base_state, model, names_two)
    with torch.no_grad():  # stand-in for a first adapt(): blocks 10 and 11 and the head move
        for name, param in model.named_parameters():
            if name in names_two:
                param.add_(0.5)
    restored = pipe.restore_base()
    assert sorted(restored) == sorted(names_two)
    assert all(torch.equal(model.state_dict()[k], base[k]) for k in base)


# --- RQA-M4: refusal probes with unanswerable BYOD records ----------------------------------------------------------


def _section_4_namespace(byod_path: str) -> dict:
    namespace = {name: getattr(pkg, name) for name in dir(pkg) if not name.startswith("_")}
    namespace.update({"os": os, "Path": Path, "__name__": "__main__"})
    return namespace


def _section_4(notebook: dict, byod_path: str, seed: int) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    return (source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
            .replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
            .replace("SPLIT_SEED = 42", f"SPLIT_SEED = {seed}", 1))


def _write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "context", "answer_text", "answer_start"])
        writer.writeheader()
        for r in records:
            a = r["answers"][0] if r["answers"] else {"text": "", "answer_start": ""}
            writer.writerow({"id": r["id"], "question": r["question"], "context": r["context"], "answer_text": a["text"], "answer_start": a["answer_start"]})


def test_rqa_m4_one_third_unanswerable_byod_probes_reject_for_every_seed(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "pairs.csv"
    _write_csv(data, [_record(i, answerable=i % 3 != 0) for i in range(60)])
    for seed in range(10):
        exec(compile(_section_4(notebook, str(data), seed), f"<section 4 seed {seed}>", "exec"), _section_4_namespace(str(data)))
        out = capsys.readouterr().out
        assert out.count("'rejected'") == 4 and "'accepted'" not in out, (seed, out[-600:])


def test_rqa_m4_wrong_offset_probe_rejects_when_the_gold_starts_at_offset_0(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    records = []
    for i in range(20):
        context = f"Paris is where tower {i} stands, by river {i}."
        records.append({"id": f"z{i:02d}", "question": f"Where is tower {i}?", "context": context, "answers": [{"text": "Paris", "answer_start": 0}]})
    data = tmp_path / "zero.csv"
    _write_csv(data, records)
    exec(compile(_section_4(notebook, str(data), 0), "<section 4>", "exec"), _section_4_namespace(str(data)))
    out = capsys.readouterr().out
    assert "'probe': 'gold span at the wrong offset', 'rejected'" in out and "'accepted'" not in out


def test_rqa_m4_all_unanswerable_skips_the_wrong_offset_probe_with_a_note(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "none.csv"
    _write_csv(data, [_record(i, answerable=False) for i in range(20)])
    exec(compile(_section_4(notebook, str(data), 0), "<section 4>", "exec"), _section_4_namespace(str(data)))
    out = capsys.readouterr().out
    assert "'skipped': 'no training record has a gold answer'" in out and out.count("'rejected'") == 3


# --- RQA-M5: the stated BYOD minimum passes; a smaller set names its own size ---------------------------------------


def test_rqa_m5_minimum_is_twelve_and_passes_section_4_for_every_seed(notebook, tmp_path, monkeypatch, capsys):
    assert samples.min_split_records() == 12
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "twelve.csv"
    _write_csv(data, [_record(i) for i in range(12)])
    for seed in range(20):
        exec(compile(_section_4(notebook, str(data), seed), "<section 4>", "exec"), _section_4_namespace(str(data)))
        assert "'splits': {'test': 2, 'validation': 2, 'train': 8}" in capsys.readouterr().out


@pytest.mark.parametrize("n", [8, 11])
def test_rqa_m5_too_small_names_the_dataset_size_and_the_minimum(n):
    with pytest.raises(ValueError, match=rf"the dataset has {n} records; at least 12 are required so that 8 stay for training"):
        samples.split_dataset([_record(i) for i in range(n)], seed=0)


def test_rqa_m5_shared_passages_name_every_split_size():
    records = [_record(i, passage=i // 8) for i in range(16)]  # 2 passages of 8 questions
    with pytest.raises(ValueError, match=r"the split of 16 records over 2 passages leaves 0 training records \(test 8, validation 8, train 0\)"):
        samples.split_dataset(records, seed=0)


def test_rqa_m5_prerequisites_state_the_real_minimum(notebook):
    text = _markdown(notebook)
    assert "a BYOD dataset needs **at least 12 records**" in text and "a dataset needs 8..20,000 records" not in text


# --- RQA-m2, RQA-m3, RQA-m5, RQA-S1 -----------------------------------------------------------------------------------


def test_rqa_m2_minor_always_null_prose_matches_0_25(notebook):
    text = _markdown(notebook)
    assert "zero here, which is the point" not in text
    assert "about zero here (0.25 on the default test split" in text


def test_rqa_m3_minor_section_9_states_the_provenance(notebook):
    text = _markdown(notebook)
    assert "dev articles that were in none of the splits" not in text
    assert "their **articles** also appear in the validation and test splits" in text
    source = _cell(notebook, "result_payload = {")
    assert "'new_questions_provenance': new_provenance" in source and "already scored in Section 8" in source


def test_rqa_m5_minor_timings_name_their_environment(notebook):
    text = _markdown(notebook)
    assert "Intel Core Ultra 9 275HX" in text and "Colab's CPU runtime is unmeasured" in text
    assert "about 3.5 minutes per epoch on the local CPU build workstation" in text


def test_rqa_s1_declares_notebook_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert "NOTEBOOK_SPEC 2.0" not in TEMPLATE.read_text(encoding="utf-8")
