"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both from the generated notebook:
- the router (install cell, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as
  one line, while every other exception keeps its traceback;
- the BYOD cell turns every user-input problem into `BYODRefusal` and removes the earlier run's
  results before it stops.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "roberta_question_answering_colab.ipynb"
BYOD_CELL_ID = "roberta_question_answering-16"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _router_cell(cells: list[dict]) -> str:
    return next(_source(c) for c in cells if '_WORKER_SOURCE = r"""' in _source(c))


def _router_namespace(cells: list[dict], start: str) -> dict:
    source = _router_cell(cells)
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(cells, messages):
    namespace = _router_namespace(cells, "class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def test_router_relays_a_refusal_as_one_line_without_a_traceback(cells, capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime(cells, [("refusal", "etc: not accepted")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: etc: not accepted"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(cells, capsys):
    namespace, runtime = _kernel_runtime(cells, [("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(cells, capsys):
    _, runtime = _kernel_runtime(cells, [("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")
    assert runtime._send.sent == [("run", "print('hello')\n")]


def test_router_has_the_refusal_channel_before_the_error_branch(cells):
    source = _router_cell(cells)
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    assert "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/." in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


# --- router, end to end with a real worker (POSIX: the router uses pass_fds and start_new_session) ----------------


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(cells, capsys):
    namespace = _router_namespace(cells, '_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('etc: not accepted')\n")
        assert caught.value._render_traceback_() == ["Refused: etc: not accepted"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


# --- BYOD cell ----------------------------------------------------------------------------------------------------

CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _byod_cell(cells: list[dict]) -> dict:
    (cell,) = [c for c in cells if c.get("id") == BYOD_CELL_ID]
    return cell


def test_byod_refusal_is_carried_verbatim_and_the_cell_id_is_stable(cells):
    assert CARRIED in _source(_byod_cell(cells))


def _one_line(exc: BaseException, file_name: str, rule: str) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert line.startswith(f"Refused: {file_name}: "), line
    assert rule in line and "\n" not in line and "Traceback" not in line, line
    assert "again" in line, line
    return line


def _colab(monkeypatch, uploaded):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


# --- roberta: Section 4 BYOD cell ---------------------------------------------------------------------------------

import roberta_question_answering_pipeline as pkg  # noqa: E402

STEM = "roberta_question_answering"
EARLIER = {"splits": "old", "records": "old", "test_records": "old", "adapted_test": "old", "comparison": "old", "new_results": "old", "result_payload": "old"}


def _record(i: int, rid: str | None = None) -> dict:
    context = f"Passage {i} says the tower stands in Paris near river number {i}."
    return {"id": rid or f"r{i:03d}", "question": f"Where is tower {i}?", "context": context, "answers": [{"text": "Paris", "answer_start": context.index("Paris")}]}


def _cell_code(cells, byod_path: str | None) -> str:
    source = _source(_byod_cell(cells))
    if byod_path is None:
        return source
    return source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)


def _namespace(tmp_path: Path) -> dict:
    outputs = tmp_path / "outputs"
    (outputs / f"{STEM}_adapter").mkdir(parents=True, exist_ok=True)
    (outputs / f"{STEM}_adapter" / "manifest.json").write_text("{}", encoding="utf-8")
    (outputs / f"{STEM}_result.json").write_text("{}", encoding="utf-8")
    (outputs / "other_notebook_result.json").write_text("{}", encoding="utf-8")
    namespace = {name: getattr(pkg, name) for name in dir(pkg) if not name.startswith("_")}
    namespace.update({"os": os, "Path": Path, "__name__": "__main__", "pipe": "kept", **EARLIER})
    return namespace


def _refused(cells, tmp_path, byod_path: str, file_name: str, rule: str) -> dict:
    ns = _namespace(tmp_path)
    with pytest.raises(ValueError) as caught:
        exec(compile(_cell_code(cells, byod_path), "<section 4>", "exec"), ns)
    _one_line(caught.value, file_name, rule)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert not list((tmp_path / "outputs").glob(f"{STEM}_*")), "this notebook's earlier exports survived the refusal"
    assert (tmp_path / "outputs" / "other_notebook_result.json").exists() and ns["pipe"] == "kept"
    return ns


def test_scripted_refusal_colab_sample_csv(cells, tmp_path, monkeypatch):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/content/sample_data/california_housing_test.csv' (present in every fresh Colab runtime)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    data = tmp_path / "california_housing_test.csv"
    data.write_text("longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value\n-122.05,37.37,27.0,3885.0,661.0,1537.0,606.0,6.6085,344700.0\n", encoding="utf-8")
    _refused(cells, tmp_path, str(data), "california_housing_test.csv", "CSV is missing columns ['answer_start', 'answer_text', 'context', 'id', 'question']")


@pytest.mark.parametrize(
    ("name", "content", "rule"),
    [
        ("missing.csv", None, "is not a file"),
        ("notes.txt", "hello", "BYOD datasets must be .csv, .json or .jsonl"),
        ("broken.json", "{not json", "Expecting property name"),
        ("object.json", '{"rows": []}', "JSON dataset must be an array"),
        ("few.json", json.dumps([_record(i) for i in range(5)]), "5"),
        ("dupes.json", json.dumps([_record(i, rid="same") for i in range(30)]), "same"),
    ],
    ids=["missing path", "wrong suffix", "broken json", "not a list", "too few records", "duplicate ids"],
)
def test_byod_file_problems_are_one_refused_line(cells, tmp_path, monkeypatch, name, content, rule):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    if content is not None:
        (tmp_path / name).write_text(content, encoding="utf-8")
    _refused(cells, tmp_path, str(tmp_path / name), name, rule)


@pytest.mark.parametrize(
    ("uploaded", "file_name", "rule"),
    [(None, "(no file)", "received 0"), ({"a.csv": b"", "b.csv": b""}, "(no file)", "received 2"), ({"pairs.xlsx": b""}, "pairs.xlsx", "upload one")],
)
def test_cancelled_or_wrong_upload_is_one_refused_line(cells, tmp_path, monkeypatch, uploaded, file_name, rule):
    monkeypatch.chdir(tmp_path)
    _colab(monkeypatch, uploaded)
    _refused(cells, tmp_path, "", file_name, rule)


def test_upload_off_colab_is_one_refused_line(cells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    _refused(cells, tmp_path, "", "(no file)", "upload dialog exists only in Google Colab")


def test_genuine_internal_error_keeps_its_traceback(cells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    data = tmp_path / "pairs.json"
    data.write_text("[]", encoding="utf-8")
    ns = _namespace(tmp_path)

    def broken(_path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        exec(compile(_cell_code(cells, str(data)), "<section 4>", "exec"), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert ns["comparison"] == "old" and (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is cleared for an internal error


def test_default_path_contract_failure_is_not_a_refusal(cells, tmp_path, monkeypatch):
    """With USE_BYOD off, a pinned sample that fails its contract is a genuine error and keeps its traceback."""
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    ns.update({"fetch_corpus": lambda **k: "corpus", "read_corpus": lambda _p: {"train": []}, "build_sample_dataset": lambda c, seed: {"train": [_record(0)], "validation": [], "test": []}})
    with pytest.raises(ValueError) as caught:
        exec(compile(_cell_code(cells, None), "<section 4>", "exec"), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert ns["comparison"] == "old"
