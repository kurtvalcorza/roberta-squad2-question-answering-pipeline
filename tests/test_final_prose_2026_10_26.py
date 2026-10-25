"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs.
roberta: Section 9 intro and Interpretation.
Fails on PR head 2bc7518.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/roberta_question_answering_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = [
    'expect most of the gain',
    'Almost all of that F1 rise is null',
    'On this corpus the fine-tune taught',
]
PRESENT = [
    'In those runs almost all of that F1 rise was null suppression',
    'so abstaining never earns credit on it',
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
