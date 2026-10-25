"""Prose numbers stay true across every recorded hosted run (Kurt, 2026-10-25: point to output + ranges).

Hosted runs: Colab T4 blob `35bae728` (2026-10-09, 228.6 s), Colab T4 blob `f8a79e6e` (2026-10-25, 169.3 s),
Kaggle T4 of an earlier blob (2026-09-19, 306 s). Timings are approximate with the runtime named.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "roberta_question_answering_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_timings_are_approximate_and_name_the_runtime():
    md = _markdown()
    for exact in ("228.6 s", "169.3 s", "306 s"):
        assert exact not in md, exact
    assert md.count("about 3–4 minutes") == 2 and "`f8a79e6e`" in md
