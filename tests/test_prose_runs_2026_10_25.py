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
    # Historical wording (2026-10-26): exact recorded times are allowed again, each labelled with its run.
    assert md.count("228.6 s (blob `35bae728`, 9 October 2026), 169.3 s") == 2 and "`f8a79e6e`" in md


def test_historical_wording_records_every_hosted_run():
    # Historical wording (2026-10-26): timings and claims are records of the named hosted runs,
    # never expectations; no run-status sentence. Fails on PR #10 head c1ca9d9.
    md = _markdown()
    for gone in ("about 3–4 minutes", "none of these is a run of this revision", "Expect the frozen F1",
                 "Watch validation F1 roughly double", "Expect the answered rate", "is unmeasured"):
        assert gone not in md, gone
    for quoted in ("165.3 s (`2ab5ce3c`, 25 October 2026)", "`2ab5ce3c` (commit `08e1cd1`, 25 October 2026)",
                   "your run may take a different time", "this can vary between runs"):
        assert quoted in md, quoted
