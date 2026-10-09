# Fleet-sweep fixes: `roberta_question_answering_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `d9e0d1c`. Changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`) and, for SWP-F, in the carried `pipeline.py`; the notebook is regenerated. Status and release
labels are unchanged.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules. Generator → the fleet's shared `build_notebook.py/2.2`; the template opts in. One kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (47 packages compiled from the unchanged pyproject pins, `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1; generator, template, validator, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 1 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4–8), What to notice + Check your reasoning after Sections 4–9 quoting the recorded Kaggle T4 run of 2026-09-19 (F1 always-null 0.25 / lexical 8.58 / frozen 12.38 / adapted 25.81; exact match 8 → 16.25; answered rate 43.5 % → 100 %; reload parity 8 of 8), Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. Literal `{{…}}` in the BYOD statement and the Prerequisites now render as `{…}`. | opening, Sections 4–9 markdown, closing, header, Prerequisites | `test_swp_g_*` (2 tests) |
| SWP-A (quality asserts) | Fixed | Confirmed (2 flagged): Section 6 asserted `frozen F1 > always-null F1` and Section 8 `adapted F1 > frozen F1`; either aborts a BYOD run before export. Both are recorded verdicts now (evaluation report and `result.json`); the reload-parity assert is kept. | Sections 6, 8, 9 | `test_swp_a_*` (5 tests) |
| SWP-F (frozen re-run) | Fixed | Confirmed: `adapt()` trained the last encoder blocks and the span head in place from whatever weights the model held, so a Section 7 re-run (the closing's optional experiments) continued training while epoch 0 was labelled "frozen model". `pipeline.py` now applies siglip-v1's `restore_base` pattern (dataclass field `_base_state`); a failed adapt leaves the weights as before; the result records `started_from` (printed in Section 7). | `src/roberta_question_answering_pipeline/pipeline.py`, Section 7 | `test_swp_f_*` (2 tests) |
| SWP-B (BYOD upload only) | Fixed | Confirmed: BYOD used only `files.upload()`. Added `BYOD_PATH` (one .json/.jsonl/.csv file; Kaggle/Jupyter); guarded upload fallback (off Colab, cancelled, multi-file, wrong extension each name the file or rule). | Section 4 | `test_swp_b_*` (3 tests) |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes several minutes; reused afterwards). Linux x86_64 only.
- `RoBERTaQuestionAnsweringPipeline.adapt()` and `load_artifact()` always start from the pinned base; new `restore_base()`; `adapt()` result has `started_from`.
- New `BYOD_PATH` field; Sections 6 and 8 print verdicts instead of raising; the evaluation report and `result.json` carry `verdict`.
- Guided-layer cells; infrastructure collapsed.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable). The Section 1 cell runs for real against a stand-in environment; the BYOD block and Sections 6 and 8 run with stand-ins; `restore_base` runs on a NumPy stand-in model. Plumbing evidence, not model evidence.
- CI installs CPU torch from `download.pytorch.org`, blocked here; torch was not installed. `pytest --continue-on-collection-errors` with the other CI pins: 47 passed + 1 torch-only collection error before → 64 passed + the same error after.
- `build_notebook.py --check` up to date; `validate_release_assets.py` PASS; `ruff check src tests tools` clean.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9, quality asserts 0.

## Remaining gates

- CI's torch-backed test module on the PR.
- A hosted **Run all in one pass** in a fresh Colab runtime (no restart expected), then a re-run of the Section 9 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with `BYOD_PATH`).
- A full Notebook Review Framework v1 review has not been done.
