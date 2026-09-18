# Release verification

`tutorials/roberta_question_answering_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`metrics.py`, `pipeline.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 7-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `RoBERTaQuestionAnsweringPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_csv`, `pipe.check_fit` per split, `validate_inputs` with the
  `max_answer_tokens` refusal probe, `pipe.answer` with the sanity checks, `null_baseline`,
  `lexical_overlap_baseline`, `pipe.evaluate` on the frozen model and on the validation and test splits after
  adaptation with the F1 assertions, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the unseen
  questions plus the single-pair `evaluation_report`, `pipe.save_artifact`,
  `RoBERTaQuestionAnsweringPipeline.from_artifact` and the reload-parity assertion, and the provenance fields
  `weight_format`, `weight_sha256` and the `corpus` block), the six expected `outputs/` paths, the learner-facing
  statements (CC-BY-4.0 attribution, extractive reader, the upstream null-vs-span rule, the empty answer, adaptation
  with gold spans, products of softmax masses, not a calibrated probability, the two non-neural baselines, no
  dispersion estimate, the snapshot note, the CC BY-SA 3.0 corpus licence, named exclusions) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `RobertaForQuestionAnswering` /
  `AutoTokenizer` / `start_logits` / `from huggingface_hub import` / `urllib.request` / `zipfile.` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `tokenizers`, `huggingface-hub`, `safetensors`
and `numpy`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected encoder, token counter and corpus fetcher, temporary manifests, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/roberta-base-squad2/` or the corpus cache `weights/adversarialqa/` (the standalone path writes
   the manifest itself, stages the missing files from the Hub, and fetches the pinned AdversarialQA zip from the
   project site, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `ANSWER_MAX_TOKENS = 15`, `EPOCHS = 2`, `LEARNING_RATE = 3e-5`,
   `BATCH_SIZE = 16`, `TRAINABLE_ENCODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's
   preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `normalize_answer`, `exact_match`, `f1`, `qa_metrics`,
     `null_baseline`, `lexical_overlap_baseline`, `RoBERTaQuestionAnsweringPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_csv`, `gold_texts` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `deepset/roberta-base-squad2` at the immutable revision, and `verify_snapshot` returning its dict (7 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`
     and no tokenizer warning (the snapshot has no `tokenizer.json`; `RobertaTokenizerFast` is built from
     `vocab.json`/`merges.txt`);
   - Section 4: `fetch_corpus` fetching the pinned 9,018,914-byte zip (SHA-256 `f4f3c232…`) from
     `adversarialqa.github.io` into `weights/adversarialqa/`, 10,000 + 1,000 raw questions read, and the seeded
     split into 1,000 / 200 / 400 records over 358 / 5 / 16 articles with `check_split_disjoint` reporting no shared
     passage and the three dataset digests `19021a6f…` / `c2ba4720…` / `3598b03c…`; `outputs/…_train.csv`
     written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the fit check dropping exactly `train-0362` (460 tokens) and nothing else; the ceilings
     (`MAX_QUESTION_CHARS` 500, `MAX_CONTEXT_CHARS` 3000, `MAX_QUESTION_TOKENS` 64, `MAX_CONTEXT_TOKENS` 384,
     `MAX_ANSWER_TOKENS` 64, `DEFAULT_MAX_ANSWER_TOKENS` 15) and `DECISION_RULE` surfaced; `validate_inputs` writing
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     `max_answer_tokens` ceiling probe); `pipe.answer` on the three authored questions with every sanity check
     `True` (the card-pass smoke returned `Gustave Eiffel`, `1887 to 1889` and the empty answer; a different span on
     another runtime is a finding to record, not a failure);
   - Section 6: the always-null baseline (0.25 / 0.25 — the sample has no unanswerable questions; one gold span, the initial `A`, normalises to the empty string under the SQuAD article-stripping rule), the
     lexical-overlap baseline (exact-match 0.0, F1 ≈ 8.6) and the frozen model's test score (exact-match ≈ 8.0,
     F1 ≈ 12.4, answered rate ≈ 43.5 % on CPU float32), with the cell's assertion that the frozen F1 beats the null
     baseline;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 14,177,282 trainable of 124,056,578 parameters,
     999 training questions, and a two-epoch history with validation F1 rising and the answered rate reaching
     100 % (≈ 14.1 → 31.6 → 30.9 in the recorded run; `best_epoch` 1);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test F1 exceeds the frozen one — on
     the sample ≈ 26.5 versus ≈ 12.4, exact-match ≈ 17.3 versus ≈ 8.0, answered rate 100 %);
   - Section 9: six unseen dev questions answered with `pipe.evaluate` returning `measured-small-sample`, the
     single-pair `evaluation_report` verdict `sample-sanity`, `outputs/…_answers.csv` written; `pipe.save_artifact`
     writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (34 tensors, about 56.7 MB) and
     `RoBERTaQuestionAnsweringPipeline.from_artifact` reloading it with 8/8 identical answers (the cell asserts it);
     `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity, licence and attribution, the
     snapshot block (`weight_format`, `weight_sha256`), the `corpus` block, the inference-contract items, the
     comparison, the artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `roberta_question_answering_colab.ipynb` (`E2E`) | `45eac74` / `c0686c47` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `roberta_question_answering_colab.ipynb` (`TASK-INFERENCE`, superseded) | `9473216` / `5a73f16cf5cd` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-roberta-question-answering` v1) | PASSED — 8/8 code cells, 233.8 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/roberta_question_answering_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/roberta_question_answering_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `45eac74` / `c0686c47` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 7 entries because the snapshot was pre-staged → `verify_snapshot` 7 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest check → 10,000 + 1,000 questions read, 1,000 / 200 / 400 cut by article over 358 / 5 / 16 articles with `check_split_disjoint` clean and digests `19021a6f…` / `c2ba4720…` / `3598b03c…` → four dataset refusals → fit check dropping `train-0362` only → input manifest + `max_answer_tokens` refusal probe → three authored answers with every sanity check `True` → null + lexical baselines → frozen evaluation → `adapt` → validation + test evaluation → six unseen questions → adapter export → reload parity) | 462.9 s | **PASSED** — 11/11 code cells; always-null 0.25 / 0.25 (one gold `A` normalises to empty), lexical overlap 0.0 / 8.58; frozen test exact-match 8.0 / F1 12.38, answered 43.5 % (17.7 s); `adapt` 14,177,282 of 124,056,578 params, 999 questions, 2 epochs, 410.2 s, validation F1 14.14 → 31.63 → 30.91 (`best_epoch` 1, train loss 3.223 → 2.986); **adapted test exact-match 17.25 / F1 26.49 (Δ +9.25 / +14.11), answered 100 %**; six unseen questions exact-match 16.67 / F1 29.76 `measured-small-sample`, single-pair report `sample-sanity`; adapter 56,713,104 B / 34 tensors, SHA-256 `754fd982…`; reload parity 8/8; six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `9473216` / `5a73f16cf5cd` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-roberta-question-answering` v1) | Default sample path of the inference-only notebook: one synthetic passage and three questions, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot`, `answer`, `sample-sanity` report | 233.8 s | **PASSED** — 8/8 code cells, 16 files, 498 MB staged; does not cover the `E2E` blob |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU — corpus
read from the cache, validation and article split, fit check, the inference contract, both baselines, two epochs of
encoder fine-tuning, held-out evaluation, unseen-question answering, adapter export and reload parity — which catches
defects but is **not** a supported runtime under REL1/REL10, and it ran with the snapshot and the AdversarialQA zip
pre-staged, so neither the 496 MB Hub fetch nor the project-site download has been exercised by this notebook end to
end; the earlier `TASK-INFERENCE` Kaggle run did exercise the Hub fetch and digest check of the same snapshot. The
repository stays at **Candidate** until a Colab or fresh-container run of the exact `E2E` release revision is recorded
above.
