# RoBERTa-SQuAD2 Question Answering Pipeline

DIMER-oriented inference and fine-tuning wrapper for **deepset/roberta-base-squad2** (RoBERTa-base fine-tuned by deepset on SQuAD 2.0, licence CC-BY-4.0), pinned to an immutable Hugging Face revision. The repository exposes extractive question answering — one question and one context in, one context span or the SQuAD 2.0 empty "no answer" out — with the upstream null-vs-span decision rule, a supply-chain check of the local weight snapshot, SQuAD-style `exact_match`/`f1` helpers, machine-readable provenance, and a bounded adaptation contract: a digest-pinned real gold-span corpus (AdversarialQA dRoBERTa), corpus exact-match and F1 with two non-neural baselines, supervised fine-tuning of the last encoder blocks and the span head with validation-F1 epoch selection, and a safetensors adapter that reloads onto the digest-verified base.

## Upstream alignment

- Model: `deepset/roberta-base-squad2`
- Revision: `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`
- Upstream weight license: CC-BY-4.0 — attribution: deepset (Branden Chan, Timo Möller, Malte Pietsch, Tanay Soni), https://huggingface.co/deepset/roberta-base-squad2, weights redistributed unmodified; see `docs/WEIGHTS.md`
- Upstream task: extractive question answering with unanswerable questions (`RobertaForQuestionAnswering`, fine-tuned on SQuAD 2.0 from `FacebookAI/roberta-base`)
- Repository adaptation: bounded supervised fine-tuning of the last *k* encoder blocks plus the span head (`adapt`, default 2 of 12 = 14,177,282 of 124,056,578 parameters) on caller-supplied or pinned AdversarialQA records; the embeddings and the vocabulary are never modified; the adapter carries only the trained tensors and is bound to the base `model.safetensors` SHA-256

## Quick start

```python
from roberta_question_answering_pipeline import RoBERTaQuestionAnsweringPipeline

pipe = RoBERTaQuestionAnsweringPipeline.from_pretrained()   # verifies weights/roberta-base-squad2 first
context = "The Eiffel Tower is named after the engineer Gustave Eiffel, whose company built it from 1887 to 1889."
result = pipe.answer("Who designed the Eiffel Tower?", context)
print(result["answer"], result["score"])          # 'Gustave Eiffel' 0.94  (CPU smoke on a longer context)
print(result["answerable"], result["no_answer_score"])
```

Adaptation on the pinned AdversarialQA sample (CPU, about seven minutes of training after the snapshot is staged):

```python
from roberta_question_answering_pipeline import (
    RoBERTaQuestionAnsweringPipeline, fetch_sample_dataset, check_split_disjoint, null_baseline, lexical_overlap_baseline,
)

splits = fetch_sample_dataset()            # one pinned 9.0 MB zip, digest-verified, cached under weights/adversarialqa/
check_split_disjoint(splits)               # 1,000 / 200 / 400 records, no passage shared between splits
pipe = RoBERTaQuestionAnsweringPipeline.from_pretrained()
fit = {name: pipe.check_fit(part)['fitting'] for name, part in splits.items()}   # drops records over the token ceilings (1 of 1,600)
print(lexical_overlap_baseline(fit['test'])['f1'], pipe.evaluate(fit['test'])['f1'])   # 8.58, 12.38 in the recorded run
pipe.adapt(fit['train'], fit['validation'])                                            # last 2 encoder blocks + span head, 2 epochs, best validation F1 kept
print(pipe.evaluate(fit['test'])['f1'])                                                # 26.49 in the recorded run
artifact = pipe.save_artifact('outputs/adapter')                                       # adapter.safetensors (56.7 MB) + manifest.json
again = RoBERTaQuestionAnsweringPipeline.from_artifact(artifact)                       # verifies base digest + artifact digest before applying
```

`answer(question, context, *, max_answer_tokens=15)` takes two non-empty strings of at most 500 / 3,000 characters (`MAX_QUESTION_CHARS`, `MAX_CONTEXT_CHARS`) that tokenise to at most 64 / 384 byte-level BPE tokens (`MAX_QUESTION_TOKENS`, `MAX_CONTEXT_TOKENS`; longer inputs are rejected, never truncated or windowed — the caller chunks long documents) and `max_answer_tokens` in 1..64 (`MAX_ANSWER_TOKENS`). The decision rule (`DECISION_RULE`) is the upstream `transformers` question-answering rule with `handle_impossible_answer=True`: softmax the start and end logits over the context tokens plus `<s>`, score the best span as `P_start(i)·P_end(j)` with `i ≤ j < i + max_answer_tokens`, score "no answer" as `P_start(<s>)·P_end(<s>)`, and return the empty string when the null score is larger. Every result carries `answer`, `score`, `start`/`end` (character offsets into `context`), `no_answer_score`, `best_span_score`, `answerable`, both token counts, `max_answer_tokens`, `decision_rule`, `device`, `source`, `model_id` and `model_revision`. `exact_match(prediction, gold_answers)` and `f1(prediction, gold_answers)` implement the SQuAD scoring normalisation (an empty gold list means unanswerable); `evaluation_report(result, gold_answers=None)` wraps them for one pair. `evaluate(records)` answers a validated `{id, question, context, answers}` dataset and reports corpus exact-match, F1 and the answered rate (`measured` / `measured-small-sample`); `check_fit(records)` drops — never truncates — the records the token ceilings would refuse; `adapt(train, val, *, epochs=2, lr=3e-5, batch_size=16, trainable_encoder_layers=2, seed=0)` fine-tunes the last encoder blocks and the span head and keeps the best-validation-F1 epoch; `save_artifact` / `from_artifact` export and reload the trained tensors as safetensors with a manifest bound to the base weight digest. Dataset helpers (`fetch_corpus`, `read_corpus`, `build_sample_dataset`, `validate_dataset`, `split_dataset`, `check_split_disjoint`, `load_byod_dataset`, `write_dataset_csv`) live in `samples.py`; baselines (`null_baseline`, `lexical_overlap_baseline`) and `qa_metrics` in `metrics.py`; records are 8–20,000 mappings with `answers` a list of `{text, answer_start}` found verbatim at its offset (empty for unanswerable), and every ceiling is a refusal, never a silent cut.

## Weights layout

```
weights/roberta-base-squad2/
  config.json  merges.txt  model.safetensors  special_tokens_map.json  tokenizer_config.json  vocab.json
  README.md  dimer-base-manifest.json  (no tokenizer.json at this revision)
```

`from_pretrained()` calls `stage_missing_files()` (fetches absent manifest entries at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (size + SHA-256 of every entry), and loads `RobertaForQuestionAnswering` + `AutoTokenizer` (a `RobertaTokenizerFast` built from `vocab.json`/`merges.txt`) with `local_files_only=True` and `trust_remote_code=False`. Without a manifest it raises unless `allow_download=True`. See `docs/WEIGHTS.md`.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake encoder, token counter and corpus fetcher plus temporary manifests, never the weights (36 tests plus 5 notebook-parity tests). `tests/test_model_backed.py` (2 tests: `evaluate` against gold spans, a one-epoch adaptation of the last encoder block with an artifact round trip) runs only when `weights/roberta-base-squad2/` is staged.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/tutorials/roberta_question_answering_colab.ipynb)

`tutorials/roberta_question_answering_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path fetches one pinned AdversarialQA zip (9.0 MB, CC BY-SA 3.0, refused on any digest mismatch) and cuts the dRoBERTa subset by article into 1,000 / 200 / 400 passage-disjoint records with four refusal probes, stages the git-ignored `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, drops the records the token ceilings would refuse with `check_fit` (1 of 1,600), exercises the inference contract with its input manifest and sanity checks on three authored questions, scores the always-null and lexical-overlap baselines and the frozen model on the test split (F1 0.0 / 8.58 / 12.38 in the recorded run, with the null answer returned for 56.5 % of the answerable questions), fine-tunes the last two encoder blocks and the span head for two epochs with validation-F1 epoch selection (416 s on CPU), re-scores the test split (F1 26.49, exact-match 8.0 → 17.25, answered rate 100 %), answers six unseen questions with a `measured-small-sample` verdict, exports a 56.7 MB safetensors adapter and reloads it with 8/8 identical answers. Every number is one seeded split with no dispersion estimate. BYOD (`{id, question, context, answers}` as a JSON array, SQuAD JSON, JSONL or CSV) is optional and gated off by default. See `docs/release-verification.md` for the release gate.

## Release status

**Candidate** — the notebook source passes all static checks and one local CPU pre-flight execution of the committed blob is recorded; a clean run in a supported hosted runtime is still required (see `STATUS.md` and `docs/release-verification.md`). The earlier inference-only notebook's Kaggle pass does not carry over to the `E2E` blob.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged upstream weights are CC-BY-4.0 and require attribution to deepset; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
