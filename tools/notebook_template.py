"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "roberta_question_answering_pipeline",
    "repo_name": "roberta-squad2-question-answering-pipeline",
    "stem": "roberta_question_answering",
    "notebook_name": "roberta_question_answering_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "RoBERTaQuestionAnsweringPipeline",
    "weights_key": "roberta-base-squad2",
    "runtime_imports": ["torch", "transformers"],
    "title": "RoBERTa-base SQuAD2 — DIMER extractive question answering tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/roberta-squad2-question-answering-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/tutorials/roberta_question_answering_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-deepset%2Froberta--base--squad2-ffcc4d?style=flat",
            "https://huggingface.co/deepset/roberta-base-squad2",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-deepset--ai%2Fhaystack-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/deepset-ai/haystack",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1907.11692-b31b1b.svg", "https://arxiv.org/abs/1907.11692"),
    ],
    "capability": "extractive question answering with the SQuAD 2.0 unanswerable case using the pinned `deepset/roberta-base-squad2` weights",
    "intro": (
        "`deepset/roberta-base-squad2` is the 125 M-parameter `roberta-base` encoder (Liu et al., 2019) with a span "
        "head, fine-tuned by deepset on SQuAD 2.0 — question-answer pairs **including unanswerable questions** — and "
        "published under **CC-BY-4.0** (attribution: deepset; the weights are redistributed unmodified). It is an "
        "**extractive reader**: it can only copy one contiguous span out of the passage you give it, and it can decide "
        "that the passage contains no answer. At inference the encoder reads `<s> question </s></s> context </s>` once "
        "and emits a start and an end logit per token; the carried module then applies the **upstream null-vs-span "
        "rule** (`DECISION_RULE`): softmax the start and end logits over the context tokens plus `<s>`, take the best "
        "span as the argmax of `P_start(i)·P_end(j)` with `i ≤ j < i + max_answer_tokens`, score \"no answer\" as "
        "`P_start(<s>)·P_end(<s>)`, and return the **empty answer** when the null score wins. **No adaptation occurs:** "
        "no training, fine-tuning, in-context conditioning, retrieval or preprocessing fitting — the pinned checkpoint "
        "is used as published. What the upstream checkpoint supplies is the model and the byte-level BPE tokenizer; "
        "what the carried pipeline module adds is manifest verification, input validation with named ceilings, the "
        "decision rule, a fixed output contract, the SQuAD-style `exact_match` and `f1` helpers, and the "
        "`validate_inputs` and `evaluation_report` stage helpers.\n\n"
        "**Snapshot note:** the pinned revision ships **no `tokenizer.json`** (a 7-file manifest), so `AutoTokenizer` "
        "builds `RobertaTokenizerFast` from `vocab.json` and `merges.txt`; the card-pass smoke printed no tokenizer "
        "warning. Section 3 stages and digest-verifies those seven files before the tokenizer or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author one passage and three "
        "questions (two answerable, one deliberately unanswerable) or upload your own, stage and digest-verify the "
        "immutable upstream snapshot, surface the pipeline's ceilings and the decision rule and validate the pairs into "
        "an input manifest before any model work, answer through the public API with an explicit `max_answer_tokens`, "
        "read `answer`, `score`, `no_answer_score`, `best_span_score` and `answerable` correctly (products of softmax "
        "masses, not calibrated probabilities), read from the machine-readable evaluation report what `exact_match`/`f1` "
        "on authored gold answers does and does not prove, and export every answer with its identifier plus provenance."
    ),
    "exclusions": (
        "generative or abstractive answering, yes/no or counting questions, multi-hop reasoning across sentences, "
        "retrieval over many documents (this reader takes one passage the caller supplies), passages over "
        "`MAX_CONTEXT_TOKENS` (refused, not windowed — chunk them yourself), languages other than English, batching, "
        "top-k alternative answers, a tuned no-answer threshold offset, or any accuracy claim beyond a single-pair "
        "sample-sanity check. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (float32 there too). CPU is adequate: the repository's model card records, for the Windows-venv smoke on an Intel Core Ultra 9 275HX, 4.41 s to load and digest-verify the 498 MB snapshot and 0.105 s / 0.032 s / 0.032 s for three `answer` calls on a 48-token passage. The pinned `torch==2.14.0` install and the 496 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-only Transformer is; what span extraction means; why a reader can return a fluent-looking wrong span, or an empty answer for a question the passage does answer.",
        "- **Data:** the default sample is **synthetic** — one two-sentence passage and three questions authored in code, with the author's own gold answers (the third question has no answer in the passage) — so nothing is downloaded and no private data is needed. Three pairs are a plumbing check, never an accuracy. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction; the expected file is a JSON list of `{\"question\", \"context\"}` objects, each optionally carrying `\"answers\"` (a list of acceptable strings; an empty list means unanswerable). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: one two-sentence passage about the Eiffel Tower and three questions "
                "authored in this cell, the same passage the repository's card-pass smoke used. Two questions are "
                "answered literally in the passage and carry the author's gold spans; the third (`What colour is the "
                "tower painted?`) is **deliberately unanswerable** and carries an empty gold list, the SQuAD 2.0 "
                "convention. The card's smoke observations — `Gustave Eiffel`, `1887 to 1889`, and the empty answer — "
                "are one run on one machine, not expected values this notebook asserts. The sample identity and a "
                "SHA-256 of its text are printed so an export can be tied to exactly these inputs.\n\n"
                "One Colab form parameter fixes the span-length cap for every call: `ANSWER_MAX_TOKENS` (default 15, "
                "the package's `DEFAULT_MAX_ANSWER_TOKENS` and the upstream pipeline's `max_answer_len` default). It "
                "is checked against the carried module's ceiling in the next section.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 JSON file holding a list of "
                "objects with `question` and `context` strings and, optionally, `answers` (a list of acceptable gold "
                "strings; `[]` for unanswerable). Each question must be at most `MAX_QUESTION_CHARS` characters and "
                "`MAX_QUESTION_TOKENS` BPE tokens, each context at most `MAX_CONTEXT_CHARS` characters and "
                "`MAX_CONTEXT_TOKENS` tokens, which the pipeline enforces by rejecting, not by truncating or windowing — "
                "chunk long documents yourself. The upload stays inside this runtime."
            ),
            "code": (
                "import hashlib\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "ANSWER_MAX_TOKENS = 15  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    records = json.loads(uploaded[sample_name].decode('utf-8'))\n"
                "    if not isinstance(records, list) or not records:\n"
                "        raise ValueError(f'{{sample_name}}: expected a non-empty JSON list of {{{{question, context[, answers]}}}} objects')\n"
                "    questions = [str(r['question']) for r in records]\n"
                "    contexts = [str(r['context']) for r in records]\n"
                "    golds = [list(r['answers']) if 'answers' in r else None for r in records]\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    passage = (\n"
                "        'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. '\n"
                "        'It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.'\n"
                "    )\n"
                "    questions = [\n"
                "        'Who designed the Eiffel Tower?',\n"
                "        'When was the tower built?',\n"
                "        'What colour is the tower painted?',\n"
                "    ]\n"
                "    contexts = [passage] * len(questions)\n"
                "    golds = [['Gustave Eiffel'], ['1887 to 1889', 'from 1887 to 1889'], []]\n"
                "    sample_name = 'synthetic_eiffel_tower_qa'\n"
                "    sample_kind = 'synthetic (authored in this cell)'\n"
                "item_ids = [f'pair{{index:02d}}' for index in range(len(questions))]\n"
                "sample_sha256 = hashlib.sha256('\\n'.join(q + '\\t' + c for q, c in zip(questions, contexts, strict=True)).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'pairs': len(questions), 'text_sha256': sample_sha256, 'max_answer_tokens': ANSWER_MAX_TOKENS, 'with_gold': sum(g is not None for g in golds)}})\n"
                "for item_id, question, gold in zip(item_ids, questions, golds, strict=True):\n"
                "    print(f'{{item_id}}: {{question}}  gold={{gold}}')"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `answer` "
                "applies — both route through the same private `_check_inputs` — so the text types, non-emptiness, the "
                "character ceilings `MAX_QUESTION_CHARS`/`MAX_CONTEXT_CHARS` and `max_answer_tokens` in "
                "1..`MAX_ANSWER_TOKENS` are enforced identically. `answer` takes one pair per call, so the helper "
                "validates the whole parallel batch the notebook will loop over and returns one **input manifest** "
                "naming the schema and ceilings, each pair's identifier and character counts, the setting in force, and "
                "the verdict; it is written to `outputs/{stem}_input_manifest.json`. The token ceilings "
                "`MAX_QUESTION_TOKENS` (64) and `MAX_CONTEXT_TOKENS` (384, the upstream fine-tuning window) need the real "
                "tokenizer and are therefore enforced inside `answer`, which **rejects with a `ValueError` naming the "
                "count, never silently truncates or windows**; every result reports both counts. `DECISION_RULE` states "
                "the null-vs-span rule in force. To show what rejection looks like, the cell also validates an "
                "out-of-range `max_answer_tokens` and records the pipeline's own error message as a finding. Nothing "
                "here trims or alters the texts."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MAX_CONTEXT_CHARS': MAX_CONTEXT_CHARS, 'MAX_QUESTION_TOKENS': MAX_QUESTION_TOKENS, 'MAX_CONTEXT_TOKENS': MAX_CONTEXT_TOKENS, 'MAX_ANSWER_TOKENS': MAX_ANSWER_TOKENS, 'DEFAULT_MAX_ANSWER_TOKENS': DEFAULT_MAX_ANSWER_TOKENS}}\n"
                "print(ceilings)\n"
                "print({{'decision_rule': DECISION_RULE}})\n"
                "input_manifest = validate_inputs(questions, contexts, max_answer_tokens=ANSWER_MAX_TOKENS, names=item_ids)\n"
                "# Demonstrate rejection on a setting that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(questions, contexts, max_answer_tokens=MAX_ANSWER_TOKENS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'max-answer-tokens-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'token_ceilings': 'enforced by answer() with the real tokenizer; reported as question_tokens / context_tokens'}})"
            ),
        },
        {
            "md": (
                "## 6. Answer and read the outputs correctly\n\n"
                "`answer(question, context, max_answer_tokens=...)` runs one pair through the encoder and returns a "
                "dict: `answer` (the extracted span, or `''` for the SQuAD 2.0 no-answer), `start`/`end` (character "
                "offsets into `context`, both `0` for no-answer), `score` (the score of what was returned), "
                "`no_answer_score` (`P_start(<s>)·P_end(<s>)`), `best_span_score` (the best span's score even when the "
                "null won), `answerable` (whether a span was returned), `question_tokens`/`context_tokens` (the counts "
                "checked against the ceilings), `max_answer_tokens`, `decision_rule`, `device`, `source` and the model "
                "identity. **Score semantics:** every score is a **product of two softmax masses** normalised over the "
                "passage — a ranking signal that shrinks as the passage grows, **not a calibrated probability** — and "
                "the only decision the pipeline ships is the null comparison; there is no shipped no-answer threshold "
                "offset, so a caller who needs to trade missed facts against fabricated ones owns that margin, judged on "
                "their own labelled pairs. The run is deterministic for a given pair, setting, weights, device and "
                "library versions (no sampling, `model.eval()`, no seed needed); float32 kernel differences between CPU "
                "and CUDA can move a score in the last digits and flip a near-tied span or null decision. The checks "
                "below are falsifiable plumbing checks — one result per pair, every span a literal substring of its "
                "passage at the reported offsets, every count within its ceiling — plus a per-call wall time measured on "
                "the runtime identified in Section 1 (the first call includes warm-up). Look for a name for `pair00`, a "
                "date range for `pair01`, and an empty answer for `pair02`; whether they are *right* is what Section 7 "
                "checks against the authored gold, for three pairs only."
            ),
            "code": (
                "import time\n\n"
                "results = []\n"
                "for item_id, question, context in zip(item_ids, questions, contexts, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.answer(question, context, max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    results.append({{'id': item_id, 'question': question, 'context': context, 'seconds': round(elapsed, 3), **result}})\n"
                "    print(f\"{{item_id}} [{{'answerable' if result['answerable'] else 'no answer'}}] score={{result['score']:.4f}} null={{result['no_answer_score']:.4f}} best_span={{result['best_span_score']:.4f}} {{result['question_tokens']}}+{{result['context_tokens']}} tokens, {{elapsed:.2f}} s\")\n"
                "    print(f\"    {{result['answer']!r}} [{{result['start']}}:{{result['end']}}]\")\n"
                "checks = {{\n"
                "    'one_result_per_pair': len(results) == len(questions),\n"
                "    'span_is_substring_at_offsets': all(r['context'][r['start']:r['end']] == r['answer'] for r in results),\n"
                "    'question_within_ceiling': all(r['question_tokens'] <= MAX_QUESTION_TOKENS for r in results),\n"
                "    'context_within_ceiling': all(r['context_tokens'] <= MAX_CONTEXT_TOKENS for r in results),\n"
                "    'scores_in_unit_interval': all(0.0 <= r[k] <= 1.0 for r in results for k in ('score', 'no_answer_score', 'best_span_score')),\n"
                "    'null_rule_consistent': all(r['answerable'] == (r['no_answer_score'] <= r['best_span_score']) for r in results),\n"
                "    'setting_echoed': all(r['max_answer_tokens'] == ANSWER_MAX_TOKENS for r in results),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'answer output failed a sanity check: {{checks}}')\n"
                "print({{'checks': checks, 'decision_rule': results[0]['decision_rule'], 'unanswered': [r['id'] for r in results if not r['answerable']]}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. The "
                "repository ships the two official SQuAD 2.0 measures as helpers — `exact_match` (the normalised "
                "prediction equals a normalised gold answer; lower-cased, punctuation and the articles a/an/the removed) "
                "and `f1` (best token-overlap F1 against any gold answer; an empty gold list means unanswerable and "
                "scores 1.0 only for an empty prediction) — and the report carries both **for one pair** with the "
                "verdict `sample-sanity`: three authored pairs are a plumbing check of the decision rule, **not an "
                "accuracy**, and the helper says so in `reason`. Without gold the verdict is `not-measurable` and "
                "`needs` names the held-out gold spans a real evaluation requires. The cell writes the report for the "
                "first pair to `outputs/{stem}_evaluation_report.json` and prints the per-pair scores for all of them. "
                "The upstream dev-set figures in the model card (exact 79.87 / F1 82.91 on 11,873 SQuAD 2.0 questions) "
                "are upstream claims, not measured here."
            ),
            "code": (
                "reports = [evaluation_report(r, gold, sample_kind=sample_kind) for r, gold in zip(results, golds, strict=True)]\n"
                "report = reports[0]\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "per_pair = [{{'id': r['id'], 'verdict': rep['verdict'], **{{m['id']: m['value'] for m in rep['metrics']}}}} for r, rep in zip(results, reports, strict=True)]\n"
                "print({{'per_pair': per_pair}})\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: no gold answers were supplied; score your own held-out pairs with exact_match and f1.')\n"
                "else:\n"
                "    print('sample-sanity only: exact_match/f1 on a handful of authored pairs is a plumbing check, not an accuracy.')"
            ),
        },
        {
            "md": (
                "## 8. Export the answers and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report: "
                "`outputs/{stem}_answers.csv` — one row per pair with its identifier, the question, the extracted answer, "
                "its offsets, the three scores, `answerable`, both token counts and the wall time, so every answer maps "
                "back to its pair — and `outputs/{stem}_result.json`, which carries the same items plus each pair's "
                "context and gold, the per-pair evaluation reports, the setting in force, the ceilings, the sanity "
                "checks, the input manifest, the sample identity and digest, the notebook's source (repository, "
                "revision, embedded module digest, generator), the model identifier, the immutable model revision, the "
                "model licence (CC-BY-4.0, attribution deepset), the verified snapshot summary, and the runtime identity "
                "(Python, `torch`, `transformers`, device, dtype). No credentials are involved in any step, so none can "
                "reach the export."
            ),
            "code": (
                "import csv\n\n"
                "items = [\n"
                "    {{'id': r['id'], 'question': r['question'], 'answer': r['answer'], 'start': r['start'], 'end': r['end'], 'score': r['score'], 'no_answer_score': r['no_answer_score'], 'best_span_score': r['best_span_score'], 'answerable': r['answerable'], 'question_tokens': r['question_tokens'], 'context_tokens': r['context_tokens'], 'seconds': r['seconds']}}\n"
                "    for r in results\n"
                "]\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(items[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(items)\n"
                "payload = {{\n"
                "    'items': [{{**item, 'context': r['context'], 'gold': gold}} for item, r, gold in zip(items, results, golds, strict=True)],\n"
                "    'evaluation_reports': reports,\n"
                "    'max_answer_tokens': ANSWER_MAX_TOKENS,\n"
                "    'decision_rule': DECISION_RULE,\n"
                "    'ceilings': ceilings,\n"
                "    'sanity_checks': checks,\n"
                "    'answers_file': 'outputs/{stem}_answers.csv',\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'pairs': len(questions), 'text_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'model_attribution': 'deepset (https://huggingface.co/deepset/roberta-base-squad2), weights redistributed unmodified under CC-BY-4.0',\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "        'source': pipe.source,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The returned spans are the model's highest-scoring contiguous substrings of *your* passage under the "
        "null-vs-span rule: a span can be fluent, plausible and wrong, and an empty answer can suppress a fact the "
        "passage states in other words. Every score is a product of softmax masses, not a calibrated probability, and "
        "no no-answer margin is shipped — `no_answer_score` and `best_span_score` are reported so you can set one on "
        "your own labelled pairs. On the synthetic sample the checks prove only that the input contract, the verified "
        "snapshot load, the decision rule and the offset bookkeeping work end to end; the `sample-sanity` "
        "`exact_match`/`f1` on three authored pairs is a plumbing check, and a real evaluation needs held-out gold "
        "spans from your own domain over enough pairs to state a dispersion, read separately for answerable and "
        "unanswerable questions. Passages above `MAX_CONTEXT_TOKENS` are refused rather than windowed, so a long "
        "document must be chunked by the caller and an answer that straddles a chunk boundary is lost. The pipeline "
        "exposes no retrieval, batching, top-k alternatives or generative answering. Inference is deterministic on a "
        "fixed device and dtype, but CPU and CUDA float32 kernels can diverge on a near-tied span.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated inputs against the enforced "
        "ceilings, execute the public pipeline path, and emit the shown machine-readable outputs in the tested runtime — "
        "without the repository being reachable. It does **not** establish benchmark superiority, reading-comprehension "
        "accuracy on any domain, a usable no-answer threshold, safety for high-consequence decisions, or production "
        "fitness on an unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: "
        "the pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the "
        "top. `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/{MODEL_KEY}/` and rerun Section 3. `ValueError: context is N "
        "tokens; ceiling is MAX_CONTEXT_TOKENS=384` in Section 6: split that BYOD passage into shorter chunks and rerun "
        "from Section 4. An empty answer for a question you believe the passage answers: the null score won — compare "
        "`no_answer_score` with `best_span_score` in the export, and try rephrasing the question closer to the passage's "
        "wording. A span cut short: raise `ANSWER_MAX_TOKENS` (ceiling `MAX_ANSWER_TOKENS`) and rerun Section 6.\n\n"
        "**Next experiments.** Rephrase the unanswerable question so it *is* answerable from the passage and watch the "
        "null score fall; append an unrelated paragraph to the passage and watch every score shrink as the softmax mass "
        "spreads; upload a small BYOD file with gold answers from your own documents and read the per-pair "
        "`exact_match`/`f1` — the first step towards the real evaluation the report asks for; run the same pairs on a "
        "CUDA runtime and diff the scores against the CPU run. None of these turns the sample result into evidence of "
        "production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance and CC-BY-4.0 attribution: https://github.com/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (deepset, CC-BY-4.0): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/deepset-ai/haystack\n"
        "- RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019): https://arxiv.org/abs/1907.11692\n"
        "- Know What You Don't Know: Unanswerable Questions for SQuAD (Rajpurkar et al., ACL 2018): https://arxiv.org/abs/1806.03822"
    ),
}
