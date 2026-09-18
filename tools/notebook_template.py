"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (metrics.py, pipeline.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E extractive-QA workflow: the pinned roberta-base-squad2 snapshot is
digest-verified and loaded, a digest-pinned real adversarial QA corpus (AdversarialQA dRoBERTa) is
fetched, validated and split by article, three authored questions are answered through the inference
contract, the frozen model is scored against gold spans beside two non-neural baselines, a bounded
fine-tuning of the last encoder blocks and the span head runs in the kernel, the held-out split is
scored again, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "roberta_question_answering_pipeline",
    "repo_name": "roberta-squad2-question-answering-pipeline",
    "stem": "roberta_question_answering",
    "notebook_name": "roberta_question_answering_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `deepset/roberta-base-squad2` snapshot (safetensors, 496 MB), fetches the digest-pinned AdversarialQA "
        "corpus from the project site (9.0 MB, no credential), filters the dRoBERTa subset and cuts it by article into "
        "1,000 / 200 / 400 training, validation and test questions, drops any record the pipeline's token ceilings would "
        "refuse, answers three authored questions through the inference contract with an input manifest and a rejection "
        "probe, scores the frozen model on the test split with exact-match and F1 beside the always-null and "
        "lexical-overlap baselines, runs a bounded fine-tuning of the last two encoder blocks and the span head on the "
        "training questions with validation-F1 epoch selection, scores the held-out split again, answers new questions "
        "with the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a "
        "fresh pipeline to verify answer parity. The default path needs no repository clone, no DIMER worker or service, "
        "no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes "
        "about six minutes of model time after the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "question–answer pairs as a JSON array of `{{id, question, context, answers}}` records, a SQuAD-format JSON file, a "
        "JSONL file, or a CSV with columns `id, question, context, answer_text, answer_start`. They pass through the same "
        "validation, seeded passage-disjoint split, fit check, baselines, fine-tuning, held-out evaluation, inference, "
        "artifact export and reload-parity cells as the AdversarialQA sample. The expected schema and the ceilings are stated "
        "in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part "
        "of the default path."
    ),
    "pipeline_class": "RoBERTaQuestionAnsweringPipeline",
    "weights_key": "roberta-base-squad2",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "RoBERTa-base SQuAD2 — DIMER E2E extractive question-answering fine-tuning tutorial (standalone)",
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
    "capability": "extractive question answering with the SQuAD 2.0 unanswerable case and bounded supervised fine-tuning of the last encoder blocks and the span head on a gold-span dataset, using the pinned `deepset/roberta-base-squad2` weights",
    "intro": (
        "`deepset/roberta-base-squad2` is the 124 M-parameter `roberta-base` encoder (Liu et al., 2019) with a span "
        "head, fine-tuned by deepset on SQuAD 2.0 — question-answer pairs **including unanswerable questions** — and "
        "published under **CC-BY-4.0** (attribution: deepset; the weights are redistributed unmodified). It is an "
        "**extractive reader**: it can only copy one contiguous span out of the passage you give it, and it can decide "
        "that the passage contains no answer. At inference the encoder reads `<s> question </s></s> context </s>` once "
        "and emits a start and an end logit per token; the carried module then applies the **upstream null-vs-span "
        "rule** (`DECISION_RULE`): softmax the start and end logits over the context tokens plus `<s>`, take the best "
        "span as the argmax of `P_start(i)·P_end(j)` with `i ≤ j < i + max_answer_tokens`, score \"no answer\" as "
        "`P_start(<s>)·P_end(<s>)`, and return the **empty answer** when the null score wins.\n\n"
        "What this notebook adds to inference is **adaptation with gold spans**. The dataset is real and deliberately "
        "hard: AdversarialQA v1.0 (Bartolo et al., TACL 2020, CC BY-SA 3.0) — SQuAD-style questions over Wikipedia "
        "passages written by annotators who could see a reader's predictions and kept only the questions it got wrong. "
        "The `dRoBERTa` subset was collected against a RoBERTa reader, so the frozen `roberta-base-squad2` scores far "
        "below its SQuAD 2.0 dev figures here (the build record measured F1 12.4 on the test split, with the null "
        "answer returned for more than half of the answerable questions), and the fine-tuning question is whether a "
        "small in-distribution adaptation of the last two encoder blocks and the span head recovers ground on held-out "
        "articles. Two metrics are implemented in the carried `metrics.py` (corpus **exact-match** and **F1** with the "
        "official SQuAD 2.0 normalisation), and two **non-neural baselines** — always-null and lexical overlap — show "
        "where a reader that does nothing, or only counts shared words, sits. Nothing here is a quality claim about your "
        "domain: it is one seeded split of one corpus.\n\n"
        "**Snapshot note:** the pinned revision ships **no `tokenizer.json`** (a 7-file manifest), so `AutoTokenizer` "
        "builds `RobertaTokenizerFast` from `vocab.json` and `merges.txt`; the weights are `model.safetensors` — no pickle "
        "is opened anywhere in this notebook. Section 3 stages and digest-verifies those seven files before the tokenizer "
        "or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned adversarial QA corpus and validate and split "
        "it by article without leakage; drop the records the token ceilings would refuse rather than truncating them; "
        "answer through the public API with an explicit `max_answer_tokens` and read `answer`, `score`, `no_answer_score`, "
        "`best_span_score` and `answerable` correctly (products of softmax masses, not calibrated probabilities); score the "
        "frozen model against gold spans beside two non-neural baselines and read what the null-answer rate says; run a "
        "bounded fine-tuning with explicit hyperparameters and validation-based epoch selection; evaluate on an "
        "article-disjoint test split; answer new questions; and export a safetensors adapter that reloads against the "
        "pinned base with verified parity."
    ),
    "exclusions": (
        "generative or abstractive answering, yes/no or counting questions, multi-hop reasoning across sentences, retrieval "
        "over many documents (this reader takes one passage the caller supplies), passages over `MAX_CONTEXT_TOKENS` "
        "(refused, not windowed — chunk them yourself), languages other than English, batching, top-k alternative answers, "
        "a tuned no-answer threshold offset, full-model or embedding fine-tuning, training on unanswerable questions "
        "beyond the SQuAD 2.0 `<s>` convention, and any claim that an AdversarialQA split stands in for your domain. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate but not fast: the build record measured 6 s to load and digest-verify the 498 MB snapshot, 20 s to answer the 400-question test split, and about 3.5 minutes per epoch of fine-tuning the last two encoder blocks and the span head on 1,000 questions (validation scoring included). The pinned `torch==2.14.0` install and the 496 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-only Transformer is; what span extraction means; why a reader can return a fluent-looking wrong span, or an empty answer for a question the passage does answer; what exact-match and F1 measure and why neither is a human judgement.",
        "- **Data contract:** records are `{{id, question, context, answers}}` — a question, its passage and a list of `{{text, answer_start}}` gold spans (empty for unanswerable), each `text` found verbatim at its offset; questions at most `MAX_QUESTION_CHARS` (500) characters and `MAX_QUESTION_TOKENS` (64) BPE tokens, passages at most `MAX_CONTEXT_CHARS` (3,000) characters and `MAX_CONTEXT_TOKENS` (384) tokens, enforced by rejecting, never by truncating or windowing; ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records; every question on the same passage lands in the same split so a test passage is never trained on. BYOD accepts a JSON array, SQuAD-format JSON, JSONL or CSV in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a question is answerable from its passage beyond the gold text being present at its offset, or that a gold span is the *best* answer — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal document set with its question log is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches one pinned object (`aqa_v1.0.zip`, 9,018,914 bytes, SHA-256 `f4f3c232…`) from `adversarialqa.github.io` over HTTPS, refused on any mismatch before it is read; only the `3_droberta/train.json` and `3_droberta/dev.json` members are read, and the corpus is CC BY-SA 3.0 (Bartolo et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Adversarial QA corpus, validation and split\n\n"
                "`fetch_corpus` downloads the pinned AdversarialQA zip (or reads it from the cache), refuses a byte-size or "
                "SHA-256 mismatch before the archive is opened, and `read_corpus` reads the two dRoBERTa members without "
                "extracting to disk, flattening the SQuAD structure into flat records that keep their article title. "
                "`build_sample_dataset` keeps answerable records whose passage is at most 1,300 characters and whose every "
                "gold span is found at its offset, drops exact duplicate pairs, draws 1,000 training questions from the "
                "10,000-question training pool by a seeded shuffle, and cuts the 1,000-question dev member **by article** "
                "into 200 validation and 400 test questions, so no passage and no article is shared between validation and "
                "test (the training pool's articles are disjoint from dev by AdversarialQA's construction). "
                "`validate_dataset` then checks every record against the contract, `check_split_disjoint` asserts no "
                "passage appears in two splits, and the training split is written to `outputs/{stem}_train.csv` in the "
                "shape BYOD expects.\n\n"
                "Look for: 10,000 + 1,000 raw questions, three digests, splits 1,000 / 200 / 400 over 358 / 5 / 16 "
                "articles, zero unanswerable records, and four refusal probes — a duplicate id, a gold span at the wrong "
                "offset, a missing field and a dataset too small to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_questions = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/adversarialqa'))\n"
                "    raw_questions = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} {{CORPUS_RELEASE}} dRoBERTa subset ({{CORPUS_LICENSE}})'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "articles = {{name: len({{r.get('title', '') for r in part}}) for name, part in splits.items()}}\n"
                "write_dataset_csv(splits['train'], 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_questions': raw_questions, 'splits': disjoint, 'articles': articles, 'corpus_sha256': CORPUS_SHA256[:16] + '...'}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_contexts': manifest['unique_contexts'], 'unanswerable': manifest['unanswerable'], 'context_chars': manifest['context_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'question': example['question'], 'answers': example['answers'], 'context': example['context'][:160] + '...'}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'gold span at the wrong offset': [{{**splits['train'][0], 'answers': [{{'text': splits['train'][0]['answers'][0]['text'], 'answer_start': 0}}]}}, *splits['train'][1:8]],\n"
                "    'missing field': [{{'id': r['id'], 'question': r['question'], 'context': r['context']}} for r in splits['train'][:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Fit check, then answer through the inference contract\n\n"
                "The token ceilings `MAX_QUESTION_TOKENS` (64) and `MAX_CONTEXT_TOKENS` (384, the upstream fine-tuning "
                "window) need the real tokenizer, so they are applied now that the model is loaded: `pipe.check_fit` "
                "partitions each split into the records `answer` accepts and the ones it would **refuse with a "
                "`ValueError` naming the count, never silently truncate or window**. The dropped ids are printed and the "
                "fitting records are what every later cell uses (the build record dropped 1 of 1,600 sample records — a "
                "1,295-character passage that tokenises to 460 pieces).\n\n"
                "Then the inference contract is exercised as it always was on three authored questions over one "
                "two-sentence passage — two answerable, one deliberately unanswerable. `validate_inputs` applies exactly "
                "the checks `answer` applies (text types, non-emptiness, the character ceilings, `max_answer_tokens` in "
                "1..`MAX_ANSWER_TOKENS`) and returns an input manifest; an out-of-range setting is validated too and its "
                "rejection recorded as a finding. `answer` returns the extracted span (or `''` for no-answer), character "
                "offsets, `score`, `no_answer_score`, `best_span_score`, `answerable`, both token counts and the model "
                "identity. **Score semantics:** every score is a **product of two softmax masses** normalised over the "
                "passage — a ranking signal that shrinks as the passage grows, **not a calibrated probability** — and no "
                "no-answer threshold offset ships. Whether the answers are *right* is what Section 6 measures on 400 gold "
                "spans, not what three authored pairs can tell you."
            ),
            "code": (
                "import time\n\n"
                "ANSWER_MAX_TOKENS = 15  # @param {{type:\"integer\"}}\n\n"
                "fit = {{name: pipe.check_fit(part) for name, part in splits.items()}}\n"
                "train_records, val_records, test_records = fit['train']['fitting'], fit['validation']['fitting'], fit['test']['fitting']\n"
                "print({{'fit_check': {{name: {{'fitting': f['n_fitting'], 'dropped': f['dropped']}} for name, f in fit.items()}}}})\n"
                "ceilings = {{'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MAX_CONTEXT_CHARS': MAX_CONTEXT_CHARS, 'MAX_QUESTION_TOKENS': MAX_QUESTION_TOKENS, 'MAX_CONTEXT_TOKENS': MAX_CONTEXT_TOKENS, 'MAX_ANSWER_TOKENS': MAX_ANSWER_TOKENS, 'DEFAULT_MAX_ANSWER_TOKENS': DEFAULT_MAX_ANSWER_TOKENS}}\n"
                "print(ceilings)\n"
                "print({{'decision_rule': DECISION_RULE}})\n\n"
                "passage = (\n"
                "    'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. '\n"
                "    'It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.'\n"
                ")\n"
                "questions = ['Who designed the Eiffel Tower?', 'When was the tower built?', 'What colour is the tower painted?']\n"
                "contexts = [passage] * len(questions)\n"
                "golds = [['Gustave Eiffel'], ['1887 to 1889', 'from 1887 to 1889'], []]\n"
                "item_ids = [f'pair{{index:02d}}' for index in range(len(questions))]\n"
                "input_manifest = validate_inputs(questions, contexts, max_answer_tokens=ANSWER_MAX_TOKENS, names=item_ids)\n"
                "try:\n"
                "    validate_inputs(questions, contexts, max_answer_tokens=MAX_ANSWER_TOKENS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'max-answer-tokens-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "results = []\n"
                "for item_id, question, context in zip(item_ids, questions, contexts, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.answer(question, context, max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "    results.append({{'id': item_id, 'question': question, 'context': context, 'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "    print(f\"{{item_id}} [{{'answerable' if result['answerable'] else 'no answer'}}] score={{result['score']:.4f}} null={{result['no_answer_score']:.4f}} best_span={{result['best_span_score']:.4f}} {{result['question_tokens']}}+{{result['context_tokens']}} tokens: {{result['answer']!r}} [{{result['start']}}:{{result['end']}}]\")\n"
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
                "print({{'checks': checks, 'unanswered': [r['id'] for r in results if not r['answerable']], 'findings': len(input_manifest['findings'])}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test split\n\n"
                "Three numbers frame the adaptation. The **always-null baseline** returns the empty answer for every "
                "question and scores exactly the unanswerable fraction of the set — zero here, which is the point: on "
                "AdversarialQA every null answer the frozen model returns is a miss. The **lexical-overlap baseline** "
                "returns the passage sentence sharing the most normalised tokens with the question — a bag-of-words reader "
                "with no model, whose F1 comes from partial overlap with long spans. The **frozen model** answers the 400 "
                "test questions with the `max_answer_tokens` from Section 5 and is scored with the same two metrics: corpus "
                "**exact-match** and **F1** with the official SQuAD 2.0 normalisation (lower-case, punctuation and articles "
                "stripped; an empty gold matches only an empty prediction). Read `answered_rate` beside them: the fraction of "
                "questions for which a span was returned at all. Expect the frozen F1 to be low — these questions were "
                "selected because a RoBERTa reader failed them — and expect the null answer for roughly half of the test "
                "questions, a systematic under-answering that the adaptation in Section 7 is meant to correct."
            ),
            "code": (
                "baseline_null = null_baseline(test_records)\n"
                "baseline_lexical = lexical_overlap_baseline(test_records)\n"
                "print({{'always_null_baseline': {{'exact_match': round(baseline_null['exact_match'], 2), 'f1': round(baseline_null['f1'], 2), 'n': baseline_null['n']}}}})\n"
                "print({{'lexical_overlap_baseline': {{'exact_match': round(baseline_lexical['exact_match'], 2), 'f1': round(baseline_lexical['f1'], 2), 'n': baseline_lexical['n']}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "print({{'frozen_model_test': {{'exact_match': round(frozen_test['exact_match'], 2), 'f1': round(frozen_test['f1'], 2), 'answered_rate': round(frozen_test['answered_rate'], 1), 'n': frozen_test['n'], 'verdict': frozen_test['verdict']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record in test_records[:3]:\n"
                "    item = pipe.answer(record['question'], record['context'], max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "    print({{'question': record['question'], 'frozen': item['answer'], 'gold': gold_texts(record)}})\n"
                "assert frozen_test['f1'] > baseline_null['f1']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last encoder blocks and the span head\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_ENCODER_LAYERS` encoder blocks plus the span head "
                "`qa_outputs` — two blocks by default, 14,177,282 of 124,056,578 parameters; the embeddings and the earlier "
                "blocks stay frozen — with start/end cross-entropy on the first gold span (position 0, `<s>`, for an "
                "unanswerable record, the SQuAD 2.0 convention), AdamW at a fixed learning rate, gradient clipping at 1.0, "
                "seeded shuffling and no scheduler. Nothing is truncated: every training record passed the fit check. "
                "Epoch 0 records the frozen model's validation exact-match and F1; every epoch is scored on the "
                "validation split, and the epoch with the highest validation F1 is kept.\n\n"
                "Watch validation F1 roughly double in the first epoch and the answered rate jump to 100 % (about 3.5 "
                "minutes per epoch on CPU, validation scoring included). The build record's counter-examples are in the "
                "model card; the default is the smallest configuration that captured most of the gain."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_ENCODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_exact_match'] = round(entry['val']['exact_match'], 2)\n"
                "        row['val_f1'] = round(entry['val']['f1'], 2)\n"
                "        row['val_answered_rate'] = round(entry['val']['answered_rate'], 1)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_encoder_layers=TRAINABLE_ENCODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and none of its passages or articles "
                "appears in the training or validation splits. The adapted model is scored exactly as the frozen model was "
                "in Section 6, and the four numbers are put side by side. Look for an F1 gain of ten points or more and an "
                "answered rate at or near 100 % — the cell asserts the adapted F1 is above the frozen F1 — and for the same "
                "three questions answered by the adapted model. Four hundred questions from one seeded split of one corpus "
                "give no dispersion estimate; the deltas are sample-sanity evidence that the adaptation contract works, not "
                "a benchmark, and a gain on adversarial Wikipedia questions says nothing about your documents until you "
                "measure it there. Note what the adaptation also removes: the model now answers every question, so its "
                "ability to say *no answer* on SQuAD 2.0-style unanswerable questions is traded away on this corpus, which "
                "contains none."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "adapted_val = pipe.evaluate(val_records, max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "comparison = {{\n"
                "    'exact_match': {{'always_null': round(baseline_null['exact_match'], 2), 'lexical_overlap': round(baseline_lexical['exact_match'], 2), 'frozen': round(frozen_test['exact_match'], 2), 'adapted': round(adapted_test['exact_match'], 2)}},\n"
                "    'f1': {{'always_null': round(baseline_null['f1'], 2), 'lexical_overlap': round(baseline_lexical['f1'], 2), 'frozen': round(frozen_test['f1'], 2), 'adapted': round(adapted_test['f1'], 2)}},\n"
                "    'answered_rate': {{'frozen': round(frozen_test['answered_rate'], 1), 'adapted': round(adapted_test['answered_rate'], 1)}},\n"
                "    'delta_vs_frozen': {{'exact_match': round(adapted_test['exact_match'] - frozen_test['exact_match'], 2), 'f1': round(adapted_test['f1'] - frozen_test['f1'], 2)}},\n"
                "}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "for record in test_records[:3]:\n"
                "    item = pipe.answer(record['question'], record['context'], max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "    print({{'question': record['question'], 'adapted': item['answer'], 'gold': gold_texts(record)}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'fit_check': {{name: {{'fitting': f['n_fitting'], 'dropped': f['dropped']}} for name, f in fit.items()}},\n"
                "    'max_answer_tokens': ANSWER_MAX_TOKENS,\n"
                "    'baselines': {{'always_null': baseline_null, 'lexical_overlap': baseline_lexical}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['f1'] > frozen_test['f1']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Answer new questions, export the adapter and reload it\n\n"
                "Six questions from dev articles that were in none of the splits (they were filtered out of the sample by "
                "the passage-length filter, so they are also a small look at longer passages) are answered by the adapted "
                "model through the same `answer` contract as Section 5 and scored with `pipe.evaluate`, which returns a "
                "`measured-small-sample` verdict because six questions carry no dispersion estimate; the single-pair "
                "`evaluation_report` helper is written for the first of them, as the inference-only tutorial did.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two encoder blocks and the span head, about "
                "57 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `RoBERTaQuestionAnsweringPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest and digest **before** deserialising, refuses "
                "any tensor that is not an adaptable encoder-block or span-head tensor, and overlays the tensors onto a "
                "freshly loaded base — a new object from files, not the in-memory model (VER2). The cell asserts identical "
                "answers (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "if USE_BYOD:\n"
                "    new_records = [{{**r, 'id': f'new-{{i:02d}}'}} for i, r in enumerate(test_records[:6])]\n"
                "else:\n"
                "    used = {{r['context'].lower() for part in splits.values() for r in part}}\n"
                "    candidates = [r for r in corpus['dev'] if r['context'].lower() not in used and r['answers'] and len(r['context']) <= MAX_CONTEXT_CHARS]\n"
                "    new_records = pipe.check_fit([{{**r, 'id': f'new-{{i:02d}}'}} for i, r in enumerate(candidates[:12])])['fitting'][:6]\n"
                "new_metrics = pipe.evaluate(new_records, max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "new_results = []\n"
                "for record in new_records:\n"
                "    item = pipe.answer(record['question'], record['context'], max_answer_tokens=ANSWER_MAX_TOKENS)\n"
                "    new_results.append({{'id': record['id'], 'question': record['question'], 'answer': item['answer'], 'gold': gold_texts(record), 'score': item['score'], 'no_answer_score': item['no_answer_score'], 'answerable': item['answerable'], 'question_tokens': item['question_tokens'], 'context_tokens': item['context_tokens']}})\n"
                "    print({{k: new_results[-1][k] for k in ('id', 'question', 'answer', 'gold')}})\n"
                "single_report = evaluation_report({{'answer': new_results[0]['answer']}}, new_results[0]['gold'], sample_kind='one unseen AdversarialQA pair' if not USE_BYOD else 'one BYOD test record')\n"
                "print({{'new_questions': {{'n': new_metrics['n'], 'exact_match': round(new_metrics['exact_match'], 2), 'f1': round(new_metrics['f1'], 2), 'verdict': new_metrics['verdict']}}, 'single_pair_report_verdict': single_report['verdict']}})\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=['id', 'question', 'answer', 'gold', 'score', 'no_answer_score', 'answerable', 'question_tokens', 'context_tokens'])\n"
                "    writer.writeheader()\n"
                "    for row in new_results:\n"
                "        writer.writerow({{**row, 'gold': ' | '.join(row['gold'])}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = RoBERTaQuestionAnsweringPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.answer(r['question'], r['context'], max_answer_tokens=ANSWER_MAX_TOKENS)['answer'] for r in test_records[:8]]\n"
                "after = [reloaded.answer(r['question'], r['context'], max_answer_tokens=ANSWER_MAX_TOKENS)['answer'] for r in test_records[:8]]\n"
                "parity = {{'identical_answers': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_answers'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'model_attribution': 'deepset (https://huggingface.co/deepset/roberta-base-squad2), weights redistributed unmodified under CC-BY-4.0',\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'url': CORPUS_URL, 'sha256': CORPUS_SHA256, 'license': CORPUS_LICENSE, 'members': CORPUS_MEMBERS}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'items': [{{k: r[k] for k in ('id', 'question', 'answer', 'start', 'end', 'score', 'no_answer_score', 'best_span_score', 'answerable', 'question_tokens', 'context_tokens', 'seconds')}} for r in results], 'golds': golds}},\n"
                "    'comparison': comparison,\n"
                "    'new_questions': new_metrics,\n"
                "    'single_pair_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model fails most adversarial questions — its test F1 sits near the lexical-overlap baseline and it "
        "returns the empty answer for more than half of the answerable questions — and a bounded fine-tuning of the last two "
        "encoder blocks and the span head on 1,000 in-distribution questions roughly doubles held-out F1 and lifts the "
        "answered rate to 100 % in a few minutes on CPU, with a 57 MB adapter that reloads to identical answers. That is the "
        "claim: the adaptation contract works end to end on a real gold-span corpus, and the numbers it produces are read "
        "against two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 400 questions over 16 articles from one seeded split of one corpus, the metrics are two "
        "reference-based scores (own implementations of the SQuAD 2.0 normalisation, and neither a human judgement), and "
        "AdversarialQA is Wikipedia prose with single-span gold answers and no unanswerable questions. So a gain here says "
        "the contract works, not that the adapted model is better on your documents, that it handles long or technical "
        "passages, or that its spans are faithful — a reader can return a plausible wrong span, and after this adaptation it "
        "answers every question, so the null decision the base model was trained for is degraded on this corpus and must be "
        "re-measured on unanswerable questions from your own domain before it is relied on. Fine-tuning on a narrow corpus "
        "can also erode the model elsewhere; nothing here measures that.\n\n"
        "Three things to carry to real data. **Baselines first:** the always-null and lexical-overlap baselines and the "
        "frozen model's score on *your* gold spans are the numbers to read before any adapted one, separately for "
        "answerable and unanswerable questions. **Leakage:** keep every question on a passage in one split (the contract does "
        "this) and split by document or article when your questions come from one, never at random over near-duplicate "
        "passages. **Ceilings:** passages over `MAX_CONTEXT_TOKENS` are refused at inference and dropped by the fit check "
        "before training — long-document reading is out of scope and must be chunked by the caller.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real gold-span corpus, "
        "validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning, evaluate against two trivial baselines and the frozen model on an article-disjoint split, and emit the "
        "shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, reading-comprehension accuracy on any other domain, a usable no-answer threshold, or production "
        "fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_ENCODER_LAYERS = 1` and compare the "
        "artifact size and the test scores; raise `EPOCHS` and watch the validation F1 pick the epoch; add unanswerable "
        "records to a BYOD set (empty `answers`) and read the always-null baseline and the adapted `answered_rate` together; "
        "or bring your own documents through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance and CC-BY-4.0 attribution: https://github.com/kurtvalcorza/roberta-squad2-question-answering-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (deepset, CC-BY-4.0): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/deepset-ai/haystack\n"
        "- RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019): https://arxiv.org/abs/1907.11692\n"
        "- Know What You Don't Know: Unanswerable Questions for SQuAD (Rajpurkar et al., ACL 2018): https://arxiv.org/abs/1806.03822\n"
        "- Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension (Bartolo et al., TACL 2020; AdversarialQA v1.0, CC BY-SA 3.0): https://arxiv.org/abs/2002.00293 — data: https://adversarialqa.github.io/\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
