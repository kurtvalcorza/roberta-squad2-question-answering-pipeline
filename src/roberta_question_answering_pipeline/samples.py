"""Extractive-QA dataset contract for fine-tuning: the pinned AdversarialQA (dRoBERTa) sample, validation,
seeded context-disjoint splitting, BYOD loaders and CSV export.

The default dataset is **real**: AdversarialQA v1.0 (Bartolo et al., TACL 2020, CC BY-SA 3.0) — SQuAD-style
question/answer pairs over Wikipedia passages, written by annotators who could see a model's predictions and
kept only the questions that model got wrong. The `3_droberta` subset was collected against a RoBERTa reader,
so it is the hardest of the three for this pipeline's `roberta-base-squad2` checkpoint: the frozen model's
exact-match/F1 here sits far below its SQuAD 2.0 dev figures, which is what makes it an honest adaptation
target. One 9.0 MB zip is fetched from the AdversarialQA site, refused on any byte-size or SHA-256 mismatch,
and two members are read without extracting to disk: `3_droberta/train.json` (10,000 questions, the
training pool) and `3_droberta/dev.json` (1,000 questions over 21 articles, split here by article into
validation and test). The `test.json` member ships without answers and is not used.

A record is ``{id, question, context, answers}`` with ``answers`` a list of ``{text, answer_start}`` (the
SQuAD convention; an empty list means unanswerable — AdversarialQA has none, so the null answer scores 0).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_CONTEXT_CHARS, MAX_QUESTION_CHARS, MODEL_ID

CORPUS_NAME = "AdversarialQA"
CORPUS_RELEASE = "v1.0 (2020-09-23)"
CORPUS_URL = "https://adversarialqa.github.io/data/aqa_v1.0.zip"
CORPUS_BYTES = 9_018_914
CORPUS_SHA256 = "f4f3c23224a5060b28c35e35581bd5cf46256dda3665418fb83d036d0e0c93cf"
CORPUS_MEMBERS = {"train": "3_droberta/train.json", "dev": "3_droberta/dev.json"}
CORPUS_LICENSE = "CC BY-SA 3.0 (Bartolo et al. 2020; adversarialqa.github.io)"
CORPUS_QUESTIONS = {"train": 10_000, "dev": 1_000}
DEFAULT_CACHE_DIR = Path("weights") / "adversarialqa"
# Sample filters: the pipeline refuses pairs over MAX_QUESTION_TOKENS (64) / MAX_CONTEXT_TOKENS (384) with the
# real tokenizer, so the sample keeps passages short enough that none is refused (about 4 chars per BPE
# token on English Wikipedia text; the build record found 0 of the sampled pairs over either ceiling).
MAX_SAMPLE_CONTEXT_CHARS = 1_300
MAX_SAMPLE_QUESTION_CHARS = 200
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 1_000, "validation": 200, "test": 400}
MIN_RECORDS = 8
MAX_RECORDS = 20_000
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """Return the pinned AdversarialQA zip bytes from the cache or the project site, digest-verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "aqa_v1.0.zip"
    if local.is_file():
        data = local.read_bytes()
        if len(data) == CORPUS_BYTES and _sha256_bytes(data) == CORPUS_SHA256:
            return data
    if fetcher is not None:
        data = fetcher(CORPUS_URL)
    else:
        with urllib.request.urlopen(CORPUS_URL, timeout=180) as response:  # noqa: S310 (pinned https URL)
            data = response.read()
    if len(data) != CORPUS_BYTES or _sha256_bytes(data) != CORPUS_SHA256:
        raise ValueError(
            f"corpus zip: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
            f"pinned {CORPUS_BYTES} / {CORPUS_SHA256[:16]}…"
        )
    local.write_bytes(data)
    return data


def flatten_squad(data: Mapping[str, Any], *, prefix: str = "") -> list[dict[str, Any]]:
    """SQuAD-format ``{"data": [{"title", "paragraphs": [{"context", "qas": [...]}]}]}`` → flat records
    carrying the article ``title`` (used for article-disjoint splitting)."""
    if not isinstance(data, Mapping) or "data" not in data:
        raise ValueError("SQuAD-format JSON must be an object with a 'data' list")
    if not isinstance(data, Mapping) or "data" not in data:
        raise ValueError("SQuAD-format JSON must be an object with a 'data' list")
    records = []
    for article in data["data"]:
        title = str(article.get("title", ""))
        for paragraph in article["paragraphs"]:
            context = str(paragraph["context"])
            for qa in paragraph["qas"]:
                records.append(
                    {
                        "id": f"{prefix}{qa['id']}",
                        "question": str(qa["question"]),
                        "context": context,
                        "answers": [
                            {"text": str(a["text"]), "answer_start": int(a["answer_start"])}
                            for a in qa.get("answers", [])
                        ],
                        "title": title,
                    }
                )
    return records


def read_corpus(data: bytes) -> dict[str, list[dict[str, Any]]]:
    """The dRoBERTa train and dev members as flat records, read without extracting to disk."""
    archive = zipfile.ZipFile(io.BytesIO(data))
    names = set(archive.namelist())
    for member in CORPUS_MEMBERS.values():
        if member not in names:
            raise ValueError(f"corpus zip is missing member {member}")
    out = {}
    for split, member in CORPUS_MEMBERS.items():
        out[split] = flatten_squad(json.loads(archive.read(member).decode("utf-8")), prefix=f"{split}-")
        if len(out[split]) != CORPUS_QUESTIONS[split]:
            raise ValueError(f"{member}: {len(out[split])} questions, expected {CORPUS_QUESTIONS[split]}")
    return out


def filter_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep answerable records whose passage and question fit the sample character filters and whose
    every answer is found at its offset; drop exact duplicate (context, question) pairs."""
    seen: set[tuple[str, str]] = set()
    kept = []
    for record in records:
        context, question = str(record["context"]), str(record["question"])
        if len(context) > MAX_SAMPLE_CONTEXT_CHARS or len(question) > MAX_SAMPLE_QUESTION_CHARS:
            continue
        answers = list(record.get("answers", []))
        if not answers or not all(_answer_at_offset(context, a) for a in answers):
            continue
        key = (context.lower(), question.lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append(dict(record))
    return kept


def _answer_at_offset(context: str, answer: Mapping[str, Any]) -> bool:
    start, text = int(answer["answer_start"]), str(answer["text"])
    return bool(text) and context[start : start + len(text)] == text


def _group_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get(key, "")).lower(), []).append(dict(record))
    return groups


def build_sample_dataset(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Training records: a seeded subset of the filtered dRoBERTa training pool. Validation and test: the
    filtered dev questions cut **by article** (a seeded shuffle of article titles fills validation first), so
    no passage — and no article — is shared between validation and test; the training pool's articles are
    disjoint from dev by AdversarialQA's construction, which `check_split_disjoint` re-checks on passages."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    train_pool = filter_records(corpus["train"])
    if sizes["train"] > len(train_pool):
        raise ValueError(f"requested {sizes['train']} training records but only {len(train_pool)} fit")
    rng.shuffle(train_pool)
    dev_pool = filter_records(corpus["dev"])
    groups = _group_by(dev_pool, "title")
    titles = sorted(groups)
    rng.shuffle(titles)
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for title in titles:
        (validation if len(validation) < sizes["validation"] else test).extend(groups[title])
    if len(test) < sizes["test"] or len(validation) < sizes["validation"]:
        raise ValueError(
            f"dev pool yields {len(validation)} validation / {len(test)} test records; "
            f"requested {sizes['validation']} / {sizes['test']}"
        )
    rng.shuffle(validation)
    rng.shuffle(test)
    splits = {
        "train": train_pool[: sizes["train"]],
        "validation": validation[: sizes["validation"]],
        "test": test[: sizes["test"]],
    }
    return {
        name: [
            {
                "id": f"{name}-{i:04d}",
                "question": r["question"],
                "context": r["context"],
                "answers": r["answers"],
                "title": r.get("title", ""),
            }
            for i, r in enumerate(part)
        ]
        for name, part in splits.items()
    }


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/question/context/answers")
    for key in ("id", "question", "context", "answers"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid, question, context, answers = record["id"], record["question"], record["context"], record["answers"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    for key, value, ceiling in (
        ("question", question, MAX_QUESTION_CHARS),
        ("context", context, MAX_CONTEXT_CHARS),
    ):
        if not isinstance(value, str):
            raise ValueError(f"{label}: {key} must be a string")
        if not value.strip():
            raise ValueError(f"{label}: {key} is empty")
        if len(value) > ceiling:
            raise ValueError(
                f"{label}: {key} has {len(value)} chars; ceiling is MAX_{key.upper()}_CHARS={ceiling}"
            )
    if isinstance(answers, Mapping) or not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        raise ValueError(
            f"{label}: answers must be a list of {{text, answer_start}} (empty for unanswerable)"
        )
    checked_answers = []
    for j, answer in enumerate(answers):
        if not isinstance(answer, Mapping) or "text" not in answer or "answer_start" not in answer:
            raise ValueError(f"{label}: answers[{j}] must be a mapping with text and answer_start")
        start, text = answer["answer_start"], answer["text"]
        if isinstance(start, bool) or not isinstance(start, int) or not isinstance(text, str):
            raise ValueError(f"{label}: answers[{j}] needs an int answer_start and a str text")
        if not _answer_at_offset(context, answer):
            raise ValueError(f"{label}: answers[{j}] text {text[:40]!r} is not found at offset {start}")
        checked_answers.append({"text": text, "answer_start": start})
    item = {"id": rid, "question": question, "context": context, "answers": checked_answers}
    if "title" in record:
        item["title"] = str(record["title"])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a QA dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, question, context, answers} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    contexts: set[str] = set()
    unanswerable = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        contexts.add(item["context"].lower())
        unanswerable += not item["answers"]
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_contexts": len(contexts),
        "unanswerable": unanswerable,
        "question_chars": {
            "min": min(len(r["question"]) for r in checked),
            "max": max(len(r["question"]) for r in checked),
        },
        "context_chars": {
            "min": min(len(r["context"]) for r in checked),
            "max": max(len(r["context"]) for r in checked),
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [r["id"], r["question"], r["context"], [[a["text"], a["answer_start"]] for a in r["answers"]]]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def gold_texts(record: Mapping[str, Any]) -> list[str]:
    """The SQuAD-style gold list for the metric helpers (empty for unanswerable)."""
    return [a["text"] for a in record["answers"]]


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no lower-cased passage appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record["context"]).lower()
            if key in seen and seen[key] != name:
                raise ValueError(
                    f"a passage ({record['context'][:60]!r}…) appears in both {seen[key]} and {name}"
                )
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of a BYOD dataset into train/validation/test **by passage**: every question on the
    same passage lands in the same split, so a test passage is never seen in training."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    groups = list(_group_by(checked, "context").values())
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(checked) * test_fraction))
    n_val = round(len(checked) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for group in groups:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["validation"]) < n_val:
            splits["validation"].extend(group)
        else:
            splits["train"].extend(group)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSON array of ``{id, question, context, answers}``, a SQuAD-format JSON file
    (``{"data": [...]}``), JSONL, or a CSV with columns ``id, question, context, answer_text, answer_start``
    (one answer per row; an empty ``answer_text`` means unanswerable)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        missing = {"id", "question", "context", "answer_text", "answer_start"} - set(
            rows[0].keys() if rows else set()
        )
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)}")
        return [
            {
                "id": r["id"],
                "question": r["question"],
                "context": r["context"],
                "answers": (
                    [{"text": r["answer_text"], "answer_start": int(r["answer_start"])}]
                    if r["answer_text"]
                    else []
                ),
            }
            for r in rows
        ]
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, Mapping) and "data" in data:
            return flatten_squad(data)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records or a SQuAD-format object")
        return data
    raise ValueError("BYOD datasets must be .csv, .json or .jsonl")


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One row per record with its first gold answer (empty for unanswerable)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "question", "context", "answer_text", "answer_start"]
        )
        writer.writeheader()
        for record in records:
            first = record["answers"][0] if record["answers"] else {"text": "", "answer_start": ""}
            writer.writerow(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "context": record["context"],
                    "answer_text": first["text"],
                    "answer_start": first["answer_start"],
                }
            )
    return out
