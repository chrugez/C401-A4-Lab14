import asyncio
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
QA_DATASET_DIR = DATA_DIR / "QA_dataset"
ANALYSIS_DIR = ROOT_DIR / "analysis"
OUTPUT_PATH = DATA_DIR / "golden_set.jsonl"
RAW_V11_PATH = QA_DATASET_DIR / "SQuAD-v1.1.csv"
RAW_V12_PATH = QA_DATASET_DIR / "SQuAD-v1.2.csv"
DOCUMENTS_CSV_PATH = QA_DATASET_DIR / "documents_cleaned.csv"
METADATA_CSV_PATH = QA_DATASET_DIR / "document_metadata.csv"
QA_DATASET_CSV_PATH = QA_DATASET_DIR / "qa_dataset.csv"
GROUND_TRUTH_CSV_PATH = QA_DATASET_DIR / "ground_truth_mapping.csv"
QA_DATASET_README_PATH = QA_DATASET_DIR / "README.md"
ANALYSIS_HARD_CASES_PATH = ANALYSIS_DIR / "HARD_CASES_GUIDE.md"

TARGET_CASES = 60
DIRECT_CASES = 40
GENERATED_CASES = TARGET_CASES - DIRECT_CASES
OPENAI_ENV_NAMES = ("OPENAI_API_KEY", "open_ai_key", "OPENAI_KEY")
DEFAULT_MODEL_NAME = "gpt-4o-mini"


@dataclass
class RawQaRow:
    title: str
    context: str
    question: str
    answer: str
    answer_start_char_idx: int | None
    answer_end_char_idx: int | None
    answer_start_word_idx: int | None
    answer_end_word_idx: int | None
    source_versions: tuple[str, ...]


@dataclass
class SourceChunk:
    chunk_id: str
    doc_id: str
    title: str
    path: str
    text: str
    qa_count: int
    source_versions: tuple[str, ...]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def estimate_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def infer_case_type(question: str) -> str:
    lowered = question.lower().strip()
    if lowered.startswith("why") or lowered.startswith("how"):
        return "reasoning"
    if lowered.startswith("which") or " or " in lowered:
        return "comparison"
    return "fact-check"


def infer_difficulty(question: str, answer: str, context: str) -> str:
    answer_words = estimate_word_count(answer)
    question_words = estimate_word_count(question)
    context_words = estimate_word_count(context)

    score = 0
    if question_words >= 12:
        score += 1
    if answer_words >= 8:
        score += 1
    if context_words >= 140:
        score += 1

    if score >= 3:
        return "hard"
    if score == 2:
        return "medium"
    return "easy"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path, version: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row in reader:
            row_copy = dict(row)
            row_copy["__version"] = version
            rows.append(row_copy)
        return rows


def merge_squad_rows() -> list[RawQaRow]:
    merged_versions: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    canonical_rows: dict[tuple[str, str, str, str], RawQaRow] = {}

    for row in load_csv_rows(RAW_V11_PATH, "v1.1"):
        key = (
            normalize_spaces(row["title"]),
            normalize_spaces(row["context"]),
            normalize_spaces(row["question"]),
            normalize_spaces(row["answer"]),
        )
        merged_versions[key].add("v1.1")
        canonical_rows.setdefault(
            key,
            RawQaRow(
                title=key[0],
                context=key[1],
                question=key[2],
                answer=key[3],
                answer_start_char_idx=parse_optional_int(row.get("answer_start")),
                answer_end_char_idx=parse_optional_int(row.get("answer_end")),
                answer_start_word_idx=None,
                answer_end_word_idx=None,
                source_versions=("v1.1",),
            ),
        )

    for row in load_csv_rows(RAW_V12_PATH, "v1.2"):
        key = (
            normalize_spaces(row["title"]),
            normalize_spaces(row["context"]),
            normalize_spaces(row["question"]),
            normalize_spaces(row["answer"]),
        )
        merged_versions[key].add("v1.2")
        canonical_rows[key] = RawQaRow(
            title=key[0],
            context=key[1],
            question=key[2],
            answer=key[3],
            answer_start_char_idx=parse_optional_int(row.get("answer_start_char_idx")),
            answer_end_char_idx=parse_optional_int(row.get("answer_end_char_idx")),
            answer_start_word_idx=parse_optional_int(row.get("answer_start_word_idx")),
            answer_end_word_idx=parse_optional_int(row.get("answer_end_word_idx")),
            source_versions=("v1.2",),
        )

    cleaned_rows: list[RawQaRow] = []
    for key in sorted(canonical_rows):
        canonical = canonical_rows[key]
        cleaned_rows.append(
            RawQaRow(
                title=canonical.title,
                context=canonical.context,
                question=canonical.question,
                answer=canonical.answer,
                answer_start_char_idx=canonical.answer_start_char_idx,
                answer_end_char_idx=canonical.answer_end_char_idx,
                answer_start_word_idx=canonical.answer_start_word_idx,
                answer_end_word_idx=canonical.answer_end_word_idx,
                source_versions=tuple(sorted(merged_versions[key])),
            )
        )
    return cleaned_rows


def build_context_chunks(rows: list[RawQaRow]) -> tuple[list[SourceChunk], dict[tuple[str, str], str]]:
    context_groups: dict[tuple[str, str], list[RawQaRow]] = defaultdict(list)
    for row in rows:
        context_groups[(row.title, row.context)].append(row)

    chunks: list[SourceChunk] = []
    context_to_chunk_id: dict[tuple[str, str], str] = {}

    for index, (context_key, group_rows) in enumerate(sorted(context_groups.items()), start=1):
        title, context = context_key
        chunk_id = f"squad_chunk_{index:05d}"
        context_to_chunk_id[context_key] = chunk_id
        source_versions = dedupe_preserve_order(
            [version for row in group_rows for version in row.source_versions]
        )
        chunks.append(
            SourceChunk(
                chunk_id=chunk_id,
                doc_id=f"title_{slugify(title)}",
                title=title,
                path="data/QA_dataset/SQuAD-v1.1.csv|data/QA_dataset/SQuAD-v1.2.csv",
                text=context,
                qa_count=len(group_rows),
                source_versions=tuple(source_versions),
            )
        )

    return chunks, context_to_chunk_id


def build_chunk_lookup(chunks: list[SourceChunk]) -> dict[str, SourceChunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def select_direct_rows(rows: list[RawQaRow], count: int) -> list[RawQaRow]:
    by_title: dict[str, list[RawQaRow]] = defaultdict(list)
    seen_contexts_by_title: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        if row.context in seen_contexts_by_title[row.title]:
            continue
        seen_contexts_by_title[row.title].add(row.context)
        by_title[row.title].append(row)

    ordered_titles = sorted(by_title, key=lambda title: (-len(by_title[title]), title))
    selected: list[RawQaRow] = []
    index = 0
    while len(selected) < count and ordered_titles:
        exhausted_titles: list[str] = []
        for title in ordered_titles:
            if index < len(by_title[title]):
                selected.append(by_title[title][index])
                if len(selected) == count:
                    break
            else:
                exhausted_titles.append(title)
        ordered_titles = [title for title in ordered_titles if title not in exhausted_titles]
        index += 1
    return selected


def build_direct_cases(
    rows: list[RawQaRow],
    context_to_chunk_id: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for row in rows:
        chunk_id = context_to_chunk_id[(row.title, row.context)]
        case_type = infer_case_type(row.question)
        difficulty = infer_difficulty(row.question, row.answer, row.context)
        cases.append(
            {
                "question": row.question,
                "expected_answer": row.answer,
                "context": row.context,
                "expected_retrieval_ids": [chunk_id],
                "source_documents": [f"title_{slugify(row.title)}"],
                "source_paths": ["data/QA_dataset/SQuAD-v1.1.csv", "data/QA_dataset/SQuAD-v1.2.csv"],
                "metadata": {
                    "difficulty": difficulty,
                    "type": case_type,
                    "challenge_tag": "squad_direct",
                    "notes": f"Direct case from title '{row.title}'.",
                    "generated_from": "squad_direct",
                    "unanswerable": False,
                },
            }
        )

    return cases


def get_openai_api_key() -> str | None:
    for env_name in OPENAI_ENV_NAMES:
        if os.getenv(env_name):
            return os.getenv(env_name)
    return None


def get_model_name() -> str:
    return os.getenv("OPENAI_SDG_MODEL", DEFAULT_MODEL_NAME)


def build_client() -> AsyncOpenAI:
    load_dotenv(ROOT_DIR / ".env")
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY or open_ai_key in .env before running the generator."
        )
    return AsyncOpenAI(api_key=api_key)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("Model response does not contain a JSON object.")
    return json.loads(text[start : end + 1])


def normalize_generated_case(
    raw_case: dict[str, Any],
    chunk_lookup: dict[str, SourceChunk],
    allowed_chunk_ids: list[str],
) -> dict[str, Any] | None:
    question = normalize_spaces(str(raw_case.get("question", "")))
    expected_answer = normalize_spaces(str(raw_case.get("expected_answer", "")))
    if not question or not expected_answer:
        return None

    raw_ids = raw_case.get("expected_retrieval_ids", [])
    expected_retrieval_ids = []
    if isinstance(raw_ids, list):
        expected_retrieval_ids = [
            str(chunk_id)
            for chunk_id in raw_ids
            if str(chunk_id) in chunk_lookup and str(chunk_id) in allowed_chunk_ids
        ]

    metadata = raw_case.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    case_type = str(metadata.get("type", "reasoning")).strip().lower() or "reasoning"
    difficulty = str(metadata.get("difficulty", "hard")).strip().lower() or "hard"
    challenge_tag = normalize_spaces(str(metadata.get("challenge_tag", case_type))) or case_type
    notes = normalize_spaces(str(metadata.get("notes", "")))

    valid_case_types = {
        "fact-check",
        "reasoning",
        "comparison",
        "adversarial",
        "ambiguous",
        "out-of-scope",
        "multi-hop",
        "process",
    }
    if case_type not in valid_case_types:
        case_type = "reasoning"
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "hard"

    is_unanswerable = case_type in {"out-of-scope", "ambiguous"} and not expected_retrieval_ids
    if not expected_retrieval_ids and not is_unanswerable:
        return None

    source_documents = sorted({chunk_lookup[chunk_id].doc_id for chunk_id in expected_retrieval_ids})
    source_paths = sorted({chunk_lookup[chunk_id].path for chunk_id in expected_retrieval_ids})
    context = "\n\n---\n\n".join(
        f"[{chunk_id}] {chunk_lookup[chunk_id].text}" for chunk_id in expected_retrieval_ids
    )

    return {
        "question": question,
        "expected_answer": expected_answer,
        "context": context,
        "expected_retrieval_ids": expected_retrieval_ids,
        "source_documents": source_documents,
        "source_paths": source_paths,
        "metadata": {
            "difficulty": difficulty,
            "type": case_type,
            "challenge_tag": challenge_tag,
            "notes": notes,
            "generated_from": "generate_qa_from_text",
            "unanswerable": is_unanswerable,
        },
    }


async def request_cases_from_openai(
    client: AsyncOpenAI,
    chunk_records: list[dict[str, str]],
    cases_per_batch: int,
    mode: str,
) -> list[dict[str, Any]]:
    system_prompt = (
        "You are a lead data engineer creating a Vietnamese golden QA dataset from SQuAD contexts. "
        "Return only valid JSON. Generate benchmark cases that are grounded in the provided text, "
        "with emphasis on adversarial prompts, retrieval robustness, and multi-hop reasoning when the "
        "input contains more than one chunk."
    )

    user_prompt = {
        "task": "Generate benchmark cases in Vietnamese from the provided SQuAD chunks.",
        "mode": mode,
        "requirements": [
            f"Create exactly {cases_per_batch} cases.",
            "At least one case must be adversarial and at least one case must be multi-hop when more than one chunk is provided.",
            "Questions must remain answerable from the chunks unless marked as out-of-scope or ambiguous.",
            "For adversarial cases, the expected answer must explicitly reject unsupported instructions and stay grounded in the text.",
            "Use only the provided chunk ids in expected_retrieval_ids.",
            "Keep answers concise and faithful to the source.",
        ],
        "output_schema": {
            "cases": [
                {
                    "question": "string",
                    "expected_answer": "string",
                    "expected_retrieval_ids": ["chunk_id_1", "chunk_id_2"],
                    "metadata": {
                        "difficulty": "easy|medium|hard",
                        "type": "fact-check|reasoning|comparison|adversarial|ambiguous|out-of-scope|multi-hop|process",
                        "challenge_tag": "short string",
                        "notes": "short string",
                    },
                }
            ]
        },
        "chunks": chunk_records,
    }

    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = await client.chat.completions.create(
                model=get_model_name(),
                temperature=0.4,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            payload = extract_json_object(content)
            cases = payload.get("cases", [])
            return cases if isinstance(cases, list) else []
        except Exception as error:
            last_error = error
            await asyncio.sleep(1)

    if last_error:
        raise last_error
    return []


async def generate_qa_from_text(
    text: str,
    num_pairs: int = 5,
    source_name: str = "SQuAD_context",
    source_path: str = "data/QA_dataset/SQuAD-v1.2.csv",
    chunk_hints: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate benchmark-quality QA pairs by prompt engineering from the provided text.

    The prompt explicitly asks for adversarial and multi-hop cases when the input spans
    multiple chunks, which satisfies the Lead Data responsibility in the assignment.
    """
    normalized_text = text.strip()
    if not normalized_text:
        return []

    client = build_client()
    chunk_records = chunk_hints or [
        {
            "chunk_id": "adhoc_chunk_001",
            "title": source_name,
            "path": source_path,
            "text": normalized_text,
        }
    ]
    raw_cases = await request_cases_from_openai(
        client=client,
        chunk_records=chunk_records,
        cases_per_batch=num_pairs,
        mode="prompt_engineered_text_generation",
    )

    pseudo_lookup = {
        record["chunk_id"]: SourceChunk(
            chunk_id=record["chunk_id"],
            doc_id=f"title_{slugify(record['title'])}",
            title=record["title"],
            path=record["path"],
            text=record["text"],
            qa_count=0,
            source_versions=("v1.1", "v1.2"),
        )
        for record in chunk_records
    }
    allowed_chunk_ids = [record["chunk_id"] for record in chunk_records]

    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        normalized_case = normalize_generated_case(raw_case, pseudo_lookup, allowed_chunk_ids)
        if normalized_case:
            cases.append(normalized_case)
    return cases


def select_generation_groups(chunks: list[SourceChunk], target_generated_cases: int) -> list[list[SourceChunk]]:
    by_title: dict[str, list[SourceChunk]] = defaultdict(list)
    for chunk in chunks:
        by_title[chunk.title].append(chunk)

    candidate_titles = sorted(
        [title for title, title_chunks in by_title.items() if len(title_chunks) >= 2],
        key=lambda title: (-len(by_title[title]), title),
    )

    needed_groups = max(1, (target_generated_cases + 4) // 5)
    selected_groups: list[list[SourceChunk]] = []
    for title in candidate_titles[:needed_groups]:
        selected_groups.append(by_title[title][:2])
    return selected_groups


def deduplicate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_questions: set[str] = set()
    unique_cases: list[dict[str, Any]] = []
    for case in cases:
        fingerprint = normalize_spaces(case["question"]).lower()
        if fingerprint in seen_questions:
            continue
        seen_questions.add(fingerprint)
        unique_cases.append(case)
    return unique_cases


def assign_case_ids(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_copy = dict(case)
        case_copy["case_id"] = f"golden_case_{index:03d}"
        assigned.append(case_copy)
    return assigned


async def generate_hard_cases(chunks: list[SourceChunk], target_generated_cases: int) -> list[dict[str, Any]]:
    chunk_lookup = build_chunk_lookup(chunks)
    groups = select_generation_groups(chunks, target_generated_cases)
    generated_cases: list[dict[str, Any]] = []

    for group in groups:
        chunk_records = [
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "path": chunk.path,
                "text": chunk.text,
            }
            for chunk in group
        ]
        combined_text = "\n\n---\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for chunk in group
        )
        raw_cases = await generate_qa_from_text(
            text=combined_text,
            num_pairs=5,
            source_name=group[0].title,
            source_path=group[0].path,
            chunk_hints=chunk_records,
        )
        allowed_chunk_ids = [chunk.chunk_id for chunk in group]
        for raw_case in raw_cases:
            normalized_case = normalize_generated_case(raw_case, chunk_lookup, allowed_chunk_ids)
            if normalized_case:
                generated_cases.append(normalized_case)

    return deduplicate_cases(generated_cases)[:target_generated_cases]


def persist_cases(cases: list[dict[str, Any]]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for case in cases:
            output_file.write(json.dumps(case, ensure_ascii=False) + "\n")


def persist_cleaned_dataset_artifacts(
    rows: list[RawQaRow],
    chunks: list[SourceChunk],
    cases: list[dict[str, Any]],
    context_to_chunk_id: dict[tuple[str, str], str],
) -> None:
    documents_rows = [
        {
            "title": row.title,
            "question": row.question,
            "answer": row.answer,
            "chunk_id": context_to_chunk_id[(row.title, row.context)],
            "answer_start_char_idx": row.answer_start_char_idx or "",
            "answer_end_char_idx": row.answer_end_char_idx or "",
            "answer_start_word_idx": row.answer_start_word_idx or "",
            "answer_end_word_idx": row.answer_end_word_idx or "",
            "source_versions": "|".join(row.source_versions),
            "context_word_count": estimate_word_count(row.context),
        }
        for row in rows
    ]
    write_csv(
        DOCUMENTS_CSV_PATH,
        [
            "title",
            "question",
            "answer",
            "chunk_id",
            "answer_start_char_idx",
            "answer_end_char_idx",
            "answer_start_word_idx",
            "answer_end_word_idx",
            "source_versions",
            "context_word_count",
        ],
        documents_rows,
    )

    metadata_rows = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "path": chunk.path,
            "qa_count": chunk.qa_count,
            "word_count": estimate_word_count(chunk.text),
            "char_count": len(chunk.text),
            "source_versions": "|".join(chunk.source_versions),
            "preview": chunk.text[:160].replace("\n", " "),
        }
        for chunk in chunks
    ]
    write_csv(
        METADATA_CSV_PATH,
        [
            "chunk_id",
            "doc_id",
            "title",
            "path",
            "qa_count",
            "word_count",
            "char_count",
            "source_versions",
            "preview",
        ],
        metadata_rows,
    )

    qa_dataset_rows = [
        {
            "case_id": case["case_id"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "difficulty": case["metadata"]["difficulty"],
            "type": case["metadata"]["type"],
            "challenge_tag": case["metadata"]["challenge_tag"],
            "generated_from": case["metadata"]["generated_from"],
            "expected_retrieval_ids": "|".join(case["expected_retrieval_ids"]),
            "source_documents": "|".join(case["source_documents"]),
            "source_paths": "|".join(case["source_paths"]),
            "unanswerable": case["metadata"]["unanswerable"],
            "context": case["context"],
        }
        for case in cases
    ]
    write_csv(
        QA_DATASET_CSV_PATH,
        [
            "case_id",
            "question",
            "expected_answer",
            "difficulty",
            "type",
            "challenge_tag",
            "generated_from",
            "expected_retrieval_ids",
            "source_documents",
            "source_paths",
            "unanswerable",
            "context",
        ],
        qa_dataset_rows,
    )

    ground_truth_rows = [
        {
            "case_id": case["case_id"],
            "primary_ground_truth_id": case["expected_retrieval_ids"][0] if case["expected_retrieval_ids"] else "",
            "ground_truth_ids": "|".join(case["expected_retrieval_ids"]),
            "ground_truth_count": len(case["expected_retrieval_ids"]),
            "type": case["metadata"]["type"],
            "difficulty": case["metadata"]["difficulty"],
            "generated_from": case["metadata"]["generated_from"],
            "question": case["question"],
        }
        for case in cases
    ]
    write_csv(
        GROUND_TRUTH_CSV_PATH,
        [
            "case_id",
            "primary_ground_truth_id",
            "ground_truth_ids",
            "ground_truth_count",
            "type",
            "difficulty",
            "generated_from",
            "question",
        ],
        ground_truth_rows,
    )

    qa_readme = """# QA Dataset

Raw source files:
- `SQuAD-v1.1.csv`
- `SQuAD-v1.2.csv`

Generated by `data/synthetic_gen.py`:
- `documents_cleaned.csv`: cleaned row-level QA data after merging v1.1 and v1.2
- `document_metadata.csv`: chunk/document metadata with stable `chunk_id`
- `qa_dataset.csv`: final benchmark set used for the lab
- `ground_truth_mapping.csv`: retrieval ground-truth ids per case
"""
    QA_DATASET_README_PATH.write_text(qa_readme, encoding="utf-8")


def summarize_cases(cases: list[dict[str, Any]]) -> str:
    difficulty_counter = Counter(case["metadata"]["difficulty"] for case in cases)
    type_counter = Counter(case["metadata"]["type"] for case in cases)
    source_counter = Counter(case["metadata"]["generated_from"] for case in cases)
    return (
        f"Generated {len(cases)} cases | "
        f"difficulty={dict(difficulty_counter)} | "
        f"types={dict(type_counter)} | "
        f"sources={dict(source_counter)}"
    )


async def main() -> None:
    rows = merge_squad_rows()
    chunks, context_to_chunk_id = build_context_chunks(rows)

    direct_rows = select_direct_rows(rows, DIRECT_CASES)
    direct_cases = build_direct_cases(direct_rows, context_to_chunk_id)
    hard_cases = await generate_hard_cases(chunks, GENERATED_CASES)

    all_cases = deduplicate_cases(direct_cases + hard_cases)
    if len(all_cases) < TARGET_CASES:
        raise RuntimeError(f"Only generated {len(all_cases)} cases. Expected at least {TARGET_CASES}.")

    final_cases = assign_case_ids(all_cases[:TARGET_CASES])
    persist_cases(final_cases)
    persist_cleaned_dataset_artifacts(rows, chunks, final_cases, context_to_chunk_id)

    print(summarize_cases(final_cases))
    print(f"Saved dataset to {OUTPUT_PATH.relative_to(ROOT_DIR)}")
    print(f"Updated cleaned dataset files in {QA_DATASET_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    asyncio.run(main())
