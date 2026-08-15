from pathlib import Path

from src.preprocessing import load_jsonl
from src.generator import AnswerGenerator
from src.evaluation import (
    load_jsonl_records,
    token_f1,
)


CORPUS_PATH = Path(
    "data/scientific/processed/"
    "scientific_chunks.jsonl"
)

EVAL_PATH = Path(
    "data/evaluation/"
    "scientific_eval.jsonl"
)


corpus = load_jsonl(
    CORPUS_PATH
)

evaluation = load_jsonl_records(
    EVAL_PATH
)


print(
    f"Loaded {len(corpus)} "
    f"corpus chunks."
)

print(
    f"Loaded {len(evaluation)} "
    f"evaluation questions."
)


print(
    "\nLoading answer generator..."
)

generator = AnswerGenerator()


def find_gold_chunks(example):
    """
    Return every corpus chunk manually annotated
    as relevant for this question.
    """

    expected_file = example[
        "expected_source_file"
    ]

    gold_locations = {
        (
            gold["page"],
            gold["chunk_id"]
        )
        for gold
        in example[
            "relevant_chunks"
        ]
    }

    matches = []

    for item in corpus:

        if (
            item.get("source_file")
            == expected_file
            and
            (
                item.get("page"),
                item.get("chunk_id")
            )
            in gold_locations
        ):
            matches.append(item)

    return matches


def select_best_gold_chunk(
    example,
    gold_chunks
):
    """
    Oracle evaluation should give the generator
    an evidence passage that actually contains
    the reference information.

    When several chunks are annotated as relevant,
    select the one with the strongest lexical
    overlap with the reference answer.
    """

    if not gold_chunks:
        return None

    reference = example[
        "reference_answer"
    ]

    return max(
        gold_chunks,
        key=lambda item: token_f1(
            item["text"],
            reference
        )
    )


answerable_total = 0
f1_sum = 0.0
missing_gold = 0


for example in evaluation:

    if (
        example["type"]
        == "out_of_domain"
    ):
        continue

    answerable_total += 1

    gold_chunks = find_gold_chunks(
        example
    )

    gold_chunk = (
        select_best_gold_chunk(
            example,
            gold_chunks
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{example['id']} | "
        f"{example['type']}"
    )

    print(
        "QUESTION:",
        example["question"],
    )

    if gold_chunk is None:

        missing_gold += 1

        print(
            "ERROR: "
            "No gold chunk found."
        )

        continue

    # IMPORTANT:
    # AnswerGenerator expects contexts.
    # Pass a LIST containing the full chunk,
    # never the raw string directly.
    generated_answer = (
        generator.generate(
            example["question"],
            [
                gold_chunk["text"]
            ]
        )
    )

    reference_answer = (
        example[
            "reference_answer"
        ]
    )

    f1 = token_f1(
        generated_answer,
        reference_answer
    )

    f1_sum += f1

    print(
        "GENERATED:",
        generated_answer,
    )

    print(
        "REFERENCE:",
        reference_answer,
    )

    print(
        f"TOKEN F1: "
        f"{f1:.3f}"
    )

    print(
        "GOLD SOURCE:",
        gold_chunk[
            "source_file"
        ],
    )

    print(
        "PAGE:",
        gold_chunk["page"],
    )

    print(
        "CHUNK:",
        gold_chunk["chunk_id"],
    )


print(
    "\n"
    + "=" * 100
)

print(
    "ORACLE GENERATION SUMMARY"
)

print(
    "=" * 100
)

print(
    f"Answerable questions: "
    f"{answerable_total}"
)

print(
    f"Missing gold chunks: "
    f"{missing_gold}"
)

if answerable_total:

    average_f1 = (
        f1_sum
        / answerable_total
    )

    print(
        f"Average token F1: "
        f"{average_f1:.3f}"
    )