from pathlib import Path

from src.preprocessing import load_jsonl
from src.rag import RAGSystem
from src.evaluation import (
    load_jsonl_records,
    token_f1,
    item_is_relevant,
    is_expected_refusal,
)


CORPUS_PATH = Path(
    "data/scientific/processed/"
    "scientific_chunks.jsonl"
)

EVAL_PATH = Path(
    "data/evaluation/"
    "scientific_eval.jsonl"
)

RELEVANCE_THRESHOLD = 0.30
RERANKER_CONFIDENCE_THRESHOLD = 0.0
CANDIDATE_POOL = 5


corpus = load_jsonl(
    CORPUS_PATH
)

evaluation = load_jsonl_records(
    EVAL_PATH
)


print(
    f"Loaded {len(corpus)} corpus chunks."
)

print(
    f"Loaded {len(evaluation)} "
    f"evaluation questions."
)


print(
    "\nBuilding Scientific RAG..."
)

rag = RAGSystem(
    corpus,
    relevance_threshold=(
        RELEVANCE_THRESHOLD
    ),
    use_reranker=True,
    reranker_confidence_threshold=(
        RERANKER_CONFIDENCE_THRESHOLD
    ),
    candidate_pool=(
        CANDIDATE_POOL
    ),
)


answerable_total = 0

f1_sum = 0.0

source_correct = 0

ood_total = 0
ood_correct = 0


for example in evaluation:

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


    result = rag.answer(
        example["question"],
        k=3,
    )


    generated_answer = (
        result["answer"]
    )

    reference_answer = (
        example[
            "reference_answer"
        ]
    )


    print(
        "GENERATED:",
        generated_answer,
    )

    print(
        "REFERENCE:",
        reference_answer,
    )

    print(
        "SELECTION:",
        result.get(
            "selection_method"
        ),
    )


    # -------------------------------------------------------------
    # Out-of-domain evaluation
    # -------------------------------------------------------------

    if (
        example["type"]
        == "out_of_domain"
    ):

        ood_total += 1

        correct_refusal = (
            is_expected_refusal(
                generated_answer
            )
        )

        if correct_refusal:
            ood_correct += 1

        print(
            "OOD REFUSAL:",
            correct_refusal,
        )

        continue


    # -------------------------------------------------------------
    # Answer evaluation
    # -------------------------------------------------------------

    answerable_total += 1

    f1 = token_f1(
        generated_answer,
        reference_answer,
    )

    f1_sum += f1

    print(
        f"TOKEN F1: "
        f"{f1:.3f}"
    )


    # -------------------------------------------------------------
    # Evidence/source evaluation
    # -------------------------------------------------------------

    sources = result.get(
        "sources",
        [],
    )

    source_is_correct = False

    if sources:

        top_source = (
            sources[0]
        )

        source_is_correct = (
            item_is_relevant(
                top_source,
                example,
            )
        )

        if source_is_correct:
            source_correct += 1

        print(
            "SOURCE:",
            top_source.get(
                "source_file"
            ),
        )

        print(
            "PAGE:",
            top_source.get(
                "page"
            ),
        )

        print(
            "CHUNK:",
            top_source.get(
                "chunk_id"
            ),
        )

    print(
        "SOURCE CORRECT:",
        source_is_correct,
    )


# -----------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------

print(
    "\n"
    + "=" * 100
)

print(
    "END-TO-END SCIENTIFIC RAG SUMMARY"
)

print(
    "=" * 100
)


if answerable_total:

    average_f1 = (
        f1_sum
        / answerable_total
    )

    source_accuracy = (
        source_correct
        / answerable_total
    )

    print(
        f"Answerable questions: "
        f"{answerable_total}"
    )

    print(
        f"Average token F1: "
        f"{average_f1:.3f}"
    )

    print(
        f"Source accuracy: "
        f"{source_accuracy:.3f}"
    )


if ood_total:

    print(
        f"OOD questions: "
        f"{ood_total}"
    )

    print(
        "OOD refusal accuracy: "
        f"{ood_correct / ood_total:.3f}"
    )