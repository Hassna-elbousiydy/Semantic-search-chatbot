from pathlib import Path

from src.preprocessing import load_jsonl
from src.retrieval import SemanticRetriever
from src.reranker import CrossEncoderReranker
from src.evaluation import (
    load_jsonl_records,
    item_is_relevant,
    first_relevant_rank,
)


CORPUS_PATH = Path(
    "data/scientific/processed/"
    "scientific_chunks.jsonl"
)

EVAL_PATH = Path(
    "data/evaluation/"
    "scientific_eval.jsonl"
)

FAISS_RELEVANCE_THRESHOLD = 0.30
RERANKER_CONFIDENCE_THRESHOLD = 0.0

# Selected experimentally from candidate-pool
# ablation: 5 / 10 / 20 / 30.
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


print("\nBuilding retriever...")

retriever = SemanticRetriever(
    corpus
)


print("\nBuilding reranker...")

reranker = CrossEncoderReranker()


answerable_total = 0

hit_at_1 = 0
hit_at_3 = 0
hit_at_5 = 0

mrr_at_5_sum = 0.0

hybrid_correct = 0

ood_total = 0
ood_correct = 0


for example in evaluation:

    question = example[
        "question"
    ]

    question_type = example[
        "type"
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{example['id']} | "
        f"{question_type}"
    )

    print(question)

    print(
        "=" * 100
    )

    candidates = retriever.search(
        question,
        k=CANDIDATE_POOL,
    )

    if not candidates:

        print(
            "No candidates returned."
        )

        continue

    faiss_top_score = (
        candidates[0]["score"]
    )

    print(
        f"FAISS top score: "
        f"{faiss_top_score:.4f}"
    )


    # -------------------------------------------------------------
    # Out-of-domain evaluation
    # -------------------------------------------------------------

    if (
        question_type
        == "out_of_domain"
    ):

        ood_total += 1

        refused = (
            faiss_top_score
            < FAISS_RELEVANCE_THRESHOLD
        )

        if refused:
            ood_correct += 1

        print(
            "OOD decision:",
            (
                "CORRECT REFUSAL"
                if refused
                else "FALSE ACCEPT"
            ),
        )

        continue


    # -------------------------------------------------------------
    # Retrieval evaluation
    # -------------------------------------------------------------

    answerable_total += 1

    relevant_rank = (
        first_relevant_rank(
            candidates,
            example,
        )
    )

    print(
        "First relevant FAISS rank:",
        relevant_rank,
    )

    if relevant_rank is not None:

        if relevant_rank <= 1:
            hit_at_1 += 1

        if relevant_rank <= 3:
            hit_at_3 += 1

        if relevant_rank <= 5:
            hit_at_5 += 1

            mrr_at_5_sum += (
                1.0
                / relevant_rank
            )


    # -------------------------------------------------------------
    # Hybrid reranking evaluation
    # -------------------------------------------------------------

    reranked = reranker.rerank(
        question,
        candidates,
        top_k=len(candidates),
    )

    if reranked:

        reranker_top = (
            reranked[0]
        )

        if (
            reranker_top[
                "rerank_score"
            ]
            > RERANKER_CONFIDENCE_THRESHOLD
        ):

            selected = (
                reranker_top
            )

            method = (
                "cross_encoder"
            )

        else:

            selected = (
                candidates[0]
            )

            method = (
                "faiss_fallback"
            )

        correct_selected = (
            item_is_relevant(
                selected,
                example,
            )
        )

        if correct_selected:
            hybrid_correct += 1

        print(
            "Hybrid selection:",
            method,
        )

        print(
            "Selected relevant chunk:",
            correct_selected,
        )


# -----------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------

print(
    "\n"
    + "=" * 100
)

print(
    "SCIENTIFIC RETRIEVAL SUMMARY"
)

print(
    "=" * 100
)


if answerable_total:

    print(
        f"Answerable questions: "
        f"{answerable_total}"
    )

    print(
        f"Hit@1: "
        f"{hit_at_1 / answerable_total:.3f}"
    )

    print(
        f"Hit@3: "
        f"{hit_at_3 / answerable_total:.3f}"
    )

    print(
        f"Hit@5: "
        f"{hit_at_5 / answerable_total:.3f}"
    )

    print(
        f"MRR@5: "
        f"{mrr_at_5_sum / answerable_total:.3f}"
    )

    print(
        "Hybrid selection accuracy: "
        f"{hybrid_correct / answerable_total:.3f}"
    )


if ood_total:

    print(
        "OOD refusal accuracy: "
        f"{ood_correct / ood_total:.3f}"
    )