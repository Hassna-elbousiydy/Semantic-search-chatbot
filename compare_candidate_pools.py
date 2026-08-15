import json
from pathlib import Path

from src.preprocessing import load_jsonl
from src.retrieval import SemanticRetriever
from src.reranker import CrossEncoderReranker


CORPUS_PATH = Path(
    "data/scientific/processed/scientific_chunks.jsonl"
)

EVAL_PATH = Path(
    "data/evaluation/scientific_eval.jsonl"
)

FAISS_RELEVANCE_THRESHOLD = 0.30
RERANKER_CONFIDENCE_THRESHOLD = 0.0

CANDIDATE_POOLS = [5, 10, 20, 30]
MAX_POOL = max(CANDIDATE_POOLS)


def load_eval(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def item_is_relevant(item, example):

    if (
        item.get("source_file")
        != example.get("expected_source_file")
    ):
        return False

    for relevant in example["relevant_chunks"]:

        if (
            item.get("page") == relevant["page"]
            and item.get("chunk_id")
            == relevant["chunk_id"]
        ):
            return True

    return False


corpus = load_jsonl(CORPUS_PATH)
evaluation = load_eval(EVAL_PATH)

print(f"Loaded {len(corpus)} corpus chunks.")
print(
    f"Loaded {len(evaluation)} evaluation questions."
)

print("\nBuilding retriever...")
retriever = SemanticRetriever(corpus)

print("\nBuilding reranker...")
reranker = CrossEncoderReranker()


# Retrieve the maximum candidate pool only once per question.
retrieval_cache = {}

for example in evaluation:

    retrieval_cache[example["id"]] = retriever.search(
        example["question"],
        k=MAX_POOL,
    )


print("\n" + "=" * 110)
print("CANDIDATE POOL COMPARISON")
print("=" * 110)


for pool_size in CANDIDATE_POOLS:

    answerable_total = 0

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_10 = 0
    hit_at_pool = 0

    reciprocal_rank_10_sum = 0.0
    reciprocal_rank_pool_sum = 0.0

    hybrid_correct = 0

    ood_total = 0
    ood_correct = 0

    for example in evaluation:

        question_type = example["type"]

        all_candidates = retrieval_cache[
            example["id"]
        ]

        candidates = all_candidates[:pool_size]

        if not candidates:
            continue

        faiss_top_score = candidates[0]["score"]

        if question_type == "out_of_domain":

            ood_total += 1

            refused = (
                faiss_top_score
                < FAISS_RELEVANCE_THRESHOLD
            )

            if refused:
                ood_correct += 1

            continue

        answerable_total += 1

        relevant_rank = None

        for rank, item in enumerate(
            candidates,
            start=1,
        ):

            if item_is_relevant(
                item,
                example,
            ):
                relevant_rank = rank
                break

        if relevant_rank is not None:

            if relevant_rank <= 1:
                hit_at_1 += 1

            if relevant_rank <= 3:
                hit_at_3 += 1

            if relevant_rank <= 10:
                hit_at_10 += 1

                reciprocal_rank_10_sum += (
                    1.0 / relevant_rank
                )

            if relevant_rank <= pool_size:
                hit_at_pool += 1

                reciprocal_rank_pool_sum += (
                    1.0 / relevant_rank
                )

        reranked = reranker.rerank(
            example["question"],
            candidates,
            top_k=len(candidates),
        )

        if reranked:

            reranker_top = reranked[0]

            if (
                reranker_top["rerank_score"]
                > RERANKER_CONFIDENCE_THRESHOLD
            ):
                selected = reranker_top

            else:
                selected = candidates[0]

            if item_is_relevant(
                selected,
                example,
            ):
                hybrid_correct += 1

    print("\n" + "-" * 110)
    print(
        f"CANDIDATE_POOL = {pool_size}"
    )
    print("-" * 110)

    if answerable_total:

        print(
            f"Hit@1: "
            f"{hit_at_1 / answerable_total:.3f}"
        )

        print(
            f"Hit@3: "
            f"{hit_at_3 / answerable_total:.3f}"
        )

        print(
            f"Hit@10: "
            f"{hit_at_10 / answerable_total:.3f}"
        )

        print(
            f"Hit@{pool_size}: "
            f"{hit_at_pool / answerable_total:.3f}"
        )

        print(
            f"MRR@10: "
            f"{reciprocal_rank_10_sum / answerable_total:.3f}"
        )

        print(
            f"MRR@{pool_size}: "
            f"{reciprocal_rank_pool_sum / answerable_total:.3f}"
        )

        print(
            f"Hybrid selection accuracy: "
            f"{hybrid_correct / answerable_total:.3f}"
        )

    if ood_total:

        print(
            f"OOD refusal accuracy: "
            f"{ood_correct / ood_total:.3f}"
        )