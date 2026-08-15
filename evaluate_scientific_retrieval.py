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
CANDIDATE_POOL = 10


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


answerable_total = 0

hit_at_1 = 0
hit_at_3 = 0
hit_at_10 = 0

reciprocal_rank_sum = 0.0

hybrid_correct = 0

ood_total = 0
ood_correct = 0


for example in evaluation:

    question = example["question"]
    question_type = example["type"]

    print("\n" + "=" * 100)
    print(
        f"{example['id']} | {question_type}"
    )
    print(question)
    print("=" * 100)

    candidates = retriever.search(
        question,
        k=CANDIDATE_POOL,
    )

    if not candidates:
        print("No candidates returned.")
        continue

    faiss_top_score = candidates[0]["score"]

    print(
        f"FAISS top score: "
        f"{faiss_top_score:.4f}"
    )

    if question_type == "out_of_domain":

        ood_total += 1

        refused = (
            faiss_top_score
            < FAISS_RELEVANCE_THRESHOLD
        )

        if refused:
            ood_correct += 1

        print(
            "OOD decision:",
            "CORRECT REFUSAL"
            if refused
            else "FALSE ACCEPT",
        )

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

    print(
        "First relevant FAISS rank:",
        relevant_rank,
    )

    if relevant_rank is not None:

        if relevant_rank <= 1:
            hit_at_1 += 1

        if relevant_rank <= 3:
            hit_at_3 += 1

        if relevant_rank <= 10:
            hit_at_10 += 1

        reciprocal_rank_sum += (
            1.0 / relevant_rank
        )

    reranked = reranker.rerank(
        question,
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
            method = "cross_encoder"

        else:
            selected = candidates[0]
            method = "faiss_fallback"

        correct_selected = item_is_relevant(
            selected,
            example,
        )

        if correct_selected:
            hybrid_correct += 1

        print(
            f"Hybrid selection: {method}"
        )

        print(
            "Selected relevant chunk:",
            correct_selected,
        )


print("\n" + "=" * 100)
print("RETRIEVAL SUMMARY")
print("=" * 100)

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
        f"Hit@10: "
        f"{hit_at_10 / answerable_total:.3f}"
    )

    print(
        f"MRR@10: "
        f"{reciprocal_rank_sum / answerable_total:.3f}"
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