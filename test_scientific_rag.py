from pathlib import Path

from src.preprocessing import load_jsonl
from src.rag import RAGSystem


DATA_PATH = Path(
    "data/scientific/processed/scientific_chunks.jsonl"
)


records = load_jsonl(DATA_PATH)

print(
    f"Loaded {len(records)} scientific chunks."
)


rag = RAGSystem(
    records,
    relevance_threshold=0.30,
    use_reranker=True,
    reranker_confidence_threshold=0.0,
    candidate_pool=5,
)


while True:

    question = input(
        "\nAsk a battery research question "
        "(or type quit): "
    )

    if question.lower().strip() == "quit":
        break

    result = rag.answer(
        question,
        k=3,
    )

    print("\nANSWER")
    print("=" * 80)

    print(result["answer"])

    print(
        "\nSelection method:",
        result["selection_method"],
    )

    print("\nSOURCES")
    print("=" * 80)

    for source in result["sources"]:

        print(
            f"\nSource {source['rank']}"
        )

        retrieval_score = source.get(
            "retrieval_score"
        )

        if retrieval_score is not None:
            print(
                f"FAISS score: "
                f"{retrieval_score:.4f}"
            )

        rerank_score = source.get(
            "rerank_score"
        )

        if rerank_score is not None:
            print(
                f"Reranker score: "
                f"{rerank_score:.4f}"
            )

        if source.get("source_file"):
            print(
                f"Article: "
                f"{source['source_file']}"
            )

        if source.get("page") is not None:
            print(
                f"Page: {source['page']}"
            )

        print(
            f"Chunk: {source['chunk_id']}"
        )

        print(
            source["text"][:500]
        )

        print("-" * 80)