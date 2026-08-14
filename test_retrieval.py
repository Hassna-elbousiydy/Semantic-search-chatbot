from pathlib import Path

from src.preprocessing import load_jsonl
from src.retrieval import SemanticRetriever


DATA_PATH = Path(
    "data/sample/squad_chunks_sample.jsonl"
)


records = load_jsonl(DATA_PATH)

print(
    f"Loaded {len(records)} chunks."
)

retriever = SemanticRetriever(records)


while True:

    query = input(
        "\nAsk a question (or type quit): "
    )

    if query.lower() == "quit":
        break

    results = retriever.search(
        query,
        k=5
    )

    print("\nTOP RESULTS\n")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. Score: "
            f"{result['score']:.4f}"
        )

        print(
            result["text"][:500]
        )

        print("-" * 80)