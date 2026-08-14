from pathlib import Path

from src.preprocessing import load_jsonl
from src.rag import RAGSystem


DATA_PATH = Path(
    "data/sample/squad_chunks_sample.jsonl"
)


records = load_jsonl(DATA_PATH)

print(f"Loaded {len(records)} chunks.")

rag = RAGSystem(records)


while True:

    question = input(
        "\nAsk a question (or type quit): "
    )

    if question.lower() == "quit":
        break

    result = rag.answer(
        question,
        k=3
    )

    print("\nANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\nSOURCES")
    print("=" * 80)

    for source in result["sources"]:

        print(
            f"\nSource {source['rank']} "
            f"| score={source['score']:.4f} "
            f"| document={source['doc_uuid']} "
            f"| chunk={source['chunk_id']}"
        )

        print(
            source["text"][:400]
        )

        print("-" * 80)