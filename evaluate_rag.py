from pathlib import Path

from src.preprocessing import load_jsonl
from src.rag import RAGSystem


DATA_PATH = Path("data/sample/squad_chunks_sample.jsonl")

records = load_jsonl(DATA_PATH)

print(f"Loaded {len(records)} chunks.")

rag = RAGSystem(records)


TEST_CASES = [
    {
        "question": "Where is the University of Notre Dame located?",
        "expected_keywords": ["south bend", "indiana"],
        "should_refuse": False,
    },
    {
        "question": "What is the University of Notre Dame?",
        "expected_keywords": ["catholic", "research university"],
        "should_refuse": False,
    },
    {
        "question": "Who is the current president of France?",
        "expected_keywords": [],
        "should_refuse": True,
    },
]


passed = 0


for i, test in enumerate(TEST_CASES, start=1):

    print("\n" + "=" * 80)
    print(f"TEST {i}")
    print("=" * 80)

    result = rag.answer(
        test["question"],
        k=3
    )

    answer = result["answer"].lower()

    print("Question:", test["question"])
    print("Answer:", result["answer"])

    top_score = (
        result["sources"][0]["score"]
        if result["sources"]
        else 0.0
    )

    print(f"Top retrieval score: {top_score:.4f}")

    if test["should_refuse"]:

        success = (
            "i don't know" in answer
        )

    else:

        success = any(
            keyword in answer
            for keyword in test["expected_keywords"]
        )

    if success:
        print("RESULT: PASS")
        passed += 1

    else:
        print("RESULT: FAIL")


print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

total = len(TEST_CASES)

print(f"Passed: {passed}/{total}")
print(f"Success rate: {(passed / total) * 100:.1f}%")