import json
import re
from collections import Counter
from pathlib import Path


REFUSAL_ANSWER = (
    "I don't know based on the provided context."
)


def load_jsonl_records(path):
    """
    Load a JSONL file into a list of dictionaries.
    """

    records = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def normalize_text(text: str) -> str:
    """
    Normalize generated and reference answers
    before lexical evaluation.

    The normalization handles common scientific
    PDF variations such as:

        9.7%  vs  9. 7 %
        3.6%  vs  3. 6 %

        lumped-parameter
        lumped - parameter
    """

    text = str(text).lower()

    # Normalize mathematical minus sign.
    text = text.replace(
        "−",
        "-"
    )

    # Reconstruct decimals split by PDF extraction.
    text = re.sub(
        r"(\d)\s*\.\s*(\d)",
        r"\1.\2",
        text,
    )

    # Normalize spaces around percentages.
    text = re.sub(
        r"\s*%\s*",
        "%",
        text,
    )

    # Normalize hyphenated scientific terms.
    text = re.sub(
        r"(?<=\w)\s*[-‐-–—]\s*(?=\w)",
        " ",
        text,
    )

    # Remove remaining punctuation while
    # preserving useful numeric symbols.
    text = re.sub(
        r"[^a-z0-9.%+ ]",
        " ",
        text,
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def token_f1(
    prediction: str,
    reference: str,
) -> float:
    """
    Compute token-level F1 between a generated
    answer and its reference answer.
    """

    pred_tokens = normalize_text(
        prediction
    ).split()

    ref_tokens = normalize_text(
        reference
    ).split()

    if (
        not pred_tokens
        or not ref_tokens
    ):
        return 0.0

    pred_counter = Counter(
        pred_tokens
    )

    ref_counter = Counter(
        ref_tokens
    )

    overlap = sum(
        (
            pred_counter
            & ref_counter
        ).values()
    )

    if overlap == 0:
        return 0.0

    precision = (
        overlap
        / len(pred_tokens)
    )

    recall = (
        overlap
        / len(ref_tokens)
    )

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def item_is_relevant(
    item,
    example,
) -> bool:
    """
    Return True when a retrieved chunk matches
    one of the manually annotated relevant chunks
    for an evaluation question.
    """

    if (
        item.get("source_file")
        != example.get(
            "expected_source_file"
        )
    ):
        return False

    relevant_chunks = (
        example.get(
            "relevant_chunks",
            []
        )
    )

    return any(
        (
            item.get("page")
            == relevant["page"]
            and
            item.get("chunk_id")
            == relevant["chunk_id"]
        )
        for relevant
        in relevant_chunks
    )


def first_relevant_rank(
    candidates,
    example,
):
    """
    Return the rank of the first manually
    annotated relevant chunk.
    """

    for rank, item in enumerate(
        candidates,
        start=1,
    ):

        if item_is_relevant(
            item,
            example,
        ):
            return rank

    return None


def is_expected_refusal(
    answer: str,
) -> bool:
    """
    Check whether the generated answer matches
    the canonical out-of-domain refusal.
    """

    return (
        normalize_text(answer)
        == normalize_text(
            REFUSAL_ANSWER
        )
    )