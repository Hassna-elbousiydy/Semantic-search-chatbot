import json
import re
from pathlib import Path

from ftfy import fix_text
from transformers import AutoTokenizer


TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOKENS = 240
STRIDE = 60


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL
)

# We use the tokenizer only to count/split tokens.
# Whole scientific pages can legitimately exceed the model input length.
tokenizer.model_max_length = 1_000_000


def clean_scientific_text(text: str) -> str:
    """
    Clean scientific text extracted from PDF files.
    """

    # Repair common Unicode / encoding corruption
    text = fix_text(text)

    # Normalize common PDF whitespace
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")

    # Reconnect words broken by line-end hyphenation
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Replace remaining line breaks with spaces
    text = re.sub(
        r"\s*\n\s*",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def chunk_text(
    text: str,
    max_tokens: int = MAX_TOKENS,
    stride: int = STRIDE
):
    """
    Split text into overlapping token-aware chunks.
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    start = 0

    while start < len(token_ids):

        end = min(
            start + max_tokens,
            len(token_ids)
        )

        chunk_ids = token_ids[start:end]

        decoded_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        chunks.append({
            "text": decoded_text.strip(),
            "n_tokens": len(chunk_ids)
        })

        if end == len(token_ids):
            break

        start = max(
            0,
            end - stride
        )

    return chunks


def process_extracted_jsonl(
    input_file: Path
):
    """
    Convert page-level JSONL extraction into
    cleaned scientific chunks with source metadata.
    """

    records = []

    with input_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            page_record = json.loads(line)

            cleaned_text = clean_scientific_text(
                page_record["text"]
            )

            if not cleaned_text:
                continue

            chunks = chunk_text(
                cleaned_text
            )

            for chunk_id, chunk in enumerate(
                chunks
            ):

                records.append({
                    "source_file":
                        page_record["source_file"],

                    "page":
                        page_record["page"],

                    "chunk_id":
                        chunk_id,

                    "text":
                        chunk["text"],

                    "n_tokens":
                        chunk["n_tokens"]
                })

    return records