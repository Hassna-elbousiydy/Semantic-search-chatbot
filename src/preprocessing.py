import json
from pathlib import Path
from transformers import AutoTokenizer


TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 240
STRIDE = 60

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    return " ".join(text.split()).strip()


def chunk_by_tokens(
    text: str,
    max_tokens: int = MAX_TOKENS,
    stride: int = STRIDE
):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))

        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        chunks.append(chunk_text)

        if end == len(tokens):
            break

        start = max(0, end - stride)

    return chunks


def load_jsonl(path):
    records = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records