from pathlib import Path
import json

from src.pdf_extraction import (
    extract_pdf_directory
)

from src.scientific_preprocessing import (
    process_extracted_jsonl
)


ROOT = Path(__file__).resolve().parent

PDF_DIR = (
    ROOT /
    "data" /
    "scientific" /
    "pdf"
)

TEXT_DIR = (
    ROOT /
    "data" /
    "scientific" /
    "text"
)

PROCESSED_DIR = (
    ROOT /
    "data" /
    "scientific" /
    "processed"
)


print("=" * 80)
print("STEP 1 - PDF EXTRACTION")
print("=" * 80)

summary = extract_pdf_directory(
    PDF_DIR,
    TEXT_DIR
)


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


all_chunks = []


print("\n" + "=" * 80)
print("STEP 2 - SCIENTIFIC CHUNKING")
print("=" * 80)


for jsonl_file in sorted(
    TEXT_DIR.glob("*.jsonl")
):

    print(
        f"\nProcessing extracted text: "
        f"{jsonl_file.name}"
    )

    records = process_extracted_jsonl(
        jsonl_file
    )

    print(
        f"  Chunks created: "
        f"{len(records)}"
    )

    all_chunks.extend(
        records
    )


output_file = (
    PROCESSED_DIR /
    "scientific_chunks.jsonl"
)


with output_file.open(
    "w",
    encoding="utf-8"
) as file:

    for record in all_chunks:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


print("\n" + "=" * 80)
print("FINAL CORPUS")
print("=" * 80)

print(
    f"PDF files: {len(summary)}"
)

print(
    f"Total chunks: {len(all_chunks)}"
)

print(
    f"Output: {output_file}"
)