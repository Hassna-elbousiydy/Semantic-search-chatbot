from pathlib import Path
import json

import pymupdf


def extract_pdf_pages(pdf_path: Path):
    """
    Extract text page by page from a PDF.

    Returns a list of dictionaries containing:
    - source_file
    - page
    - text
    """

    document = pymupdf.open(pdf_path)

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text(
                "text",
                sort=True
            ).strip()

            if not text:
                continue

            pages.append({
                "source_file": pdf_path.name,
                "page": page_number,
                "text": text
            })

    finally:
        document.close()

    return pages


def extract_pdf_directory(pdf_dir: Path, output_dir: Path):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        pdf_dir.glob("*.pdf")
    )

    print(f"Found {len(pdf_files)} PDF files.")

    summary = []

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        pages = extract_pdf_pages(
            pdf_path
        )

        output_path = (
            output_dir /
            f"{pdf_path.stem}.jsonl"
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            for page in pages:

                file.write(
                    json.dumps(
                        page,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        print(
            f"  Extracted pages: {len(pages)}"
        )

        print(
            f"  Saved to: {output_path}"
        )

        summary.append({
            "source_file": pdf_path.name,
            "pages_extracted": len(pages),
            "output_file": output_path.name
        })

    return summary