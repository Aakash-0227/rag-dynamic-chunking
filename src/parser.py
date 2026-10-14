
from pathlib import Path
import pymupdf


def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF page by page.
    Preserve page numbers and source information.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    parsed_pages = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                print(f"Warning: Page {page_number} contains no extractable text.")
                continue

            parsed_pages.append({
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                    "total_pages": len(pdf)
                }
            })

    return parsed_pages


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "sample_policy.pdf"

    pages = parse_pdf(str(pdf_path))

    print(f"Extracted {len(pages)} pages.")

    for page in pages[:2]:
        print(f"\nPage: {page['metadata']['page']}")
        print(page["text"][:500])
