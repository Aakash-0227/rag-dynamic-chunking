
import re
from pathlib import Path

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


def fixed_size_chunking(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 30,
) -> list[str]:
    """Baseline: split text using a fixed character limit."""

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    return splitter.split_text(text)


def recursive_chunking(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 30,
) -> list[str]:
    """Baseline: split using progressively smaller boundaries."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)


def split_heading(line: str):
    """Identify simple headings and heading-prefixed paragraphs."""

    line = line.strip()

    if not line:
        return None, None

    # A short uppercase line is likely a document heading.
    if (
        len(line) <= 100
        and any(char.isalpha() for char in line)
        and line == line.upper()
    ):
        return line, ""

    # Example: "Password Reset: Select Forgot Password."
    match = re.match(
        r"^([A-Za-z][A-Za-z0-9 /&()_-]{1,59}):\s*(.*)$",
        line,
    )

    if match:
        return match.group(1).strip(), match.group(2).strip()

    return None, None


def parse_sections(page_text: str) -> list[dict]:
    """Group text into heading-aware paragraph blocks."""

    sections = []
    current_heading = "General"
    paragraph_lines = []

    def save_paragraph():
        if paragraph_lines:
            text = " ".join(
                line.strip() for line in paragraph_lines
            ).strip()

            if text:
                sections.append({
                    "heading": current_heading,
                    "text": text,
                })

            paragraph_lines.clear()

    for line in page_text.splitlines():
        line = line.strip()

        if not line:
            save_paragraph()
            continue

        heading, remainder = split_heading(line)

        if heading is not None:
            save_paragraph()
            current_heading = heading

            if remainder:
                paragraph_lines.append(remainder)

        else:
            paragraph_lines.append(line)

    save_paragraph()

    return sections


def split_long_text(
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Split oversized content without exceeding the limit."""

    if len(text) <= max_chars:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=min(overlap, max_chars // 4),
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)


def dynamic_chunking(
    pages: list[dict],
    chunk_size: int = 400,
    chunk_overlap: int = 40,
) -> list[dict]:
    """
    Create heading-aware chunks while preserving source metadata.

    Note: chunk_size and chunk_overlap are measured in characters,
    not tokens.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    chunks = []

    for page in pages:
        page_metadata = page["metadata"]
        sections = parse_sections(page["text"])

        current_heading = None
        current_text = ""

        def save_chunk():
            nonlocal current_text

            if not current_text.strip():
                return

            chunks.append({
                "text": current_text.strip(),
                "metadata": {
                    **page_metadata,
                    "section_path": current_heading or "General",
                    "chunk_id": (
                        f"{page_metadata['source']}_"
                        f"page{page_metadata['page']}_"
                        f"chunk{len(chunks) + 1}"
                    ),
                    "chunking_strategy": "dynamic",
                },
            })

            current_text = ""

        for section in sections:
            heading = section["heading"]
            body = section["text"]

            # Keep the section heading attached to its content.
            section_text = f"{heading}\n{body}".strip()

            pieces = split_long_text(
                section_text,
                max_chars=chunk_size,
                overlap=chunk_overlap,
            )

            for piece in pieces:
                if (
                    current_text
                    and heading == current_heading
                    and len(current_text) + 2 + len(piece) <= chunk_size
                ):
                    current_text += "\n\n" + piece

                else:
                    save_chunk()
                    current_heading = heading
                    current_text = piece

                if len(current_text) >= chunk_size:
                    save_chunk()

        save_chunk()

    return chunks


if __name__ == "__main__":
    from parser import parse_pdf

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "sample_policy.pdf"

    pages = parse_pdf(str(pdf_path))
    text = "\n\n".join(page["text"] for page in pages)

    fixed_chunks = fixed_size_chunking(text)
    recursive_chunks = recursive_chunking(text)
    dynamic_chunks = dynamic_chunking(pages)

    print("Fixed-size chunks:", len(fixed_chunks))
    print("Recursive chunks:", len(recursive_chunks))
    print("Dynamic chunks:", len(dynamic_chunks))

    print("\nDynamic chunk details:")

    for chunk in dynamic_chunks:
        print("\nText:", chunk["text"])
        print("Metadata:", chunk["metadata"])
        print("Character count:", len(chunk["text"]))

