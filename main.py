
from pathlib import Path

from src.parser import parse_pdf
from src.chunkers import dynamic_chunking
from src.indexer import create_vector_store


def main():
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / "data" / "sample_policy.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Sample PDF not found: {pdf_path}"
        )

    print("Step 1: Parsing PDF...")
    pages = parse_pdf(str(pdf_path))
    print(f"Extracted {len(pages)} pages.")

    print("\nStep 2: Creating dynamic chunks...")
    chunks = dynamic_chunking(pages)
    print(f"Created {len(chunks)} chunks.")

    print("\nStep 3: Generating embeddings and indexing...")
    collection = create_vector_store(
        chunks,
        collection_name="rag_main_pipeline",
    )

    print(f"Indexed records: {collection.count()}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
