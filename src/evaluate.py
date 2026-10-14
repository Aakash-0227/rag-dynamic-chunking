
from pathlib import Path
import time

from parser import parse_pdf
from chunkers import (
    fixed_size_chunking,
    recursive_chunking,
    dynamic_chunking,
)
from indexer import create_vector_store, embedding_model


def prepare_chunks(pages, strategy):
    """Create chunks with consistent source metadata."""

    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        if strategy == "fixed_size":
            text_chunks = fixed_size_chunking(text)

        elif strategy == "recursive":
            text_chunks = recursive_chunking(text)

        elif strategy == "dynamic":
            continue

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for index, chunk_text in enumerate(text_chunks, start=1):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": (
                        f"{strategy}_{metadata['source']}_"
                        f"page{metadata['page']}_{index}"
                    ),
                    "chunking_strategy": strategy,
                },
            })

    if strategy == "dynamic":
        chunks = dynamic_chunking(pages)

        for index, chunk in enumerate(chunks, start=1):
            chunk["metadata"]["chunk_id"] = (
                f"dynamic_{chunk['metadata']['source']}_"
                f"page{chunk['metadata']['page']}_{index}"
            )
            chunk["metadata"]["chunking_strategy"] = "dynamic"

    return chunks


def retrieve(collection, question, top_k=3):
    """Retrieve relevant chunks using semantic similarity."""

    count = collection.count()

    if count == 0:
        return []

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for index, document in enumerate(results["documents"][0]):
        retrieved.append({
            "text": document,
            "metadata": results["metadatas"][0][index],
            "distance": results["distances"][0][index],
        })

    return retrieved


def main():
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "sample_policy.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Step 1: Parse the uploaded internship offer.
    pages = parse_pdf(str(pdf_path))

    if not pages:
        raise ValueError(
            "No text was extracted from the PDF. "
            "Check whether the PDF is scanned or empty."
        )

    print("=" * 75)
    print("RAG EVALUATION - COIREI INTERNSHIP OFFER")
    print("=" * 75)
    print(f"Document: {pdf_path.name}")
    print(f"Pages extracted: {len(pages)}")

    # Step 2: Questions based on the extracted document.
    # Expected phrases are evaluation targets, not generated answers.
    test_cases = [
        {
            "question": "What are the internship working hours?",
            "expected": ["10:00 AM", "7:00 PM"],
        },
        {
            "question": "What is the minimum attendance requirement?",
            "expected": ["90%", "attendance"],
        },
        {
            "question": "How many sick leave days are provided?",
            "expected": ["sick leave", "1 day"],
        },
        {
            "question": "How many casual leave days are provided?",
            "expected": ["casual leave", "1 day"],
        },
        {
            "question": "Which Saturdays are off?",
            "expected": ["2nd and 4th Saturdays"],
        },
        {
            "question": "What is the notice period for termination?",
            "expected": ["one week", "written notice"],
        },
        {
            "question": "What confidentiality obligations apply?",
            "expected": ["confidential", "information"],
        },
    ]

    strategies = ["fixed_size", "recursive", "dynamic"]
    comparison = []

    # Step 3: Compare each strategy.
    for strategy in strategies:
        print("\n" + "-" * 75)
        print(f"STRATEGY: {strategy.upper()}")
        print("-" * 75)

        chunks = prepare_chunks(pages, strategy)

        if not chunks:
            print("No chunks generated. Skipping strategy.")
            continue

        # Use a separate collection for each strategy.
        collection = create_vector_store(
            chunks,
            collection_name=f"rag_eval_{strategy}",
        )

        print(f"Chunks generated: {len(chunks)}")

        total_hits = 0
        total_expected = 0
        total_retrieval_time = 0.0
        question_count = 0

        # Step 4: Retrieve evidence for every question.
        for test in test_cases:
            question = test["question"]
            expected_phrases = test["expected"]

            start_time = time.perf_counter()

            results = retrieve(
                collection,
                question,
                top_k=3,
            )

            elapsed = time.perf_counter() - start_time
            total_retrieval_time += elapsed
            question_count += 1

            retrieved_text = " ".join(
                result["text"] for result in results
            ).lower()

            hits = sum(
                1
                for phrase in expected_phrases
                if phrase.lower() in retrieved_text
            )

            total_hits += hits
            total_expected += len(expected_phrases)

            print(f"\nQuestion: {question}")

            for rank, result in enumerate(results, start=1):
                metadata = result["metadata"]

                print(
                    f"\n  Rank {rank} | "
                    f"Page {metadata.get('page', 'Unknown')} | "
                    f"Distance {result['distance']:.4f}"
                )
                print(f"  Evidence: {result['text'][:500]}")

            print(
                f"Evidence phrases found: "
                f"{hits}/{len(expected_phrases)}"
            )

        evidence_rate = (
            total_hits / total_expected
            if total_expected
            else 0.0
        )

        comparison.append({
            "strategy": strategy,
            "chunks": len(chunks),
            "evidence_rate": evidence_rate,
            "retrieval_time": (
                total_retrieval_time / question_count
                if question_count
                else 0.0
            ),
        })

    # Step 5: Print the final comparison.
    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Strategy':<15}"
        f"{'Chunks':<10}"
        f"{'Evidence rate':<18}"
        f"{'Avg time (s)':<15}"
    )

    for result in comparison:
        print(
            f"{result['strategy']:<15}"
            f"{result['chunks']:<10}"
            f"{result['evidence_rate']:.1%}"
            f"{'':<13}"
            f"{result['retrieval_time']:.4f}"
        )

    print("\nEvaluation completed.")
    print(
        "Note: Evidence rate measures exact phrase matches "
        "in retrieved text, not answer correctness."
    )


if __name__ == "__main__":
    main()
