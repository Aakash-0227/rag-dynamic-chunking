
from pathlib import Path
import time
import json

import ollama

from parser import parse_pdf
from chunkers import dynamic_chunking
from indexer import create_vector_store, embedding_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "sample_policy.pdf"

MODEL_NAME = "llama3.2"
TOP_K = 4

# These expected phrases are based on the internship-document
# details shown in your application. Verify them against your PDF.
TEST_CASES = [
    {
        "question": "What is the internship start date?",
        "expected": ["21st September, 2026"],
    },
    {
        "question": "What is the duration of the internship?",
        "expected": ["2 Months", "21st November, 2026"],
    },
    {
        "question": "What is the minimum attendance requirement?",
        "expected": ["90%"],
    },
    {
        "question": "What should the intern follow?",
        "expected": [
            "code of conduct",
            "policies and procedures",
        ],
    },
    {
        "question": "What happens if attendance is below 90%?",
        "expected": [
            "termination of the internship",
        ],
    },
]


def build_index():
    """Parse the PDF, create dynamic chunks, and index them."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place your test PDF at data/sample_policy.pdf."
        )

    pages = parse_pdf(str(PDF_PATH))

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    for page in pages:
        page.setdefault("metadata", {})["source"] = PDF_PATH.name

    chunks = dynamic_chunking(pages)

    if not chunks:
        raise ValueError("Dynamic chunking returned no chunks.")

    collection = create_vector_store(
        chunks,
        collection_name="rag_evaluation",
    )

    print(f"Document: {PDF_PATH.name}")
    print(f"Pages: {len(pages)}")
    print(f"Dynamic chunks: {len(chunks)}")
    print(f"Indexed records: {collection.count()}")

    return collection


def retrieve(collection, question):
    """Retrieve relevant chunks and measure search latency."""

    count = collection.count()

    if count == 0:
        return [], 0.0

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    start = time.perf_counter()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )

    elapsed = time.perf_counter() - start

    evidence = []

    for i, document in enumerate(result["documents"][0]):
        evidence.append({
            "text": document,
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i],
        })

    return evidence, elapsed


def check_answer_faithfulness(question, answer, evidence):
    """
    Ask the local LLM whether the answer is supported by the evidence.

    This is an automated estimate, not a guaranteed correctness score.
    """

    context = "\n\n".join(
        item["text"] for item in evidence
    )

    if not context.strip():
        return "NOT_VERIFIABLE"

    prompt = f"""
Evaluate whether the answer is supported by the source evidence.

Question:
{question}

Source evidence:
{context}

Answer:
{answer}

Return only valid JSON in this format:
{{
  "verdict": "SUPPORTED",
  "reason": "Short explanation"
}}

Use exactly one verdict:
SUPPORTED - all factual claims are supported by the evidence.
PARTIALLY_SUPPORTED - some claims are supported but others are not.
UNSUPPORTED - the answer contradicts or lacks support from the evidence.
NOT_VERIFIABLE - the evidence is insufficient to judge.

Do not use outside knowledge.
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
            format="json",
        )

        result = json.loads(response["message"]["content"])
        verdict = result.get("verdict", "NOT_VERIFIABLE").upper()

        allowed = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "UNSUPPORTED",
            "NOT_VERIFIABLE",
        }

        if verdict not in allowed:
            return "NOT_VERIFIABLE"

        return verdict

    except Exception as exc:
        print(f"Faithfulness check failed: {exc}")
        return "NOT_VERIFIABLE"


def generate_answer(question, evidence):
    """Generate an answer using only retrieved evidence."""

    if not evidence:
        return "The retrieved evidence does not contain an answer."

    context = "\n\n".join(
        f"Page {item['metadata'].get('page', 'Unknown')}: "
        f"{item['text']}"
        for item in evidence
    )

    prompt = f"""
Answer the question using only the evidence below.

If the evidence is insufficient, say so.
Do not invent dates, conditions, or consequences.

Evidence:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    return response["message"]["content"]


def main():
    print("=" * 75)
    print("RAG EVALUATION — DYNAMIC CHUNKING")
    print("=" * 75)

    collection = build_index()

    total_expected = 0
    total_found = 0
    retrieval_times = []
    faithfulness_results = []

    for number, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected = test["expected"]

        evidence, elapsed = retrieve(collection, question)
        retrieval_times.append(elapsed)

        retrieved_text = " ".join(
            item["text"] for item in evidence
        ).casefold()

        found_phrases = [
            phrase
            for phrase in expected
            if phrase.casefold() in retrieved_text
        ]

        total_found += len(found_phrases)
        total_expected += len(expected)

        print("\n" + "-" * 75)
        print(f"Test {number}: {question}")
        print(f"Retrieved chunks: {len(evidence)}")
        print(f"Retrieval time: {elapsed:.4f} seconds")
        print(
            f"Expected phrases found: "
            f"{len(found_phrases)}/{len(expected)}"
        )

        for phrase in expected:
            status = (
                "FOUND"
                if phrase.casefold() in retrieved_text
                else "MISSING"
            )
            print(f"  [{status}] {phrase}")

        for rank, item in enumerate(evidence, start=1):
            metadata = item["metadata"]

            print(
                f"\n  Evidence {rank}: "
                f"page={metadata.get('page', 'Unknown')}, "
                f"section={metadata.get('section_path', 'Unknown')}"
            )
            print(f"  {item['text'][:350]}")

        answer = generate_answer(question, evidence)
        verdict = check_answer_faithfulness(
            question,
            answer,
            evidence,
        )
        faithfulness_results.append(verdict)

        print(f"\nGenerated answer: {answer}")
        print(f"Faithfulness estimate: {verdict}")

    evidence_rate = (
        total_found / total_expected
        if total_expected else 0.0
    )

    average_time = (
        sum(retrieval_times) / len(retrieval_times)
        if retrieval_times else 0.0
    )

    print("\n" + "=" * 75)
    print("FINAL EVALUATION REPORT")
    print("=" * 75)
    print(f"Test questions: {len(TEST_CASES)}")
    print(f"Expected phrases found: {total_found}/{total_expected}")
    print(f"Evidence phrase coverage: {evidence_rate:.1%}")
    print(f"Average retrieval time: {average_time:.4f} seconds")

    print("\nFaithfulness verdict counts:")
    for verdict in [
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "NOT_VERIFIABLE",
    ]:
        print(
            f"  {verdict}: "
            f"{faithfulness_results.count(verdict)}"
        )

    print(
        "\nNote: phrase coverage measures exact phrase matches "
        "in retrieved text. Faithfulness verdicts are LLM estimates, "
        "not proof of correctness."
    )


if __name__ == "__main__":
    main()
