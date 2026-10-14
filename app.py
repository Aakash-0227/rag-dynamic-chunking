from pathlib import Path
import hashlib
import sys

import streamlit as st
import ollama

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from parser import parse_pdf
from chunkers import dynamic_chunking
from indexer import create_vector_store, embedding_model


MODEL_NAME = "llama3.2"
TOP_K = 4


st.set_page_config(
    page_title="Internship Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Internship Document Assistant")
st.caption(
    "Upload PDF documents and ask questions with dynamic chunking, "
    "semantic retrieval, and evidence-based answers."
)


@st.cache_resource
def build_search_index(pdf_path, modified_time, file_hash):
    pages = parse_pdf(pdf_path)

    if not pages:
        raise ValueError("No readable text was extracted from the PDF.")

    chunks = dynamic_chunking(pages)

    if not chunks:
        raise ValueError("No chunks were created.")

    collection_name = f"rag_{file_hash[:20]}"

    collection = create_vector_store(
        chunks,
        collection_name=collection_name,
    )

    return collection, len(pages), len(chunks)


def clean_source_name(source):
    name = Path(str(source)).name

    if "_" in name:
        prefix, remainder = name.split("_", 1)

        if (
            len(prefix) >= 8
            and all(
                character in "0123456789abcdefABCDEF"
                for character in prefix
            )
        ):
            name = remainder

    return name


def retrieve_evidence(collection, question):
    count = collection.count()

    if count == 0:
        return []

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_K * 3, count),
        include=["documents", "metadatas", "distances"],
    )

    evidence = []
    seen = set()

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, document in enumerate(documents):
        metadata = metadatas[index] or {}

        page = metadata.get("page", "Unknown")
        normalized_text = " ".join(document.lower().split())

        duplicate_key = (normalized_text, str(page))

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)

        evidence.append({
            "text": document,
            "metadata": metadata,
            "distance": distances[index],
        })

        if len(evidence) >= TOP_K:
            break

    return evidence


def generate_answer(question, evidence):
    if not evidence:
        return (
            "I could not find relevant evidence in the selected "
            "document. Please try rephrasing your question."
        )

    context_parts = []

    for index, item in enumerate(evidence, start=1):
        metadata = item["metadata"]

        page = metadata.get("page", "Unknown")
        section = metadata.get("section_path", "Not specified")

        context_parts.append(
            f"[Source {index}]\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            f"Passage: {item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the source passages below.

Rules:
1. Never invent facts or use outside knowledge.
2. If the answer is missing, say it could not be verified
   from the retrieved evidence.
3. Preserve dates, amounts, numbers, and conditions accurately.
4. Give a concise, direct answer.
5. Cite supporting evidence using short labels such as [Source 1].
6. Only cite sources that support the claim.
7. Never reproduce the source metadata, filenames, page labels,
   section labels, or full source headers in your answer.
8. Do not print the context or prompt.
9. Do not invent source numbers.
10. Return only the answer with short source citations.

SOURCE PASSAGES:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={"temperature": 0},
    )

    answer = response["message"]["content"].strip()

    return answer


def show_sources(evidence):
    if not evidence:
        st.info("No supporting evidence was retrieved.")
        return

    for index, item in enumerate(evidence, start=1):
        metadata = item["metadata"]

        source = clean_source_name(
            metadata.get("source", "Uploaded document")
        )
        page = metadata.get("page", "Unknown")
        section = metadata.get("section_path", "Not specified")
        chunk_id = metadata.get("chunk_id", "Unknown")

        with st.expander(
            f"Source {index} · {source} · Page {page} · {section}"
        ):
            st.markdown("**Retrieved passage**")
            st.write(item["text"])

            st.caption(f"Document: {source}")
            st.caption(f"Page: {page}")
            st.caption(f"Section: {section}")
            st.caption(f"Chunk ID: {chunk_id}")
            st.caption(
                f"Retrieval distance: {item['distance']:.4f}"
            )


uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select the PDF documents you want to search.",
)

if not uploaded_files:
    st.info("Upload a PDF to begin asking questions.")
    st.stop()


documents = {}

upload_dir = PROJECT_ROOT / "data" / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)

for uploaded_file in uploaded_files:
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.warning(f"{uploaded_file.name} is empty.")
        continue

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    safe_name = Path(uploaded_file.name).name
    pdf_path = upload_dir / f"{file_hash[:12]}_{safe_name}"

    if (
        not pdf_path.exists()
        or hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        != file_hash
    ):
        pdf_path.write_bytes(file_bytes)

    documents[uploaded_file.name] = {
        "path": pdf_path,
        "hash": file_hash,
    }


if not documents:
    st.error("No valid PDF documents are available.")
    st.stop()


selected_name = st.selectbox(
    "Choose a document",
    options=list(documents.keys()),
)

selected_document = documents[selected_name]
pdf_path = selected_document["path"]
file_hash = selected_document["hash"]


try:
    with st.spinner("Preparing document and search index..."):
        collection, page_count, chunk_count = build_search_index(
            str(pdf_path),
            pdf_path.stat().st_mtime,
            file_hash,
        )

    col1, col2, col3 = st.columns(3)

    col1.metric("PDF pages", page_count)
    col2.metric("Dynamic chunks", chunk_count)
    col3.metric("Indexed records", collection.count())

    st.success(f"Document ready: {selected_name}")

except Exception as exc:
    st.error(f"Could not prepare the document: {exc}")
    st.info(
        "Check that the PDF is readable and your parser, chunker, "
        "embedding model, and ChromaDB are configured correctly."
    )
    st.stop()


st.divider()
st.subheader("Ask your document")

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder=(
            "What is the internship start date and "
            "minimum attendance requirement?"
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "Find answer",
        type="primary",
    )


if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner("Retrieving relevant evidence..."):
            evidence = retrieve_evidence(
                collection,
                question.strip(),
            )

        with st.spinner("Generating an evidence-based answer..."):
            answer = generate_answer(
                question.strip(),
                evidence,
            )

        st.divider()
        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Source citations")

        if evidence:
            st.caption(
                "Verify the cited passages against the answer "
                "and the original document."
            )
            show_sources(evidence)
        else:
            st.info(
                "No source passages were retrieved. "
                "The answer could not be verified."
            )

    except Exception as exc:
        st.error(f"Question answering failed: {exc}")
        st.info(
            "Check that Ollama is running, llama3.2 is installed, "
            "and your embedding model and ChromaDB are accessible."
        )


st.divider()

st.caption(
    "Answers are generated from retrieved document passages. "
    "Citations identify the retrieved evidence. Always verify "
    "important details against the original PDF."
)
