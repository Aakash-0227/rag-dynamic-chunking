
from sentence_transformers import SentenceTransformer
import chromadb


# Load the embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(
    chunks: list[dict],
    collection_name: str = "rag_documents",
):
    """Create embeddings and store chunks in ChromaDB."""

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunks:
        print(f"No chunks to index for {collection_name}.")
        return collection

    documents = [chunk["text"] for chunk in chunks]

    ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            key: str(value)
            for key, value in chunk["metadata"].items()
        }
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(
        f"Indexed {len(documents)} chunks "
        f"into '{collection_name}'."
    )

    return collection


if __name__ == "__main__":
    from pathlib import Path
    from parser import parse_pdf
    from chunkers import dynamic_chunking

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "sample_policy.pdf"

    pages = parse_pdf(str(pdf_path))
    chunks = dynamic_chunking(pages)

    collection = create_vector_store(
        chunks,
        collection_name="rag_dynamic",
    )

    print("Total indexed records:", collection.count())

