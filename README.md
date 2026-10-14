
# RAG: Dynamic Parsing and Dynamic Chunking

## Overview

This project implements a small Retrieval-Augmented Generation (RAG)
ingestion pipeline. It extracts PDF content, compares chunking
strategies, generates embeddings, and stores vectors in ChromaDB.

## Features

- PDF text extraction with page metadata
- Fixed-size chunking
- Recursive chunking
- Dynamic chunking
- Sentence Transformer embeddings
- Persistent ChromaDB vector storage
- Retrieval evaluation using test questions

## Project Structure

```text
rag-dynamic-chunking/
├── data/
│   └── sample_policy.pdf
├── src/
│   ├── parser.py
│   ├── chunkers.py
│   ├── indexer.py
│   └── evaluate.py
├── main.py
├── requirements.txt
└── README.md
```

## Technologies

- Python
- PyMuPDF
- LangChain Text Splitters
- Sentence Transformers
- ChromaDB
- pandas

## Setup

Create and activate a virtual environment.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Ingestion Pipeline

```bash
python main.py
```

## Evaluate Chunking Strategies

```bash
python src/evaluate.py
```

The evaluation compares fixed-size, recursive, and dynamic
chunking using the same test questions and a common embedding model.

## Current Prototype Results

The initial sample document produced two chunks for each
strategy. The current evaluation reported a 100% evidence
rate for all three strategies on the two configured test questions.

These results are preliminary and apply only to the current
sample document and evaluation questions.

## Limitations

- The sample corpus is very small.
- Dynamic chunking currently uses recursive splitting with
  page-level metadata rather than a complete structure-aware parser.
- OCR and advanced table extraction are not implemented.
- The evidence rate is a simple phrase-based metric.
- This project implements ingestion and retrieval evaluation;
  it does not yet provide a complete question-answering interface.

## Future Improvements

- Test multiple larger documents.
- Improve structure-aware chunking.
- Add more test questions and retrieval metrics.
- Evaluate scanned PDFs and tables.
- Add a user interface for document question answering.


