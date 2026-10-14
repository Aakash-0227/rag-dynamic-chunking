
# RAG: Dynamic Parsing and Dynamic Chunking

A Retrieval-Augmented Generation (RAG) application that answers questions from PDF documents using document parsing, multiple chunking strategies, semantic search, and a locally running Large Language Model (LLM).

## Overview

This project explores how different document chunking strategies affect retrieval quality in a PDF question-answering system. Users can upload a PDF, ask questions about its contents, and review the retrieved passages used to generate answers.

## Key Features

- **PDF parsing:** Extracts text and page-level information from PDF documents.
- **Multiple chunking strategies:** Supports fixed-size, recursive, and dynamic chunking.
- **Semantic search:** Generates embeddings and retrieves relevant passages using ChromaDB.
- **Local LLM integration:** Uses Ollama with the Llama 3.2 model to generate answers.
- **Interactive interface:** Streamlit UI for uploading documents and asking questions.
- **Source evidence:** Displays retrieved passages and associated metadata to help users verify answers.
- **Evaluation scripts:** Compares chunking strategies and evaluates RAG answer support.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| PDF parsing | PyMuPDF |
| Text splitting | LangChain Text Splitters |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| LLM runtime | Ollama |
| Language model | Llama 3.2 |
| User interface | Streamlit |

## Architecture

```text
PDF Document
     |
     v
PDF Parsing and Metadata Extraction
     |
     v
Text Chunking
(Fixed-Size / Recursive / Dynamic)
     |
     v
Embedding Generation
     |
     v
ChromaDB Vector Index
     |
     v
User Question
     |
     v
Semantic Retrieval
     |
     v
Retrieved Evidence + User Question
     |
     v
Ollama / Llama 3.2
     |
     v
Answer with Source Evidence
     |
     v
Streamlit Interface
```

## Project Structure

```text
rag-dynamic-chunking/
├── data/
│   ├── sample_policy.pdf
│   └── uploads/
├── src/
│   ├── parser.py
│   ├── chunkers.py
│   ├── indexer.py
│   ├── evaluate.py
│   └── evaluate_rag.py
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** The sample PDF and uploaded documents are excluded from the public repository where configured in `.gitignore`. The folder structure above describes the local development layout; ignored folders or files may not appear on GitHub.

## Installation and Setup

### Prerequisites

- Python installed
- Git installed
- Ollama installed
- The Llama 3.2 model available through Ollama

### 1. Clone the repository

```bash
git clone https://github.com/Aakash-0227/rag-dynamic-chunking.git
cd rag-dynamic-chunking
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the LLM

```bash
ollama pull llama3.2
```

Ensure the Ollama service is running before asking questions in the application.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal by Streamlit.

### 6. Run the command-line pipeline

```bash
python main.py
```

The command-line pipeline uses the PDF path configured in `main.py`. Since the sample PDF is not included in the public repository, provide an appropriate local PDF or update the configured path before running it.

### 7. Run evaluation scripts

Run these only if the corresponding files and required input data are available:

```bash
python src/evaluate.py
python src/evaluate_rag.py
```

## Chunking Strategies

- **Fixed-size chunking:** Splits text into chunks according to configured size and overlap.
- **Recursive chunking:** Splits text using a hierarchy of separators to preserve larger text units where possible.
- **Dynamic chunking:** Uses document structure and headings to create chunks that better reflect the content's organization.

The evaluation scripts can be used to compare chunk counts, retrieval evidence coverage, and answer support. Results depend on the document, chunking configuration, retrieval settings, and evaluation method.

## Example Questions

After uploading a suitable PDF, try questions such as:

- What is the main purpose of this document?
- What are the important rules or requirements?
- What dates or deadlines are mentioned?
- Summarize the key points of a specific section.

Answers depend on the contents of the selected document.

## Limitations

- Answer quality depends on PDF text extraction, chunk boundaries, embedding quality, and retrieval relevance.
- LLM-generated answers may be incomplete or incorrect and should be verified against the original document.
- The application should not be assumed to combine evidence across multiple uploaded documents unless cross-document retrieval has been implemented.
- Evaluation results are experimental and are not a guarantee of performance on every PDF.

## Future Enhancements

- Improve table and scanned-PDF extraction.
- Add OCR support for image-based PDFs.
- Expand evaluation with a larger question-and-answer dataset.
- Compare chunking strategies across different document types.
- Add cross-document retrieval and more detailed evaluation reports.

## Author

**Aakash**

GitHub: [Aakash-0227](https://github.com/Aakash-0227)

---

*This project demonstrates document processing, text chunking, semantic retrieval, vector databases, and local LLM integration in a RAG pipeline.*


