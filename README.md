# RAG Assistant

A local Retrieval-Augmented Generation (RAG) application for asking questions about text-based PDF documents. It includes a Streamlit chat application and a separate conversation-aware FastAPI backend.

## Features

- Add PDF files through the Streamlit chat input or place them in `documents/`.
- Extract PDF text page by page, retaining each source filename and 1-based page number.
- Retrieve passages with semantic and keyword search, with optional Cross-Encoder reranking.
- Generate answers with a local Hugging Face language model and display citations to retrieved evidence.
- Keep Streamlit conversation history in SQLite; the FastAPI backend keeps sessions in memory while the server is running.
- Apply deterministic grounding checks to answer sentences. Comparison answers require a retrieved sentence that names and explicitly contrasts both entities.

## Architecture and answer workflow

1. `main.py` reads PDFs from `documents/` using `pypdf`, cleans extracted text, and splits pages into overlapping text chunks. Page metadata carries the original PDF filename and 1-based page number.
2. `sentence-transformers/all-MiniLM-L6-v2` creates normalized embeddings. Search combines FAISS semantic ranking with BM25 keyword ranking using reciprocal-rank fusion.
3. The application can rerank the retrieved shortlist with `cross-encoder/ms-marco-MiniLM-L-6-v2`. If the reranker is unavailable, it continues with the hybrid ranking.
4. `Qwen/Qwen2.5-1.5B-Instruct`, loaded through Transformers and PyTorch, proposes an answer from the retrieved passages.
5. Deterministic checks accept only complete sentences found in the retrieved evidence and supported by a recognized question relation. Citations use the filename and page attached to the passage containing the accepted sentence.

The Streamlit UI is implemented in `app.py`. The FastAPI backend is implemented in `api.py`; it uses the same PDF, retrieval, and answer functions from `main.py`.

## Project files

```text
RAG-Assistant/
├── app.py                         # Streamlit chat application
├── api.py                         # FastAPI backend (/health and /ask)
├── main.py                        # PDF extraction, retrieval, reranking, and grounding
├── conversation_store.py          # SQLite conversation storage for Streamlit
├── streamlit_api_client.py        # Optional Streamlit client for the FastAPI backend
├── requirements.txt               # Core RAG and Streamlit dependencies
├── documents/                     # Local PDF library (ignored by Git)
├── tests/
│   └── test_grounding_and_history.py
└── README.md
```

## Setup

Run these commands from the project directory. Python 3.10 or newer is recommended.

### Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

Install the dependencies listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The FastAPI backend has additional dependencies that are not listed in `requirements.txt`. Install them if you plan to run the API:

```bash
python -m pip install fastapi uvicorn
```

### Add PDF files

Place readable, text-based PDFs in `documents/`, or upload PDFs through the Streamlit chat input. The Streamlit app stores uploaded files in `documents/`; this directory is excluded from Git. The API loads the PDFs from that directory when it starts, so restart the API after changing its PDF library.

## Run the applications

Start the Streamlit application:

```bash
streamlit run app.py
```

Start the FastAPI backend from the project directory:

```bash
python -m uvicorn api:app --reload
```

The API listens on `http://127.0.0.1:8000` by default. It provides `GET /health`, `POST /ask`, and FastAPI's interactive API documentation at `/docs`. The optional `streamlit_api_client.py` frontend connects to that backend; it additionally requires the `requests` package, which is not currently listed in `requirements.txt`.

## Run tests

`pytest` is not currently listed in `requirements.txt`; install it in the active virtual environment if needed:

```bash
python -m pip install pytest
```

Run the focused grounding and conversation-history tests:

```bash
python -m pytest tests/test_grounding_and_history.py -v
```

Run all discovered tests:

```bash
python -m pytest -q
```

## Limitations

- The application depends on text that `pypdf` can extract. Scanned pages without an extractable text layer are not OCR-processed.
- Model files may be downloaded from Hugging Face on first use; startup time, disk use, and response speed depend on available network access and hardware.
- Retrieval may not include a passage that supports a question. The assistant abstains when evidence is missing, when an exact answer sentence cannot be verified against retrieved text, or when the requested relation is unsupported. Comparison questions require an explicit contrast supported by a retrieved sentence.
- Deterministic grounding supports a limited set of question relations; it does not establish correctness beyond the retrieved PDF evidence.
- API conversation sessions are held in memory and are cleared when the API process restarts. Streamlit conversation history is stored locally in `.rag_cache/conversations.sqlite3`.
- Do not place secrets or private documents in source control. `documents/`, `.rag_cache/`, virtual environments, and model files are ignored by Git.

## Author

Jangilappola Ravi Kumar
