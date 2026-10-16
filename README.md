# RAG Assistant

A Retrieval-Augmented Generation (RAG) application that lets users ask questions about PDF documents and receive answers grounded in retrieved document content.

## Features

- Upload and process PDF documents.
- Split document text into chunks for retrieval.
- Generate semantic embeddings using Sentence Transformers.
- Retrieve relevant content using FAISS and BM25.
- Generate answers using a local language model.
- Maintain conversation history.
- Display source evidence to help users verify answers.

## Tech Stack

- **Language:** Python
- **User interface:** Streamlit
- **PDF processing:** pypdf
- **Embeddings:** Sentence Transformers
- **Vector search:** FAISS
- **Keyword search:** BM25
- **Language models:** Hugging Face Transformers and PyTorch
- **Conversation storage:** SQLite

## Project Structure

```text
RAG-Assistant/
├── app.py
├── main.py
├── chunking.py
├── conversation_store.py
├── requirements.txt
├── tests/
├── documents/       # Local PDF documents; not included in Git
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add PDF documents

Place your PDF files in the local `documents/` directory. The folder is excluded from version control, so your private documents are not uploaded.

### 5. Run the application

```bash
streamlit run app.py
```

## Notes

- The first run may download model files and require additional disk space and time.
- Model availability and performance depend on your local hardware and configuration.
- Answers should be checked against the retrieved source evidence.
- Do not commit private documents, model weights, API keys, or other secrets.

## Testing

Run the project's tests with:

```bash
python -m pytest
```

## Author

Jangilappola Ravi Kumar

