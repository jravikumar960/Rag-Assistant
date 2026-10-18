"""Evaluate page-level retrieval quality before and after Cross-Encoder reranking.

Run from the RAG project directory:
    python evaluate_rag.py

This script reuses main.py's PDF, chunking, embedding, hybrid retrieval and
Cross-Encoder code. It does not load the answer-generation LLM.

IMPORTANT: The relevance labels below are a starter set based on pages observed
in the Random Forest PDF. Review the PDF and adjust these labels for your copy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import main as rag
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Keep this exact name in sync with the PDF in your documents/ folder.
RF_PDF = "811790898-Random-Forest-Algorithm.pdf"
K = 5
CANDIDATE_COUNT = max(20, getattr(rag, "RETRIEVAL_CANDIDATES", 20))

# Gold labels are PDF page numbers (1-based) that you have manually checked
# as supporting the question. Add or remove pages after inspecting the PDF.
EVALUATION_SET = [
    {
        "question": "Explain Random Forest.",
        "source": RF_PDF,
        "relevant_pages": {2, 9},
    },
    {
        "question": "How does Random Forest reduce overfitting?",
        "source": RF_PDF,
        "relevant_pages": {20, 36},
    },
    {
        "question": "What are the disadvantages of Random Forest?",
        "source": RF_PDF,
        "relevant_pages": {36, 37},
    },
]


def page_key(item: dict) -> tuple[str, int]:
    """Unique document page identity, avoiding multiple chunks inflating scores."""
    return str(item.get("source", "")), int(item.get("page", -1))


def top_unique_pages(results: list[dict], k: int) -> list[tuple[str, int]]:
    """Take the first k distinct pages in passage ranking order."""
    pages: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in results:
        key = page_key(item)
        if key not in seen:
            seen.add(key)
            pages.append(key)
            if len(pages) == k:
                break
    return pages


def page_metrics(
    results: list[dict], source: str, relevant_pages: Iterable[int], k: int = K
) -> dict[str, float | list[int]]:
    """Calculate page-level P@K and R@K, deduplicating chunks from same page.

    P@K uses K as its denominator. If fewer than K distinct pages are returned,
    missing positions count as non-relevant. Recall denominator is the manually
    labelled relevant pages for this source/query.
    """
    relevant = set(relevant_pages)
    selected_pages = top_unique_pages(results, k)
    selected_for_source = [page for src, page in selected_pages if src == source]
    hits = len(set(selected_for_source) & relevant)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "hits": hits,
        "retrieved_pages": selected_for_source,
    }


def print_metric_line(label: str, metrics: dict) -> None:
    print(
        f"  {label:<18} "
        f"Precision@{K}: {metrics['precision']:.1%}   "
        f"Recall@{K}: {metrics['recall']:.1%}   "
        f"pages={metrics['retrieved_pages']}"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    document_dir: Path = rag.DOCUMENTS_DIR
    pdf_files = sorted(document_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in: {document_dir}")
        print("Put the PDFs you use in your Streamlit app into the documents/ folder.")
        return 1

    if not any(path.name == RF_PDF for path in pdf_files):
        print(f"Warning: expected evaluation PDF not found: {RF_PDF}")
        print("Edit RF_PDF and the relevance labels in evaluate_rag.py to match your file.")
        print("Available PDFs:")
        for path in pdf_files:
            print(f"  - {path.name}")
        return 1

    print("Loading PDF text and building/reusing embeddings...")
    pages = rag.load_pdfs()
    chunks = rag.create_chunks(pages)
    if not chunks:
        print("No searchable chunks were created. Check your PDF text extraction.")
        return 1

    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL)
    embeddings, cache_hit = rag.load_or_create_embeddings(chunks, pdf_files, embedding_model)
    index = rag.build_faiss_index(embeddings)
    bm25 = BM25Okapi([rag._tokens(chunk["text"]) for chunk in chunks])
    reranker = rag.load_reranker()
    print(f"Loaded {len(pages)} text pages, {len(chunks)} chunks; embeddings cache hit={cache_hit}.")
    print(f"Evaluation uses page-level labels, deduplicates chunks by PDF page, and scores at K={K}.")

    baseline_p: list[float] = []
    baseline_r: list[float] = []
    reranked_p: list[float] = []
    reranked_r: list[float] = []

    for number, test in enumerate(EVALUATION_SET, start=1):
        question = test["question"]
        source = test["source"]
        relevant_pages = test["relevant_pages"]
        print("\n" + "=" * 78)
        print(f"Test {number}: {question}")
        print(f"Expected supporting pages for {source}: {sorted(relevant_pages)}")

        candidates = rag.search_chunks(
            question=question,
            model=embedding_model,
            index=index,
            embeddings=embeddings,
            chunks=chunks,
            bm25=bm25,
            top_k=CANDIDATE_COUNT,
            page_filter=None,
        )
        if not candidates:
            print("No candidates retrieved.")
            continue

        # Baseline: the top distinct pages from the hybrid RRF candidate ranking.
        baseline = page_metrics(candidates, source, relevant_pages, K)

        # Re-rank the candidate pool, then evaluate the top distinct pages.
        reranked_all = rag.rerank_chunks(
            question=question,
            candidates=candidates,
            reranker=reranker,
            top_k=len(candidates),
        )
        reranked = page_metrics(reranked_all, source, relevant_pages, K)

        print_metric_line("Hybrid baseline", baseline)
        print_metric_line("After reranking", reranked)

        baseline_p.append(float(baseline["precision"]))
        baseline_r.append(float(baseline["recall"]))
        reranked_p.append(float(reranked["precision"]))
        reranked_r.append(float(reranked["recall"]))

    if baseline_p and reranked_p:
        print("\n" + "=" * 78)
        print("MEAN OVER TEST QUESTIONS")
        print(f"  Hybrid baseline  Precision@{K}: {sum(baseline_p)/len(baseline_p):.1%}   Recall@{K}: {sum(baseline_r)/len(baseline_r):.1%}")
        print(f"  After reranking  Precision@{K}: {sum(reranked_p)/len(reranked_p):.1%}   Recall@{K}: {sum(reranked_r)/len(reranked_r):.1%}")
        print("\nInterpretation: higher is better, but scores depend on the relevance labels.")
        print("Manually verify the labels before treating these results as a benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
