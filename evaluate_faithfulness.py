"""Generate a claim-level faithfulness review for the Random Forest RAG pipeline.

Run from the RAG project root (the folder containing main.py and documents/):
    python evaluate_faithfulness.py

Then manually label each factual claim in faithfulness_review.csv as:
    supported | unsupported | contradicted

After reviewing the CSV, calculate the score without re-running models:
    python evaluate_faithfulness.py --score-only

Important: this deliberately uses human review rather than treating a second LLM
as ground truth. The report includes the exact passages sent to answer generation.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Any

import main as rag
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

RF_PDF = "811790898-Random-Forest-Algorithm.pdf"
CANDIDATE_COUNT = max(20, int(getattr(rag, "RETRIEVAL_CANDIDATES", 20)))
FINAL_K = 5
OUTPUT_CSV = Path(rag.DOCUMENTS_DIR).parent / "faithfulness_review.csv"
VALID_LABELS = {"supported", "unsupported", "contradicted"}

# Questions span definition, method, use cases, benefits, limitations, and
# deliberate uncertainty tests. Whether each answer is supported must be checked
# against the exact evidence included in the CSV; don't assume every question
# is answerable from the PDF.
EVALUATION_SET = [
    ("RF01", "Explain Random Forest.", "Definition and high-level mechanism"),
    ("RF02", "How does Random Forest reduce overfitting?", "Bagging, multiple trees, variance/generalization"),
    ("RF03", "What is the bagging principle in Random Forest?", "Bagging or bootstrap samples"),
    ("RF04", "How does Random Forest make a classification prediction?", "Tree outputs and majority vote"),
    ("RF05", "How does Random Forest make a regression prediction?", "Prediction aggregation for regression"),
    ("RF06", "What types of machine-learning problems can Random Forest solve?", "Classification and regression"),
    ("RF07", "What is bootstrap sampling?", "Sampling with replacement, if explained by source"),
    ("RF08", "Why does Random Forest build multiple decision trees?", "Ensemble/diversity and combined predictions"),
    ("RF09", "What are the advantages of Random Forest?", "Advantages stated by the PDF"),
    ("RF10", "What are the disadvantages of Random Forest?", "Limitations stated by the PDF"),
    ("RF11", "Why can Random Forest take a long time to train?", "Training cost / many trees"),
    ("RF12", "Why can prediction with Random Forest be slow?", "Inference cost / runtime performance"),
    ("RF13", "How does Random Forest help produce a generalized model?", "Generalization and combining trees"),
    ("RF14", "How does Random Forest help with feature selection?", "Feature-selection description in PDF"),
    ("RF15", "Does Random Forest always need hyperparameter tuning?", "Only what the PDF says about tuning"),
    ("RF16", "What is the difference between a decision tree and Random Forest?", "Comparison supported by source"),
    ("RF17", "What role does majority voting play in Random Forest?", "Voting in classification"),
    ("RF18", "When might another approach be preferable to Random Forest?", "Runtime-critical cases if stated by source"),
    ("RF19", "What exact time complexity does the PDF give for Random Forest?", "Abstain if the source gives no exact complexity"),
    ("RF20", "Who invented Random Forest, and in what year?", "Abstain if the PDF does not provide this information"),
]


def extract_claims(answer: str) -> list[str]:
    """Split an answer into reviewable factual units (heuristic, not NLP)."""
    text = (answer or "").strip()
    # Keep source details separately in the evidence column, not as answer claims.
    text = re.split(r"\n\s*Sources?:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
    raw_parts = re.split(r"\n+|(?<=[.!?])\s+", text)
    claims: list[str] = []
    for part in raw_parts:
        part = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part).strip()
        part = re.sub(r"\*\*(.*?)\*\*", r"\1", part)
        part = part.strip(" \t-•")
        # Drop headings and empty fragments; retain meaningful bullet lines.
        if not part or len(part.split()) < 4 or part.endswith(":"):
            continue
        claims.append(part)
    # Stable deduplication of obvious duplicate fragments.
    seen: set[str] = set()
    unique: list[str] = []
    for claim in claims:
        key = claim.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(claim)
    return unique


def evidence_text(passages: list[dict[str, Any]]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(
            f"[{i}] {p.get('source', 'unknown source')} | page {p.get('page', '?')}\n"
            f"{str(p.get('text', '')).strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def write_review_row(writer, test_id, question, focus, answer, claim, evidence, sources, verdict="", notes=""):
    writer.writerow({
        "test_id": test_id,
        "question": question,
        "expected_focus": focus,
        "answer": answer,
        "claim_to_review": claim,
        "retrieved_evidence": evidence,
        "source_pages": sources,
        "manual_verdict": verdict,
        "reviewer_notes": notes,
    })


def score_existing_csv(path: Path) -> int:
    if not path.exists():
        print(f"Review file not found: {path}")
        print("Run without --score-only first to generate it.")
        return 1
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    reviewed = [r for r in rows if (r.get("claim_to_review") or "").strip()
                and (r.get("manual_verdict") or "").strip().lower() in VALID_LABELS]
    unreviewed_claims = [r for r in rows if (r.get("claim_to_review") or "").strip()
                          and (r.get("manual_verdict") or "").strip().lower() not in VALID_LABELS]
    if not reviewed:
        print("No reviewed claims found yet.")
        print("In faithfulness_review.csv, fill manual_verdict with supported, unsupported, or contradicted.")
        return 1
    supported = sum(r["manual_verdict"].strip().lower() == "supported" for r in reviewed)
    unsupported = sum(r["manual_verdict"].strip().lower() == "unsupported" for r in reviewed)
    contradicted = sum(r["manual_verdict"].strip().lower() == "contradicted" for r in reviewed)
    score = supported / len(reviewed)
    print(f"Reviewed factual claims: {len(reviewed)}")
    print(f"Supported:              {supported}")
    print(f"Unsupported:            {unsupported}")
    print(f"Contradicted:            {contradicted}")
    print(f"Unreviewed claims:       {len(unreviewed_claims)}")
    print(f"Manual faithfulness rate: {score:.1%} (supported / reviewed claims)")
    print("Interpretation: a manually reviewed claim is faithful only when the supplied evidence supports it.")
    print("This score is only as reliable as the review labels and claim splitting.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true", help="Score manually labeled rows in the existing CSV.")
    args = parser.parse_args()
    if args.score_only:
        return score_existing_csv(OUTPUT_CSV)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    document_dir: Path = rag.DOCUMENTS_DIR
    pdf_path = document_dir / RF_PDF
    if not pdf_path.exists():
        print(f"Expected PDF not found: {pdf_path}")
        print("Put the Random Forest PDF into documents/ or update RF_PDF in this script.")
        return 1

    print("Loading the Random Forest PDF and retrieval components...")
    pages = [p for p in rag.load_pdfs() if p.get("source") == RF_PDF]
    if not pages:
        print("No extractable text found in the selected PDF.")
        return 1
    chunks = rag.create_chunks(pages)
    if not chunks:
        print("No searchable chunks were created.")
        return 1

    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL)
    embeddings, cache_hit = rag.load_or_create_embeddings(chunks, [pdf_path], embedding_model)
    index = rag.build_faiss_index(embeddings)
    bm25 = BM25Okapi([rag._tokens(chunk["text"]) for chunk in chunks])
    reranker = rag.load_reranker()
    tokenizer, llm = rag.load_local_llm()
    print(f"PDF pages: {len(pages)}; chunks: {len(chunks)}; embedding cache hit: {cache_hit}")
    print(f"Questions: {len(EVALUATION_SET)}; candidates/query: {CANDIDATE_COUNT}; final passages: {FINAL_K}")
    print("Generating answers and exporting claim-level review rows. This may take a while on CPU.")

    fieldnames = [
        "test_id", "question", "expected_focus", "answer", "claim_to_review",
        "retrieved_evidence", "source_pages", "manual_verdict", "reviewer_notes",
    ]
    total_claims = 0
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (test_id, question, focus) in enumerate(EVALUATION_SET, start=1):
            print(f"[{idx:02d}/{len(EVALUATION_SET)}] {question}", flush=True)
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
            if candidates:
                passages = rag.rerank_chunks(
                    question=question,
                    candidates=candidates,
                    reranker=reranker,
                    top_k=FINAL_K,
                )
            else:
                passages = []
            answer = rag.generate_answer(question, passages, tokenizer, llm) if passages else (
                "I couldn't find relevant information in the PDF to answer this question."
            )
            evidence = evidence_text(passages)
            sources = "; ".join(dict.fromkeys(
                f"{p.get('source', '?')}, page {p.get('page', '?')}" for p in passages
            ))
            claims = extract_claims(answer)
            if claims:
                for claim in claims:
                    write_review_row(writer, test_id, question, focus, answer, claim, evidence, sources)
                    total_claims += 1
            else:
                # Keep an explicit row for abstentions/no-claim answers, but exclude it from claim score.
                is_abstention = bool(re.search(r"couldn't|could not|insufficient|not enough evidence|does not provide enough", answer, re.I))
                write_review_row(
                    writer, test_id, question, focus, answer, "", evidence, sources,
                    verdict="ABSTENTION" if is_abstention else "REVIEW_NO_CLAIMS",
                    notes="No factual claim was extracted; inspect the answer manually.",
                )
        f.flush()

    print(f"\nWrote: {OUTPUT_CSV}")
    print(f"Reviewable factual-claim rows: {total_claims}")
    print("Open the CSV. For each claim, choose supported, unsupported, or contradicted based only on retrieved_evidence.")
    print("Then run: python evaluate_faithfulness.py --score-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
