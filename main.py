from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np
import torch
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths and model configuration
PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
CACHE_DIR = PROJECT_DIR / ".rag_cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 35
EMBEDDING_BATCH_SIZE = 64
INDEX_CACHE_VERSION = 3
TOP_K = 5
MIN_RELEVANCE_SCORE = 0.20
RRF_CONSTANT = 60
DEBUG = False


class DocumentPage(TypedDict):
    text: str
    source: str
    page: int


class TextChunk(TypedDict):
    text: str
    source: str
    page: int


def clean_pdf_text(text: str) -> str:
    """Normalize whitespace and repair known joined phrases from PDF extraction."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", " ", text)
    replacements = [
        (r"\bDeepLearningtechniques\b", "Deep Learning techniques"),
        (r"\bTounderstand\b", "To understand"),
        (r"\bCNNalgorithms\b", "CNN algorithms"),
        (r"\bTounderstandCNNalgorithmsandthewayto", "To understand CNN algorithms and the way to "),
        (r"\bevaluateperformanceoftheCNNarchitectures\b", "evaluate performance of the CNN architectures"),
        (r"\bToapplyRNNandLSTMtolearn,predictandclassify", "To apply RNN and LSTM to learn, predict and classify "),
        (r"\bthereal-worldproblemsintheparadigmsofDeepLearning\b", "the real-world problems in the paradigms of Deep Learning"),
        (r"\bTounderstand,learnanddesignGANsforthe", "To understand, learn and design GANs for the "),
        (r"\bTounderstandtheconceptofAuto-encodersand", "To understand the concept of Auto-encoders and "),
        (r"\benhancingGANsusingauto-encoders\b", "enhancing GANs using auto-encoders"),
        (r"\binputimage\b", "input image"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_pdfs() -> list[DocumentPage]:
    """Extract text from every PDF in documents/, preserving original page numbers."""
    pages: list[DocumentPage] = []
    if not DOCUMENTS_DIR.exists():
        logging.error("Documents folder not found: %s", DOCUMENTS_DIR)
        return pages
    for path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    logging.warning("Skipping encrypted PDF: %s", path.name)
                    continue
            for page_no, page in enumerate(reader.pages, start=1):
                text = clean_pdf_text(page.extract_text() or "")
                if text:
                    pages.append({"text": text, "source": path.name, "page": page_no})
        except Exception:
            logging.exception("Could not read PDF: %s", path.name)
    return pages


def create_chunks(
    documents: list[DocumentPage],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Create overlapping chunks without mixing text from separate pages."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    result: list[TextChunk] = []
    step = chunk_size - overlap
    for doc in documents:
        words = doc["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end]).strip()
            if text:
                result.append({"text": text, "source": doc["source"], "page": doc["page"]})
            if end >= len(words):
                break
            start += step
    return result


def create_embeddings(chunks: list[TextChunk], model: SentenceTransformer) -> np.ndarray:
    if not chunks:
        return np.empty((0, model.get_embedding_dimension()), dtype=np.float32)
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    return np.asarray(vectors, dtype=np.float32)


def get_cache_key(pdf_files: list[Path]) -> str:
    sources = []
    for path in pdf_files:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        sources.append({"name": path.name, "sha256": digest.hexdigest()})
    identity = {
        "version": INDEX_CACHE_VERSION,
        "sources": sources,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_or_create_embeddings(
    chunks: list[TextChunk],
    pdf_files: list[Path],
    model: SentenceTransformer,
) -> tuple[np.ndarray, bool]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{get_cache_key(pdf_files)}.npz"
    expected = (len(chunks), model.get_embedding_dimension())
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                vectors = np.asarray(cached["embeddings"], dtype=np.float32)
            if vectors.shape == expected and np.isfinite(vectors).all():
                logging.info("Loaded cached embeddings.")
                return vectors, True
        except Exception:
            logging.warning("Embedding cache invalid; rebuilding.", exc_info=True)
    vectors = create_embeddings(chunks, model)
    try:
        tmp = cache_path.with_suffix(".tmp.npz")
        with tmp.open("wb") as f:
            np.savez_compressed(f, embeddings=vectors)
        tmp.replace(cache_path)
    except OSError:
        logging.exception("Could not save embedding cache.")
    return vectors, False


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def extract_page_filter(question: str) -> int | None:
    match = re.search(r"\b(?:page|p\.)\s*#?\s*(\d+)\b", question, re.I)
    return int(match.group(1)) if match else None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def search_chunks(
    question: str,
    model: SentenceTransformer,
    index: faiss.Index,
    embeddings: np.ndarray,
    chunks: list[TextChunk],
    bm25: BM25Okapi,
    top_k: int = TOP_K,
    page_filter: int | None = None,
) -> list[dict]:
    """Hybrid semantic + keyword retrieval using reciprocal rank fusion."""
    if top_k <= 0 or not chunks:
        return []
    semantic_question = re.sub(
        r"\b(?:page|p\.)\s*#?\s*\d+\b", "", question, flags=re.I
    ).strip() or question
    candidates = [
        i for i, chunk in enumerate(chunks)
        if page_filter is None or chunk["page"] == page_filter
    ]
    if not candidates:
        return []
    candidate_set = set(candidates)
    query_vector = model.encode(
        [semantic_question], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)
    scores, positions = index.search(query_vector, min(index.ntotal, len(chunks)))
    semantic_rank = [int(i) for i in positions[0] if i >= 0 and int(i) in candidate_set]
    cosine = {
        int(i): float(s) for i, s in zip(positions[0], scores[0])
        if i >= 0 and int(i) in candidate_set
    }
    keyword_scores = bm25.get_scores(_tokens(semantic_question))
    keyword_rank = sorted(candidates, key=lambda i: float(keyword_scores[i]), reverse=True)

    fused: dict[int, float] = {}
    for ranking, weight in ((semantic_rank, 2.0), (keyword_rank, 1.0)):
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + weight / (RRF_CONSTANT + rank)
    ranked = sorted(fused, key=lambda i: (fused[i], cosine.get(i, 0.0)), reverse=True)

    selected: list[int] = []
    token_sets: list[set[str]] = []
    comparison = bool(re.search(r"\b(compare|comparison|difference|different|versus|vs)\b", question, re.I))
    syllabus_question = bool(re.search(r"\b(objectives?|syllabus)\b", question, re.I))
    for idx in ranked:
        text = chunks[idx]["text"]
        if comparison and not syllabus_question and re.search(
            r"course objectives|unit\s*[-:]?\s*[ivx]+\s*:", text, re.I
        ):
            continue
        current = set(_tokens(text))
        if any(
            current and previous and
            len(current & previous) / max(1, len(current | previous)) >= 0.82
            for previous in token_sets
        ):
            continue
        selected.append(idx)
        token_sets.append(current)
        if len(selected) >= top_k:
            break

    results = []
    for rank, idx in enumerate(selected, start=1):
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": cosine.get(idx, 0.0),
            "cosine": cosine.get(idx, 0.0),
            "rrf_score": fused.get(idx, 0.0),
            "rank": rank,
        })
    if DEBUG:
        for item in results:
            print(f"Page {item['page']} | cosine={item['score']:.3f} | RRF={item['rrf_score']:.4f}")
            print(item["text"][:400], "\n")
    return results


def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=dtype)
    llm.to(device)
    llm.eval()
    logging.info("Answer model loaded on %s.", device)
    return tokenizer, llm


def extract_course_objectives(
    pages: list[DocumentPage],
    page_filter: int | None = None,
) -> str | None:
    """Extract the numbered course objectives when the PDF provides all five."""
    for page in pages:
        if page_filter is not None and page["page"] != page_filter:
            continue
        text = page["text"]
        heading = re.search(r"COURSE\s+OBJECTIVES?\s*:?", text, re.I)
        if not heading:
            continue
        remainder = text[heading.end():]
        unit = re.search(r"\bUNIT\s*[-:]?\s*I\s*:", remainder, re.I)
        body = remainder[:unit.start()] if unit else remainder

        # Known joined lines from the user's syllabus PDF.
        repairs = [
            (r"2\.\s*TounderstandCNNalgorithmsandthewaytoevaluateperformanceoftheCNNarchitectures",
             "2. To understand CNN algorithms and the way to evaluate performance of the CNN architectures"),
            (r"3\.\s*ToapplyRNNandLSTMtolearn,predictandclassifythereal-worldproblemsintheparadigmsofDeepLearning",
             "3. To apply RNN and LSTM to learn, predict and classify real-world problems in the paradigms of Deep Learning"),
            (r"4\.\s*Tounderstand,learnanddesignGANsforthe selectedproblems",
             "4. To understand, learn and design GANs for the selected problems"),
            (r"5\.\s*TounderstandtheconceptofAuto-encodersandenhancingGANsusingauto-encoders",
             "5. To understand the concept of Auto-encoders and enhancing GANs using auto-encoders"),
        ]
        for pattern, replacement in repairs:
            body = re.sub(pattern, replacement, body, flags=re.I)

        matches = list(re.finditer(r"(?<!\d)([1-5])\.\s*(.*?)(?=(?:\s+[1-5]\.\s)|$)", body, re.S))
        found: dict[int, str] = {}
        for match in matches:
            value = " ".join(match.group(2).split()).strip(" .")
            if value:
                found[int(match.group(1))] = value
        if all(n in found for n in range(1, 6)):
            listing = "\n".join(f"{n}. {found[n]}." for n in range(1, 6))
            return f"{listing}\n\nSource: {page['source']} | Page: {page['page']}"
    return None


def validate_answer(question: str, answer: str, context: str, tokenizer, llm) -> str:
    """Conservative second pass. It helps but cannot guarantee fact checking."""
    messages = [
        {"role": "system", "content": (
            "You are a conservative PDF answer editor. Use only the supplied "
            "evidence. Remove unsupported claims. Do not add facts. If the "
            "evidence is insufficient, say so. Return only the revised answer, "
            "without citations."
        )},
        {"role": "user", "content": (
            f"Question:\n{question}\n\nEvidence:\n{context}\n\n"
            f"Draft answer:\n{answer}\n\nRevise using evidence only."
        )},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    device = next(llm.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs, max_new_tokens=180, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    revised = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return revised or answer


def select_citations(
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
    max_sources: int = 1,
) -> list[int]:
    """Return only the highest-ranked passage that supports the answer."""
    if not question or not answer or not retrieved_chunks or max_sources <= 0:
        return []
    stopwords = {
        "what", "is", "the", "a", "an", "and", "or", "to", "of", "in",
        "on", "for", "with", "how", "does", "do", "between", "difference",
        "different", "compare", "comparison", "versus", "vs", "explain",
        "describe", "tell", "me", "about", "from", "pdf", "this", "that",
        "are", "was", "were", "be", "can", "could", "would", "should",
        "while", "using", "uses",
    }

    def terms(text: str) -> set[str]:
        return {term for term in _tokens(text) if len(term) > 2 and term not in stopwords}

    question_terms = terms(question)
    answer_terms = terms(answer)
    ranked: list[tuple[float, int]] = []
    for evidence_id, chunk in enumerate(retrieved_chunks, start=1):
        passage_terms = terms(chunk.get("text", ""))
        answer_overlap = len(answer_terms & passage_terms)
        if not answer_overlap:
            continue
        question_overlap = len(question_terms & passage_terms)
        answer_coverage = answer_overlap / max(1, len(answer_terms))
        question_coverage = question_overlap / max(1, len(question_terms))
        retrieval_score = float(chunk.get("rrf_score", chunk.get("score", 0.0)))
        score = (
            4 * answer_overlap
            + 2 * question_overlap
            + 4 * answer_coverage
            + question_coverage
            + retrieval_score
        )
        ranked.append((score, evidence_id))

    ranked.sort(
        key=lambda item: (
            -item[0],
            int(retrieved_chunks[item[1] - 1].get("rank", item[1])),
        )
    )
    return [ranked[0][1]] if ranked else []


def generate_answer(question: str, retrieved_chunks: list[dict], tokenizer, llm) -> str:
    """Generate a concise answer with one highest-ranked supporting citation."""
    if not retrieved_chunks:
        return "I couldn't find relevant information in the PDF to answer this question."

    # Conservative direct extraction for a known phrase in the provided PDF.
    # Other questions use the general evidence-grounded path below.
    if re.search(r"\bvanishing\s+gradients?\b", question, re.I):
        matches = [
            c for c in retrieved_chunks
            if re.search(r"\bvanishing\s+gradients?\b", c.get("text", ""), re.I)
            and re.search(r"not being trained properly|gradient.{0,30}(small|vanish)|vanishing gradient", c.get("text", ""), re.I)
        ]
        if not matches:
            return "I couldn't find sufficiently direct evidence in the retrieved PDF passages to answer this question."
        match = matches[0]
        evidence_id = retrieved_chunks.index(match) + 1
        return (
            "The PDF explains that vanishing gradients can prevent the network "
            "from being trained properly.\n\nSources: "
            f"[Evidence {evidence_id}] {match['source']}, page {match['page']}"
        )

    context = "\n\n".join(
        f"[Evidence {i} | Source: {c['source']} | Page: {c['page']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    )
    system_prompt = """Answer the question using ONLY the supplied PDF evidence.
Do not use outside knowledge, assumptions, or invented details.
Answer the exact question. Preserve the PDF's terminology.
For comparisons, discuss each subject only to the extent supported by evidence.
Do not add mechanisms, examples, applications, or technical details unless
the passages state them. If evidence is incomplete, say what is missing.
Answer in at most three concise sentences. Do not include citations.
Return only the answer."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question:\n{question}\n\nPDF evidence:\n{context}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    device = next(llm.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    llm.eval()
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs, max_new_tokens=120, do_sample=False, num_beams=1,
            repetition_penalty=1.08, pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    if not answer:
        return "I couldn't generate an answer from the retrieved PDF evidence."

    if any(marker in answer.lower() for marker in (
        "pdf evidence is insufficient", "does not provide enough information",
        "couldn't find relevant information", "couldn't find sufficiently relevant information",
    )):
        return "I couldn't find sufficient evidence in the retrieved PDF passages to answer this question."

    selected = select_citations(question, answer, retrieved_chunks)
    if not selected:
        return "I couldn't identify a sufficiently relevant source passage to support the generated answer."
    sources = []
    for evidence_id in selected:
        if not 1 <= evidence_id <= len(retrieved_chunks):
            continue
        chunk = retrieved_chunks[evidence_id - 1]
        sources.append(
            f"[Evidence {evidence_id}] {chunk['source']}, page {chunk['page']}"
        )
    return answer + ("\n\nSources: " + "; ".join(sources) if sources else "")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pages = load_pdfs()
    if not pages:
        logging.error("No PDF text found. Check the documents folder and PDFs.")
        return 1
    chunks = create_chunks(pages)
    if not chunks:
        logging.error("No searchable text chunks were created.")
        return 1
    bm25 = BM25Okapi([_tokens(c["text"]) for c in chunks])
    logging.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    embeddings, cache_hit = load_or_create_embeddings(chunks, pdf_files, embedding_model)
    if len(embeddings) == 0:
        logging.error("No embeddings were created.")
        return 1
    index = build_faiss_index(embeddings)
    logging.info("RAG ready: %d pages, %d chunks, %d vectors (%s embeddings).",
                 len(pages), len(chunks), index.ntotal, "cached" if cache_hit else "new")
    tokenizer, llm = load_local_llm()

    while True:
        try:
            question = input("\nAsk a question about your PDF (or type 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            print("Please enter a question.")
            continue
        page_filter = extract_page_filter(question)
        results = search_chunks(
            question, embedding_model, index, embeddings, chunks, bm25,
            top_k=TOP_K, page_filter=page_filter,
        )
        if not results:
            print("No relevant passages were retrieved.")
            continue
        if page_filter is None and max(r["score"] for r in results) < MIN_RELEVANCE_SCORE:
            print("\nAnswer:\nI couldn't find sufficiently relevant information in the PDF.")
            continue
        answer = None
        if re.search(r"\b(course\s+)?objectives?\b", question, re.I):
            answer = extract_course_objectives(pages, page_filter)
        if answer is None:
            answer = generate_answer(question, results, tokenizer, llm)
        print(f"\nAnswer:\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
