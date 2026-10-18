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
from sentence_transformers import SentenceTransformer, CrossEncoder
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
INDEX_CACHE_VERSION = 4
TOP_K = 5
RETRIEVAL_CANDIDATES = 20
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
    """Normalize PDF text while preserving paragraph boundaries for safer chunking."""
    if not text:
        return ""

    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)

    # Preserve blank-line paragraph breaks. Collapse only line wraps inside a
    # paragraph, rather than flattening the whole page into one text block.
    paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in paragraphs if part.strip()]
    text = "\n\n".join(paragraphs)

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

    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return "\n\n".join(paragraphs)


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
    """Chunk within paragraphs so one concept is less likely to bleed into another."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    result: list[TextChunk] = []
    step = chunk_size - overlap
    for doc in documents:
        # The PDF cleaner preserves blank lines. Do not join adjacent paragraphs
        # by default: adjacent paragraphs may discuss different concepts.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", doc["text"]) if p.strip()]
        if not paragraphs:
            paragraphs = [doc["text"].strip()] if doc["text"].strip() else []

        for paragraph in paragraphs:
            words = paragraph.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end]).strip()
                if chunk_text:
                    result.append({"text": chunk_text, "source": doc["source"], "page": doc["page"]})
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


def load_reranker() -> CrossEncoder:
    """Load a cross-encoder used to rerank the hybrid retriever's shortlist."""
    logging.info("Loading reranker: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def rerank_chunks(
    question: str,
    candidates: list[dict],
    reranker: CrossEncoder,
    top_k: int = TOP_K,
) -> list[dict]:
    """Re-score retrieved query/passage pairs and return the most relevant passages.

    The original cosine score is retained in `score` for existing relevance checks;
    the cross-encoder score is stored separately as `rerank_score`.
    """
    if not candidates or top_k <= 0:
        return []
    pairs = [(question, str(item.get("text", ""))) for item in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = []
    for item, score in zip(candidates, scores):
        updated = dict(item)
        updated["rerank_score"] = float(score)
        ranked.append(updated)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    selected = ranked[:top_k]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    return selected


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
            "evidence. Remove unsupported claims. Do not add facts. Do not transfer "
            "a property or disadvantage from one concept to another concept that is "
            "merely listed as an example nearby. A mention of a topic is not evidence "
            "for every property of that topic. If the evidence does not specifically "
            "support the requested property of the requested topic, state that the "
            "PDF does not provide enough evidence. Return only the revised answer, "
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



def _requested_relation(question: str) -> tuple[str, str] | None:
    """Return a narrowly supported topic/relation pair, or None."""
    # Ignore output instructions appended after the actual question, so they
    # do not become part of the requested topic.
    q = question.strip().split("?", 1)[0]
    q = re.sub(r"[?!.]+$", "", q)
    patterns = (
        (r"what\s+are\s+(?:the\s+)?(?:disadvantages?|drawbacks?|limitations?|weaknesses?)\s+of\s+(.+)", "limitation"),
        (r"what\s+is\s+(?:the\s+)?(?:disadvantage|drawback|limitation|weakness)\s+of\s+(.+)", "limitation"),
        (r"what\s+is\s+(?:the\s+)?(.+)", "definition"),
        (r"define\s+(.+)", "definition"),
        (r"explain\s+(.+)", "definition"),
        (r"what\s+does\s+(.+?)\s+use", "uses"),
    )
    for pattern, relation in patterns:
        match = re.fullmatch(pattern, q, re.IGNORECASE)
        if match:
            topic = re.sub(r"^(?:a|an|the)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
            return (topic, relation) if topic else None
    if re.search(r"\bvanishing\s+gradients?\b", q, re.IGNORECASE) and re.search(
        r"\b(how|why|effect|affect|impact|prevent|cause|training)\b", q, re.IGNORECASE
    ):
        return ("vanishing gradients", "gradient_effect")
    return None


def _answer_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _normalize_sentence_for_match(sentence: str) -> str:
    """Ignore PDF list glyphs at sentence start while preserving sentence text."""
    sentence = re.sub(r"^\s*[\uf097\uf0b7\u2022\u25cf\u25aa\u2023]\s*", "", sentence)
    return " ".join(sentence.split())


def _supports_requested_relation(question: str, sentence: str) -> bool:
    requested = _requested_relation(question)
    if requested is None:
        return False
    topic, relation = requested
    topic_re = r"\b" + r"\s+".join(re.escape(word) for word in topic.split()) + r"\b"
    if relation == "definition":
        return bool(re.search(topic_re + r"\s+(?:is|are|means|refers\s+to|is\s+defined\s+as)\b", sentence, re.IGNORECASE))
    if relation == "uses":
        return bool(re.search(r"^\s*(?:(?:the|a|an)\s+)?" + topic_re + r"\s+uses?\b", sentence, re.IGNORECASE))
    if relation == "limitation":
        relation_re = r"(?:disadvantage|drawback|limitation|weakness)(?:s)?"
        return bool(
            re.search(relation_re + r"\s+of\s+" + topic_re + r"\s+(?:is|are|include|includes)\b", sentence, re.IGNORECASE)
            or re.search(topic_re + r"\s+(?:has|have)\s+(?:a\s+)?(?:\w+\s+){0,3}" + relation_re + r"\b", sentence, re.IGNORECASE)
        )
    if relation == "gradient_effect":
        return bool(
            re.search(r"\bvanishing\s+gradients?\b", sentence, re.IGNORECASE)
            and re.search(r"\b(?:prevent|stop|cause|lead\s+to|result\s+in)\b", sentence, re.IGNORECASE)
            and re.search(r"\b(?:train(?:ed|ing)?|network|gradient)\b", sentence, re.IGNORECASE)
        )
    return False


def _verified_answer_sentences(question: str, answer: str, passages: list[dict]):
    """Accept only complete generated sentences found verbatim in evidence."""
    generated = _answer_sentences(answer)
    if not generated or _requested_relation(question) is None:
        return []
    verified = []
    for proposed in generated:
        match = None
        for index, passage in enumerate(passages, start=1):
            for source_sentence in _answer_sentences(str(passage.get("text", ""))):
                if (
                    _normalize_sentence_for_match(proposed)
                    == _normalize_sentence_for_match(source_sentence)
                    and _supports_requested_relation(question, source_sentence)
                ):
                    match = (_normalize_sentence_for_match(source_sentence), index)
                    break
            if match:
                break
        if match is None:
            return []
        verified.append(match)
    return verified


def _first_supported_definition(question: str, passages: list[dict]):
    """Extract a direct definition from ranked evidence for supported definition queries."""
    requested = _requested_relation(question)
    if requested is None or requested[1] != "definition":
        return None
    for index, passage in enumerate(passages, start=1):
        for sentence in _answer_sentences(str(passage.get("text", ""))):
            if _supports_requested_relation(question, sentence):
                return (_normalize_sentence_for_match(sentence), index)
    return None


def _first_supported_limitation(question: str, passages: list[dict]):
    """Extract an exact, topic-linked limitation sentence from retrieved evidence."""
    requested = _requested_relation(question)
    if requested is None or requested[1] != "limitation":
        return None
    for index, passage in enumerate(passages, start=1):
        for sentence in _answer_sentences(str(passage.get("text", ""))):
            if _supports_requested_relation(question, sentence):
                return (_normalize_sentence_for_match(sentence), index)
    return None


def select_citations(question, answer, passages):
    """Select only passages containing exact, relation-checked answer sentences."""
    return list(dict.fromkeys(index for _, index in _verified_answer_sentences(question, answer, passages)))


def generate_answer(question: str, retrieved_chunks: list[dict], tokenizer, llm) -> str:
    """Generate once, then return only exact evidence sentences that pass a supported relation rule."""
    if not retrieved_chunks:
        return "I couldn't find relevant information in the PDF to answer this question."
    if _requested_relation(question) is None:
        return "I couldn't verify a direct answer to this question from the retrieved PDF passages."

    context = "\n\n".join(
        f"[Evidence {i} | Source: {c['source']} | Page: {c['page']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    )
    messages = [
        {"role": "system", "content": (
            "Select only exact sentence(s) from the supplied PDF evidence that directly state "
            "the requested relation. Copy every selected sentence verbatim, including punctuation. "
            "Do not paraphrase, combine, explain, or add facts. If no sentence directly answers "
            "the question, return exactly: INSUFFICIENT EVIDENCE. Return only selected sentence(s) "
            "or that abstention phrase."
        )},
        {"role": "user", "content": f"Question:\n{question}\n\nPDF evidence:\n{context}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    device = next(llm.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    llm.eval()
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs, max_new_tokens=220, do_sample=False, num_beams=1,
            repetition_penalty=1.08, pad_token_id=tokenizer.eos_token_id,
        )
    proposed = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    if not proposed or proposed.upper() == "INSUFFICIENT EVIDENCE":
        return "I couldn't find sufficient direct evidence in the retrieved PDF passages to answer this question."

    verified = _verified_answer_sentences(question, proposed, retrieved_chunks)
    if not verified:
        direct_definition = _first_supported_definition(question, retrieved_chunks)
        if direct_definition is not None:
            verified = [direct_definition]
    if not verified:
        direct_limitation = _first_supported_limitation(question, retrieved_chunks)
        if direct_limitation is not None:
            verified = [direct_limitation]
    if not verified:
        return "I couldn't verify a direct answer to this question from the retrieved PDF passages."

    sources = []
    for _, evidence_id in verified:
        chunk = retrieved_chunks[evidence_id - 1]
        source = f"[Evidence {evidence_id}] {chunk['source']}, page {chunk['page']}"
        if source not in sources:
            sources.append(source)
    answer = " ".join(sentence for sentence, _ in verified)
    return answer + "\n\nSources: " + "; ".join(sources)

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
            if DEBUG:
                print("\n=== CHUNKS PASSED TO ANSWER GENERATION ===")
                for i, chunk in enumerate(results, start=1):
                    print(
                        f"\nEvidence {i} | Page: {chunk.get('page')} "
                        f"| Score: {chunk.get('score', 'N/A')}\n"
                        f"{chunk.get('text', '')}\n"
                    )
            answer = generate_answer(question, results, tokenizer, llm)
        print(f"\nAnswer:\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
