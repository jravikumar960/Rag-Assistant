"""Conversation-aware FastAPI backend for the local PDF RAG application.

Run from the project directory (the folder containing main.py and documents/):
    python -m uvicorn api:app --reload

Conversation memory is stored in process memory and is cleared when the API restarts.
Clients should pass the session_id returned by the first /ask response on later
requests to keep follow-up questions in the same conversation.
"""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from threading import RLock
from time import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

import main as rag

logger = logging.getLogger("rag_api")
CANDIDATE_COUNT = max(20, int(getattr(rag, "RETRIEVAL_CANDIDATES", 20)))
FINAL_K = int(getattr(rag, "TOP_K", 5))
MAX_SESSIONS = 500
MAX_HISTORY_MESSAGES = 12


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to answer from the PDF documents.",
        examples=["How does Random Forest reduce overfitting?"],
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Session ID returned by an earlier /ask response; omit for a new conversation.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty or whitespace.")
        return value

    @field_validator("session_id")
    @classmethod
    def session_id_must_be_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value):
            raise ValueError("session_id may contain only letters, numbers, dots, underscores, colons, and hyphens.")
        return value


class SourceItem(BaseModel):
    filename: str
    page: int
    excerpt: str
    cosine_score: float | None = None
    rerank_score: float | None = None


class QuestionResponse(BaseModel):
    session_id: str
    question: str
    resolved_question: str
    conversation_context_used: bool
    answer: str
    sources: list[SourceItem]
    reranking_used: bool


def is_context_dependent_question(question: str) -> bool:
    """Detect pronoun-based or underspecified follow-up questions."""
    q = (question or "").strip()
    return bool(
        re.search(r"\b(it|its|they|them|their|this|that|these|those|former|latter)\b", q, re.I)
        or re.match(r"^(?:what|how|why) about (?:it|this|that|them|those)\b", q, re.I)
        or re.fullmatch(r"(?:tell me more|explain more|give me an example|give examples|why|how so)[?.!]*", q, re.I)
        or re.fullmatch(r"what are the (?:advantages|disadvantages|limitations|benefits|drawbacks)[?.!]*", q, re.I)
    )


def extract_explicit_topic(question: str) -> str:
    """Conservatively extract a topic from common standalone question forms."""
    q = (question or "").strip().strip("?.! ")
    if not q:
        return ""

    # Common evaluative questions: "What are the disadvantages of Random Forest?"
    match = re.match(
        r"^(?:what are|list|explain)\s+(?:the\s+)?(?:advantages|disadvantages|limitations|benefits|drawbacks|applications|uses)\s+of\s+(.+)$",
        q, re.I,
    )
    if match:
        return match.group(1).strip(" ?.!")

    # Commands such as "Explain Random Forest" or "Describe gradient descent".
    match = re.match(
        r"^(?:please\s+)?(?:explain|define|describe|discuss|summarize|teach me about|tell me about)\s+(.+)$",
        q, re.I,
    )
    if match:
        return match.group(1).strip(" ?.!")

    # Identity / definition questions.
    match = re.match(r"^(?:what is|who is|what are|who are)\s+(.+)$", q, re.I)
    if match:
        candidate = match.group(1).strip(" ?.!")
        # Prefer the subject named after "in" where the wording is a subtopic of a concept.
        # Keep the full noun phrase otherwise, e.g. "bootstrap sampling".
        return candidate

    # Questions explicitly naming a topic before a common action.
    match = re.match(
        r"^how does\s+(.+?)\s+(?:reduce|improve|work|help|make|build|use|perform|support|handle|overcome|calculate|select|predict|classify)\b.*$",
        q, re.I,
    )
    if match:
        return match.group(1).strip(" ?.!")

    # Keep a recognizable explicit topic even in slightly different wording.
    match = re.search(r"\b(Random Forest)\b", q, re.I)
    if match:
        return match.group(1)

    return ""


def resolve_followup_question(question: str, topic: str) -> str:
    """Resolve common pronouns to the previous topic without adding prompt instructions."""
    q = (question or "").strip()
    if not topic or not q or not is_context_dependent_question(q):
        return q

    topic = topic.strip().rstrip("?.!")

    # Natural phrasing for frequent follow-ups.
    match = re.match(r"^(?:what are|list)\s+(?:its|their)\s+(advantages|disadvantages|limitations|benefits|drawbacks)\s*[?.!]*$", q, re.I)
    if match:
        descriptor = match.group(1).lower()
        return f"What are the {descriptor} of {topic}?"

    match = re.match(r"^what about\s+(?:it|this|that|them|those)\s*[?.!]*$", q, re.I)
    if match:
        return f"What about {topic}?"

    # Deterministic substitution retains the user's actual question and intent.
    substitutions = [
        (r"\bits\b", f"{topic}'s"),
        (r"\btheir\b", f"{topic}'s"),
        (r"\bit\b", topic),
        (r"\bthey\b", topic),
        (r"\bthem\b", topic),
        (r"\bthis\b", topic),
        (r"\bthat\b", topic),
        (r"\bthese\b", topic),
        (r"\bthose\b", topic),
    ]
    resolved = q
    for pattern, replacement in substitutions:
        resolved = re.sub(pattern, replacement, resolved, flags=re.I)
    resolved = re.sub(r"\s+", " ", resolved).strip()
    return resolved


def load_rag_resources() -> dict[str, Any] | None:
    """Load the document index and models once when the API starts."""
    pdf_files = sorted(rag.DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s. API will start but /ask returns 503.", rag.DOCUMENTS_DIR)
        return None

    pages = rag.load_pdfs()
    if not pages:
        raise RuntimeError("PDF files exist, but no readable PDF text was extracted.")
    chunks = rag.create_chunks(pages)
    if not chunks:
        raise RuntimeError("No searchable chunks were created from the PDFs.")

    logger.info("Loading embedding model: %s", rag.EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL)
    embeddings, cache_hit = rag.load_or_create_embeddings(chunks, pdf_files, embedding_model)
    index = rag.build_faiss_index(embeddings)
    bm25 = BM25Okapi([rag._tokens(chunk["text"]) for chunk in chunks])

    reranker = None
    reranker_error = None
    try:
        reranker = rag.load_reranker()
    except Exception as exc:
        reranker_error = str(exc)
        logger.exception("Cross-Encoder failed to load; hybrid retrieval will still work.")

    tokenizer, llm = rag.load_local_llm()
    logger.info(
        "RAG ready: %d PDF page(s), %d chunk(s); embedding cache hit=%s; reranker=%s",
        len(pages), len(chunks), cache_hit, reranker is not None,
    )
    return {
        "pages": pages,
        "chunks": chunks,
        "embedding_model": embedding_model,
        "embeddings": embeddings,
        "index": index,
        "bm25": bm25,
        "reranker": reranker,
        "reranker_error": reranker_error,
        "tokenizer": tokenizer,
        "llm": llm,
        "pdf_files": [p.name for p in pdf_files],
        "cache_hit": cache_hit,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy RAG resources and initialize session memory once."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app.state.rag_resources = None
    app.state.rag_load_error = None
    app.state.sessions = {}
    app.state.session_lock = RLock()
    try:
        app.state.rag_resources = load_rag_resources()
    except Exception as exc:
        app.state.rag_load_error = str(exc)
        logger.exception("RAG resources could not be loaded.")
    yield
    app.state.rag_resources = None
    app.state.sessions.clear()


app = FastAPI(
    title="RAG Assistant API",
    description="Conversation-aware PDF Q&A using hybrid retrieval, Cross-Encoder reranking, and local Qwen.",
    version="2.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check(request: Request) -> dict[str, Any]:
    resources = request.app.state.rag_resources
    error = request.app.state.rag_load_error
    return {
        "status": "ok" if resources is not None else "not_ready",
        "rag_ready": resources is not None,
        "pdf_count": len(resources["pdf_files"]) if resources else 0,
        "reranking_enabled": bool(resources and resources["reranker"] is not None),
        "active_sessions": len(request.app.state.sessions),
        "message": "RAG API is ready." if resources is not None else error or f"Add readable PDFs to {rag.DOCUMENTS_DIR} and restart the API.",
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(payload: QuestionRequest, request: Request) -> QuestionResponse:
    """Answer a question, resolving follow-up references within its session."""
    resources = request.app.state.rag_resources
    if resources is None:
        detail = request.app.state.rag_load_error or f"RAG is not ready. Add readable PDFs to {rag.DOCUMENTS_DIR} and restart the API."
        raise HTTPException(status_code=503, detail=detail)

    original_question = payload.question
    session_id = payload.session_id or uuid4().hex
    lock: RLock = request.app.state.session_lock

    # Snapshot the existing session and resolve the current question.
    with lock:
        sessions = request.app.state.sessions
        session = sessions.get(session_id)
        if session is None:
            if len(sessions) >= MAX_SESSIONS:
                oldest_id = min(sessions, key=lambda sid: sessions[sid]["updated"])
                sessions.pop(oldest_id, None)
            session = {"messages": [], "topic": "", "updated": time()}
            sessions[session_id] = session
        previous_topic = str(session.get("topic", ""))
        history = list(session.get("messages", []))

    context_dependent = is_context_dependent_question(original_question)
    resolved_question = resolve_followup_question(original_question, previous_topic) if context_dependent else original_question
    if context_dependent and previous_topic and resolved_question != original_question:
        query_strategy = "session-topic-anchored follow-up"
        context_used = True
    elif context_dependent and not previous_topic:
        query_strategy = "follow-up without a known topic; original query retained"
        context_used = False
        resolved_question = original_question
    else:
        query_strategy = "standalone question"
        context_used = False

    page_filter = rag.extract_page_filter(resolved_question)
    candidates = rag.search_chunks(
        question=resolved_question,
        model=resources["embedding_model"],
        index=resources["index"],
        embeddings=resources["embeddings"],
        chunks=resources["chunks"],
        bm25=resources["bm25"],
        top_k=CANDIDATE_COUNT,
        page_filter=page_filter,
    )

    passages = candidates
    reranking_used = False
    if candidates and resources["reranker"] is not None:
        try:
            passages = rag.rerank_chunks(
                question=resolved_question,
                candidates=candidates,
                reranker=resources["reranker"],
                top_k=FINAL_K,
            )
            reranking_used = True
        except Exception:
            logger.exception("Reranking failed; using hybrid retrieval order.")
            passages = candidates[:FINAL_K]
    else:
        passages = candidates[:FINAL_K]

    if not passages:
        answer = "I couldn't find relevant information in the retrieved PDF passages to answer this question."
        sources: list[SourceItem] = []
    elif page_filter is None and max(float(item.get("score", 0.0)) for item in passages) < rag.MIN_RELEVANCE_SCORE:
        answer = "I couldn't find sufficiently relevant information in the PDFs to answer this question."
        sources = []
    else:
        answer = None
        if re.search(r"\b(course\s+)?objectives?\b", resolved_question, flags=re.IGNORECASE):
            answer = rag.extract_course_objectives(resources["pages"], page_filter)
        if answer is None:
            answer = rag.generate_answer(
                resolved_question, passages, resources["tokenizer"], resources["llm"]
            )
        evidence_ids = [
            int(match) for match in re.findall(r"\[Evidence\s+(\d+)\]", answer)
        ]
        cited_passages = [
            passages[index - 1] for index in evidence_ids
            if 1 <= index <= len(passages)
        ]
        if not evidence_ids:
            objective_source = re.search(
                r"\n\nSource:\s*(.+?)\s*\|\s*Page:\s*(\d+)\s*$", answer
            )
            if objective_source:
                cited_passages = [
                    page for page in resources["pages"]
                    if page.get("source") == objective_source.group(1)
                    and page.get("page") == int(objective_source.group(2))
                ]
        sources = [
            SourceItem(
                filename=str(item.get("source", "unknown")),
                page=int(item.get("page", 0)),
                excerpt=str(item.get("text", ""))[:1000],
                cosine_score=float(item["score"]) if item.get("score") is not None else None,
                rerank_score=float(item["rerank_score"]) if item.get("rerank_score") is not None else None,
            )
            for item in cited_passages
        ]

    # Save both the user's actual wording and the returned answer for later follow-ups.
    # Only update the active topic from a standalone question with a recognized subject.
    with lock:
        session = request.app.state.sessions.setdefault(
            session_id, {"messages": [], "topic": "", "updated": time()}
        )
        if not context_dependent:
            new_topic = extract_explicit_topic(original_question)
            # Avoid carrying an unrelated prior topic into a new standalone question.
            session["topic"] = new_topic
        session["messages"].append({"role": "user", "content": original_question})
        session["messages"].append({"role": "assistant", "content": answer})
        session["messages"] = session["messages"][-MAX_HISTORY_MESSAGES:]
        session["updated"] = time()

    return QuestionResponse(
        session_id=session_id,
        question=original_question,
        resolved_question=resolved_question,
        conversation_context_used=context_used,
        answer=answer,
        sources=sources,
        reranking_used=reranking_used,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
