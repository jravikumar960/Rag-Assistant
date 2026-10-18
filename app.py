from io import BytesIO

import re

from pathlib import Path

from textwrap import dedent

from time import time

from uuid import uuid4

import streamlit as st

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from conversation_store import (

  delete_conversations,

  load_conversations,

  save_conversation,

)

import main as rag

# ============================================================

# CONFIGURATION

# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = PROJECT_DIR / "documents"

HISTORY_DB = PROJECT_DIR / ".rag_cache" / "conversations.sqlite3"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(

  page_title="RAG Assistant",

  page_icon=":material/auto_awesome:",

  layout="wide",

  initial_sidebar_state="expanded",

)

st.markdown(

  dedent("""

  <style>

  .stApp {

    background:

      radial-gradient(ellipse at 50% -18%, rgba(105, 91, 255, .17), transparent 48%),

      #0C0F15;

    background-attachment: fixed;

  }

  .rag-hero {

    max-width: 760px;

    margin: 3.4rem auto 2rem;

    padding: 1rem 1.25rem 1.75rem;

    text-align: center;

  }

  .rag-eyebrow {

    display: inline-flex;

    align-items: center;

    gap: .45rem;

    padding: .4rem .75rem;

    border: 1px solid rgba(139, 124, 255, .28);

    border-radius: 999px;

    background: rgba(139, 124, 255, .09);

    color: #bcb2ff;

    font-size: .78rem;

    font-weight: 650;

    letter-spacing: .04em;

  }

  .rag-hero h1 {

    margin: 1.1rem 0 .65rem;

    font-size: clamp(2.3rem, 6vw, 4rem);

    font-weight: 720;

    letter-spacing: -.055em;

    line-height: 1.06;

  }

  .rag-hero p {

    max-width: 560px;

    margin: 0 auto;

    color: rgba(225, 229, 239, .72);

    font-size: 1.04rem;

    line-height: 1.7;

  }

  div[data-testid="stChatMessage"] {

    padding: 1rem 1.15rem;

    border: 1px solid rgba(145, 154, 180, .12);

    border-radius: 1.15rem;

    background: rgba(255, 255, 255, .025);

  }

  .st-key-chat_composer {

    border: 1px solid rgba(145, 154, 180, .24);

    border-radius: 1.1rem;

    background: rgba(18, 21, 30, .94);

    box-shadow: 0 12px 36px rgba(0, 0, 0, .18);

  }

  @media (max-width: 700px) {

    .rag-hero {

      margin: 1.8rem auto 1rem;

      padding: .5rem .25rem 1rem;

    }

    .rag-hero p {

      font-size: .96rem;

    }

    div[data-testid="stChatMessage"] {

      padding: .8rem;

    }

  }

  </style>

  """),

  unsafe_allow_html=True,

)

SUGGESTED_QUESTIONS = {

  "Summarize my PDFs": "Summarize the key ideas in my PDFs.",

  "Find key concepts": "What are the most important concepts in my PDFs?",

  "Explain a topic": "Explain a key topic from my PDFs in simple terms.",

}

def create_conversation() -> str:

  """Create and activate a conversation in this browser session."""

  conversation_id = uuid4().hex

  st.session_state.conversations[conversation_id] = {

    "title": "New chat",

    "messages": [],

    "updated": 0.0,

  }

  st.session_state.active_conversation_id = conversation_id

  return conversation_id

def persist_active_conversation() -> None:

  conversation_id = st.session_state.active_conversation_id

  save_conversation(

    HISTORY_DB,

    conversation_id,

    st.session_state.conversations[conversation_id],

  )

if "conversations" not in st.session_state:

  st.session_state.conversations = load_conversations(HISTORY_DB)

st.session_state.setdefault("active_conversation_id", None)

st.session_state.setdefault("pending_uploads", [])

if st.session_state.active_conversation_id not in st.session_state.conversations:

  create_conversation()

active_conversation = st.session_state.conversations[

  st.session_state.active_conversation_id

]

# ============================================================

# PDF SIGNATURE AND RAG INITIALIZATION

# ============================================================

def pdf_signature():

  """Detect added, removed, or modified PDFs."""

  pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

  return tuple(

    (

      pdf.name,

      pdf.stat().st_size,

      pdf.stat().st_mtime_ns,

    )

    for pdf in pdf_files

  )

@st.cache_resource(show_spinner="Loading PDF index and AI models...")

def load_rag(signature):

  """Build the retrieval index and load models once per PDF collection."""

  if signature != pdf_signature():

    raise RuntimeError(

      "The PDF library changed while its index was being prepared."

    )

  pages = rag.load_pdfs()

  if not pages:

    raise ValueError(

      "No readable PDF text found. Check your uploaded documents."

    )

  chunks = rag.create_chunks(pages)

  if not chunks:

    raise ValueError("No searchable text chunks were created.")

  embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL)

  pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

  embeddings, cache_hit = rag.load_or_create_embeddings(

    chunks,

    pdf_files,

    embedding_model,

  )

  index = rag.build_faiss_index(embeddings)

  bm25 = rag.BM25Okapi([rag._tokens(chunk['text']) for chunk in chunks])

  tokenizer, llm = rag.load_local_llm()

  # Cross-encoder is loaded once and reused across Streamlit reruns.
  # If it cannot be downloaded/loaded, keep hybrid retrieval available and
  # show a warning when answering rather than crashing the whole app.
  try:
    reranker = rag.load_reranker()
    reranker_error = None
  except Exception as exc:
    reranker = None
    reranker_error = str(exc)

  return {

    "pages": pages,

    "chunks": chunks,

    "embedding_model": embedding_model,

    "embeddings": embeddings,

    "index": index,

    "bm25": bm25,

    "tokenizer": tokenizer,

    "llm": llm,

    "reranker": reranker,

    "reranker_error": reranker_error,

    "cache_hit": cache_hit,

  }

# QUERY REWRITING

def build_conversation_context(messages, max_messages: int = 6, max_chars_per_message: int = 900) -> str:
    """Format a small amount of recent chat history for query rewriting."""
    recent_messages = messages[-max_messages:] if messages else []
    formatted = []

    for message in recent_messages:
        role = str(message.get("role", "")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            continue
        # Keep the user's text even when it was submitted with a PDF attachment.
        # A prompt such as "Explain Random Forest" can be the topic anchor for
        # later follow-ups, so attachment metadata must not erase its text.
        if role == "Assistant" and content.startswith("Added ") and "to your library" in content.lower():
            continue
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message].rstrip() + " …"
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)

# Follow-up detection and deterministic topic anchoring.
# Small local LLMs can rewrite pronouns to the wrong topic, so for clear
# follow-ups we keep the last explicit user topic in the retrieval query.
REFERENCE_PATTERN = re.compile(
    r"\b(it|its|they|them|their|theirs|this|that|these|those|former|latter)\b",
    flags=re.IGNORECASE,
)
GENERIC_FOLLOWUP_PATTERN = re.compile(
    r"^(?:(?:what about|how about|and what about)(?:\s+(?:it|this|that|them|those|the same))?|"
    r"tell me more|explain more|give me (?:an )?example|give examples|why is that|how so|why|how does it work)[?.!]*$",
    flags=re.IGNORECASE,
)
GENERIC_TOPICLESS_QUESTION_PATTERN = re.compile(
    r"^what are the (?:advantages|disadvantages|limitations|benefits|drawbacks|pros and cons)[?.!]*$",
    flags=re.IGNORECASE,
)
GENERIC_STARTER_QUESTIONS = {
    "summarize my pdfs",
    "what are the most important concepts in my pdfs?",
    "explain a key topic from my pdfs in simple terms.",
}

def is_context_dependent_question(question: str) -> bool:
    """Return True when the question likely refers to an earlier topic."""
    text = (question or "").strip()
    return bool(
        REFERENCE_PATTERN.search(text)
        or GENERIC_FOLLOWUP_PATTERN.fullmatch(text)
        or GENERIC_TOPICLESS_QUESTION_PATTERN.fullmatch(text)
    )

def find_explicit_topic_question(messages) -> str:
    """Find the most recent prior user question that names its topic explicitly."""
    for message in reversed(messages or []):
        if str(message.get("role", "")).lower() != "user":
            continue
        # Do not skip messages just because they carried an attachment. Their
        # text may contain the explicit subject (e.g. "Explain Random Forest").
        content = str(message.get("content", "")).strip()
        if not content or content.casefold() in GENERIC_STARTER_QUESTIONS:
            continue
        if is_context_dependent_question(content):
            continue
        # Ignore upload acknowledgements or generic prompts that do not anchor a topic.
        if content.casefold().startswith((
            "please add these pdfs",
            "add these pdfs",
            "summarize my pdfs",
            "what are the most important concepts in my pdfs",
            "explain a key topic from my pdfs",
        )):
            continue
        return content
    return ""


def extract_topic_from_question(topic_question: str) -> str:
    """Extract a simple topic phrase from a prior standalone user question."""
    topic = (topic_question or "").strip().strip("?.! ")
    prefixes = (
        r"^(?:please\s+)?(?:explain|describe|define|discuss|summarize|teach me about|tell me about)\s+",
        r"^(?:please\s+)?(?:what is|what are|who is|who are)\s+",
        r"^(?:give me an overview of|provide an overview of)\s+",
    )
    for pattern in prefixes:
        topic = re.sub(pattern, "", topic, flags=re.IGNORECASE).strip()
        # Remove only one recognized leading construction.
        if not re.match(r"^(?:please\s+)?(?:explain|describe|define|discuss|summarize|teach me about|tell me about|what is|what are|who is|who are)\b", topic, flags=re.IGNORECASE):
            break
    topic = re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.IGNORECASE).strip()
    return topic or (topic_question or "").strip().strip("?.! ")


def resolve_followup_question(question: str, topic_question: str) -> str:
    """Turn a context-dependent follow-up into a clean, standalone query.

    Keep this deterministic: instructions about how to answer belong in the
    generation prompt, not in the text embedded/searched by FAISS and BM25.
    """
    topic = extract_topic_from_question(topic_question)
    q = (question or "").strip()
    if not topic or not q:
        return q

    if re.search(r"\bdisadvantages?\b|\bdrawbacks?\b|\bcons\b", q, re.IGNORECASE):
        return f"What are the disadvantages of {topic}?"
    if re.search(r"\blimitations?\b", q, re.IGNORECASE):
        return f"What are the limitations of {topic}?"
    if re.search(r"\badvantages?\b|\bbenefits?\b|\bpros\b", q, re.IGNORECASE):
        return f"What are the advantages or benefits of {topic}?"
    if re.search(r"\bapplications?\b|\buses?\b|\bused for\b", q, re.IGNORECASE):
        return f"What are the applications or uses of {topic}?"
    if re.search(r"\bexamples?\b", q, re.IGNORECASE):
        return f"What are examples of {topic}?"

    # Resolve common questions where the pronoun is embedded in a longer query.
    # For example: "How does it reduce overfitting?" ->
    # "How does Random Forest reduce overfitting?"
    templates = (
        (r"^how\s+does\s+it\s+(.+?)[?.!]*$", lambda m: f"How does {topic} {m.group(1).strip()}?"),
        (r"^how\s+can\s+it\s+(.+?)[?.!]*$", lambda m: f"How can {topic} {m.group(1).strip()}?"),
        (r"^why\s+does\s+it\s+(.+?)[?.!]*$", lambda m: f"Why does {topic} {m.group(1).strip()}?"),
        (r"^why\s+is\s+it\s+(.+?)[?.!]*$", lambda m: f"Why is {topic} {m.group(1).strip()}?"),
        (r"^when\s+does\s+it\s+(.+?)[?.!]*$", lambda m: f"When does {topic} {m.group(1).strip()}?"),
        (r"^can\s+it\s+(.+?)[?.!]*$", lambda m: f"Can {topic} {m.group(1).strip()}?"),
        (r"^does\s+it\s+(.+?)[?.!]*$", lambda m: f"Does {topic} {m.group(1).strip()}?"),
        (r"^is\s+it\s+(.+?)[?.!]*$", lambda m: f"Is {topic} {m.group(1).strip()}?"),
        (r"^what\s+does\s+it\s+(.+?)[?.!]*$", lambda m: f"What does {topic} {m.group(1).strip()}?"),
        (r"^which\s+(.+?)\s+does\s+it\s+(.+?)[?.!]*$", lambda m: f"Which {m.group(1).strip()} does {topic} {m.group(2).strip()}?"),
    )
    for pattern, formatter in templates:
        match = re.match(pattern, q, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", formatter(match)).strip()

    if re.search(r"\bhow does it work\b|\bhow it works\b", q, re.IGNORECASE):
        return f"How does {topic} work?"
    if re.search(r"\bwhat is it\b|\bexplain it\b|\btell me more\b|\bexplain more\b", q, re.IGNORECASE):
        return f"Explain {topic} in more detail."

    # Generic pronoun substitution is preferable to sending instruction text to
    # the retriever. Handle possessives before the standalone pronoun "it".
    resolved = re.sub(r"\bits\b", f"{topic}'s", q, flags=re.IGNORECASE)
    resolved = re.sub(r"\b(it|this|that)\b", topic, resolved, flags=re.IGNORECASE)
    resolved = re.sub(r"\b(they|them|their|theirs)\b", topic, resolved, flags=re.IGNORECASE)
    resolved = re.sub(r"\s+", " ", resolved).strip()
    if not resolved.endswith(("?", "!", ".")):
        resolved += "?"
    return resolved


def rewrite_query(question: str, conversation_context: str, tokenizer, model) -> str:
    """Rewrite a follow-up as a standalone search query using the loaded local LLM."""
    question = question.strip()
    if not question or not conversation_context.strip():
        return question

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite follow-up questions for a PDF retrieval system. "
                "The conversation is ordered from oldest to newest. Focus first "
                "on the immediately preceding user question and assistant answer "
                "when resolving words such as it, its, they, this, or that. "
                "Do not switch to a different topic merely because it appeared "
                "earlier in the conversation. Produce exactly one concise, "
                "standalone SEARCH QUERY, not an answer. Preserve the user's "
                "intent and all technical names, numbers, dates, and page limits. "
                "Do not invent a topic. If the reference is genuinely ambiguous, "
                "return the original latest question unchanged. Output only the query."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{conversation_context}\n\n"
                f"Latest user question:\n{question}\n\n"
                "Standalone search query:"
            ),
        },
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)

        import torch
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            # Encoder-decoder models return generated tokens without the input prompt.
            new_tokens = output_ids[0]
        else:
            # Causal language models echo the prompt before newly generated tokens.
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        rewritten = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
        rewritten = rewritten.strip("\"'")

        for prefix in ("Search query:", "Standalone search query:", "Rewritten query:"):
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix):].strip()
                break

        if not rewritten or len(rewritten) > 500:
            return question
        return rewritten

    except Exception as exc:
        # Show the failure in the app as well as the terminal so it is not silent.
        print(f"Query rewriting failed; using original question: {exc}")
        st.warning(
            "Query rewriting failed, so retrieval is using your original question. "
            "Check the terminal for the technical error."
        )
        return question

# ============================================================
# COMPACT SOURCE CITATIONS
# ============================================================

def display_answer_with_sources(answer, evidence_items):
    """Display the answer and resolve a source citation to retrieved evidence."""
    raw_answer = answer.strip()
    evidence = None
    answer_without_sources = raw_answer

    # Parse the source block before removing the supporting-evidence section.
    source_match = re.search(
        r"\n{1,2}\s*(?:\*\*)?Sources?:\s*(.+)$",
        raw_answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if source_match:
        source_text = source_match.group(1).strip()
        answer_without_sources = raw_answer[:source_match.start()].strip()

        evidence_id = re.search(r"\[Evidence\s+(\d+)\]", source_text, re.IGNORECASE)
        if evidence_id:
            evidence_index = int(evidence_id.group(1)) - 1
            if 0 <= evidence_index < len(evidence_items):
                evidence = evidence_items[evidence_index]
        else:
            source_page = re.search(
                r"(.+?),\s*page\s*(\d+)",
                source_text,
                flags=re.IGNORECASE,
            )
            if source_page:
                source_name = source_page.group(1).strip()
                page_number = int(source_page.group(2))
                evidence = next(
                    (
                        item for item in evidence_items
                        if item.get("source") == source_name
                        and item.get("page") == page_number
                    ),
                    None,
                )

    clean_answer = re.split(
        r"\n\s*(?:\*\*)?\[Supporting Evidence:?\](?:\*\*)?\s*\n",
        answer_without_sources,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    st.markdown(clean_answer)

    if evidence is None:
        intentional_abstention = bool(re.search(
            r"\b(?:couldn't\s+(?:find|verify)|insufficient\s+evidence|"
            r"does\s+not\s+provide\s+enough\s+evidence|"
            r"doesn't\s+provide\s+enough\s+evidence)\b",
            clean_answer,
            flags=re.IGNORECASE,
        ))
        if source_match or not intentional_abstention:
            st.warning("Source could not be verified for this answer.")
        return

    _, citation_column = st.columns([5, 1.5])
    with citation_column:
        with st.popover(
            f"📄 {evidence['source']} · p. {evidence['page']}",
            help="Open the highest-ranked supporting passage",
            use_container_width=True,
        ):
            st.caption(f"{evidence['source']} — Page {evidence['page']}")
            st.write(evidence["text"])


# ============================================================
# SIDEBAR: CONVERSATION HISTORY AND RELATED ITEMS

# ============================================================

with st.sidebar:

  st.markdown("## :material/auto_awesome: RAG Assistant")

  if st.button(

    "New chat",

    icon=":material/edit_square:",

    type="primary",

    width="stretch",

  ):

    create_conversation()

    st.rerun()

  st.caption("Your history")

  conversation_ids = sorted(

    (

      conversation_id

      for conversation_id, conversation

      in st.session_state.conversations.items()

      if conversation["messages"]

    ),

    key=lambda item: st.session_state.conversations[item]["updated"],

    reverse=True,

  )

  if not conversation_ids:

    st.caption("Your conversations will appear here.")

  for conversation_id in conversation_ids:

    conversation = st.session_state.conversations[conversation_id]

    is_active = (

      conversation_id == st.session_state.active_conversation_id

    )

    if st.button(

      conversation["title"],

      key=f"history_{conversation_id}",

      icon=":material/chat_bubble_outline:",

      type="primary" if is_active else "secondary",

      width="stretch",

    ):

      st.session_state.active_conversation_id = conversation_id

      st.rerun()

  if conversation_ids:

    if st.button(

      "Delete chat history",

      key="delete_chat_history",

      icon=":material/delete:",

      width="stretch",

    ):

      st.session_state.confirm_delete_chat_history = True

  if st.session_state.get("confirm_delete_chat_history"):

    st.warning("Delete all saved conversations? This cannot be undone.")

    confirm_col, cancel_col = st.columns(2)

    if confirm_col.button(

      "Delete all",

      key="confirm_delete_chat_history_button",

      type="primary",

      width="stretch",

    ):

      delete_conversations(HISTORY_DB)

      st.session_state.conversations.clear()

      st.session_state.active_conversation_id = None

      st.session_state.pop("pending_question", None)

      st.session_state.pop("pending_uploads", None)

      st.session_state.pop("queued_prompt", None)

      st.session_state.confirm_delete_chat_history = False

      create_conversation()

      st.rerun()

    if cancel_col.button(

      "Cancel",

      key="cancel_delete_chat_history",

      width="stretch",

    ):

      st.session_state.confirm_delete_chat_history = False

      st.rerun()

  st.caption("Related items")

  for label, prompt in SUGGESTED_QUESTIONS.items():

    if st.button(

      label,

      key=f"related_{label}",

      icon=":material/arrow_outward:",

      width="stretch",

    ):

      st.session_state.queued_prompt = prompt

      st.rerun()

  current_pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))

  with st.expander(

    f"Your library · {len(current_pdfs)} PDF"

    f"{'s' if len(current_pdfs) != 1 else ''}",

    expanded=True,

    icon=":material/folder_open:",

  ):

    if current_pdfs:

      for pdf in current_pdfs:

        st.caption(f":material/picture_as_pdf: {pdf.name}")

    else:

      st.caption("Attach a PDF in the message box to get started.")

  st.caption("Private document search · powered locally")

# ============================================================

# APP HEADER AND RAG INITIALIZATION

# ============================================================

if not active_conversation["messages"]:

  st.markdown(

    '<div class="rag-hero">'

    '<div class="rag-eyebrow">YOUR AI RESEARCH SPACE</div>'

    '<h1>Ask your documents.</h1>'

    '<p>Get clear, grounded answers from your PDFs, with the source page '

    'always close at hand.</p>'

    '</div>',

    unsafe_allow_html=True,

  )

else:

  st.caption(":material/auto_awesome: LOCAL PDF ASSISTANT")

  st.subheader(active_conversation["title"])

signature = pdf_signature()

rag_data = None

if signature:

  try:

    with st.spinner("Preparing your document library..."):

      rag_data = load_rag(signature)

  except Exception as exc:

    st.error(f"Could not prepare the document library: {exc}")

    st.exception(exc)

if rag_data:

  unique_sources = {page["source"] for page in rag_data["pages"]}

  st.badge(

    f"{len(unique_sources)} source"

    f"{'s' if len(unique_sources) != 1 else ''} connected",

    icon=":material/check_circle:",

    color="green",

  )

elif not signature:

  st.badge(

    "Attach a PDF to begin",

    icon=":material/upload_file:",

    color="violet",

  )

active_conversation = st.session_state.conversations[

  st.session_state.active_conversation_id

]

if (

  rag_data

  and st.session_state.pending_uploads

  and not st.session_state.get("pending_question")

):

  uploaded_names = st.session_state.pop("pending_uploads")

  active_conversation["messages"].append({

    "role": "assistant",

    "content": (

      f"Added {', '.join(uploaded_names)} to your library. "

      "Your documents are ready—ask me anything about them."

    ),

    "evidence": [],

  })

  active_conversation["updated"] = time()

  persist_active_conversation()

# ============================================================

# ACTIVE CONVERSATION

# ============================================================

# Do not auto-submit a persisted starter selection during Streamlit reruns.

starter_prompt = None

for message in active_conversation["messages"]:

  with st.chat_message(

    message["role"],

    avatar=(

      ":material/auto_awesome:"

      if message["role"] == "assistant"

      else ":material/person:"

    ),

  ):

    if message["content"]:

      if message["role"] == "assistant":

        display_answer_with_sources(

          message["content"],

          message.get("evidence", []),

        )

      else:

        st.markdown(message["content"])

    for filename in message.get("attachments", []):

      st.caption(f":material/attach_file: {filename}")

# ============================================================

# CHAT COMPOSER AND ANSWER GENERATION

# ============================================================

submission = st.chat_input(

  "Message your documents...",

  accept_file="multiple",

  file_type=["pdf"],

  key="chat_composer",

)

question = st.session_state.pop("queued_prompt", None)

user_message_recorded = False

pending_question = st.session_state.get("pending_question")

if pending_question and rag_data:

  question = pending_question

  user_message_recorded = True

attached_files = []

upload_failed = False

if submission is not None:

  if isinstance(submission, str):

    question = submission.strip()

  else:

    question = submission.text.strip()

    attached_files = submission.files

  user_message_recorded = False

if attached_files:

  try:

    existing_names = {

      pdf.name.casefold() for pdf in DOCUMENTS_DIR.glob("*.pdf")

    }

    candidates = []

    skipped_names = []

    for uploaded in attached_files:

      safe_name = Path(uploaded.name).name

      if not safe_name or Path(safe_name).suffix.lower() != ".pdf":

        raise ValueError(f"{uploaded.name} is not a PDF file.")

      payload = uploaded.getvalue()

      if b"%PDF-" not in payload[:1024]:

        raise ValueError(f"{safe_name} is not a readable PDF.")

      reader = PdfReader(BytesIO(payload), strict=False)

      if not reader.pages or not any(

        (page.extract_text() or "").strip()

        for page in reader.pages

      ):

        raise ValueError(

          f"{safe_name} has no extractable PDF text."

        )

      if safe_name.casefold() in existing_names:

        skipped_names.append(safe_name)

        continue

      existing_names.add(safe_name.casefold())

      candidates.append((safe_name, payload))

    for safe_name, payload in candidates:

      (DOCUMENTS_DIR / safe_name).write_bytes(payload)

  except Exception as exc:

    st.error(f"Could not add the selected PDF: {exc}")

    st.exception(exc)

    candidates = []

    skipped_names = []

    upload_failed = True

  if candidates:

    active_conversation["messages"].append({

      "role": "user",

      "content": question or "Please add these PDFs to my library.",

      "attachments": [name for name, _ in candidates],

    })

    active_conversation["updated"] = time()

    if active_conversation["title"] == "New chat" and question:

      active_conversation["title"] = question[:42].strip()

    persist_active_conversation()

    st.session_state.pending_uploads = [

      name for name, _ in candidates

    ]

    if question:

      st.session_state.pending_question = question

    else:

      st.session_state.pop("pending_question", None)

    load_rag.clear()

    st.rerun()

  if skipped_names and not question:

    st.info(

      "Those PDFs are already in your library: "

      + ", ".join(skipped_names)

    )

if upload_failed:

  question = None

if question and not user_message_recorded:

  active_conversation["messages"].append({

    "role": "user",

    "content": question,

  })

  active_conversation["updated"] = time()

  if active_conversation["title"] == "New chat":

    active_conversation["title"] = question[:42].strip()

  persist_active_conversation()

  with st.chat_message("user", avatar=":material/person:"):

    st.markdown(question)

if question:

  if not rag_data:

    answer = (

      "Attach a readable PDF in the message box first, and I can "

      "answer questions using its content."

    )

    evidence_items = []

  else:

    with st.chat_message(

      "assistant",

      avatar=":material/auto_awesome:",

    ):

      try:

        with st.spinner("Searching your library..."):

          page_filter = rag.extract_page_filter(question)

          # Exclude the current user question from conversation context.
          history_messages = active_conversation["messages"]
          if (
            history_messages
            and history_messages[-1].get("role") == "user"
            and history_messages[-1].get("content", "").strip() == question.strip()
          ):
            history_messages = history_messages[:-1]

          conversation_context = build_conversation_context(history_messages)
          explicit_topic_question = find_explicit_topic_question(history_messages)
          search_query = question
          generation_question = question
          query_strategy = "Original question (no rewriting needed)"

          if is_context_dependent_question(question) and explicit_topic_question:
            # Resolve the follow-up for BOTH retrieval and answer generation.
            # Previously only retrieval received the topic context, while the LLM
            # answering "What are its disadvantages?" did not know what "its" meant.
            standalone_question = resolve_followup_question(
              question,
              explicit_topic_question,
            )
            search_query = standalone_question
            generation_question = standalone_question
            query_strategy = "Deterministic topic-anchored follow-up"
          elif is_context_dependent_question(question) and conversation_context:
            # Use the same rewritten query for answering so a successfully
            # resolved pronoun is not lost after retrieval.
            search_query = rewrite_query(
              question=question,
              conversation_context=conversation_context,
              tokenizer=rag_data["tokenizer"],
              model=rag_data["llm"],
            )
            generation_question = search_query
            query_strategy = "LLM rewrite (no explicit topic anchor found)"

          # Always log the query and chosen strategy for debugging.
          print(f"[RAG] Original question: {question}")
          print(f"[RAG] Explicit topic anchor: {explicit_topic_question or '(none)'}")
          print(f"[RAG] Query strategy: {query_strategy}")
          print(f"[RAG] Retrieval query: {search_query}")

          results = rag.search_chunks(
            question=search_query,

            model=rag_data["embedding_model"],

            index=rag_data["index"],

            embeddings=rag_data["embeddings"],

            chunks=rag_data["chunks"],

            bm25=rag_data["bm25"],

            top_k=rag.RETRIEVAL_CANDIDATES,

            page_filter=page_filter,

          )

          # For an explicit follow-up topic, do not let passages about a different
          # concept leak into the answer just because they share words such as
          # "disadvantages". Keep only passages that explicitly mention the topic.
          topic_filter_phrase = ""
          topic_filter_note = "No explicit topic filter was needed."
          if is_context_dependent_question(question) and explicit_topic_question:
            topic_filter_phrase = extract_topic_from_question(explicit_topic_question)
            if topic_filter_phrase:
              topic_pattern = re.compile(re.escape(topic_filter_phrase), re.IGNORECASE)
              original_result_count = len(results)
              results = [
                item for item in results
                if topic_pattern.search(str(item.get("text", "")))
              ]
              topic_filter_note = (
                f"Kept {len(results)} of {original_result_count} retrieved passage(s) "
                f"that explicitly mention '{topic_filter_phrase}'."
              )
              if not results:
                topic_filter_note += " No topic-matching passages were available to answer from."

          # Rerank the hybrid shortlist after applying any topic filter.
          # This scores each query/passage pair together, then keeps the top TOP_K.
          rerank_note = "Cross-encoder reranking was not available; showing hybrid ranking."
          if results and rag_data.get("reranker") is not None:
            before_rerank_count = len(results)
            try:
              results = rag.rerank_chunks(
                question=search_query,
                candidates=results,
                reranker=rag_data["reranker"],
                top_k=rag.TOP_K,
              )
              rerank_note = (
                f"Cross-encoder reranked {before_rerank_count} candidate(s); "
                f"kept the top {len(results)} passage(s)."
              )
            except Exception as exc:
              rerank_note = f"Reranking failed; using hybrid order. Error: {exc}"
              st.warning("Cross-encoder reranking failed; using the hybrid retrieval order.")
          elif rag_data.get("reranker_error"):
            st.warning(
              "Reranker model could not be loaded. Hybrid retrieval is still active. "
              "Check internet/model access. Details: " + rag_data["reranker_error"]
            )

          # Optional diagnostics for checking query rewriting and retrieval.
          # Leave collapsed by default to keep the chat interface uncluttered.
          with st.expander("Retrieval debug (query and matched passages)", expanded=False):
            st.markdown("**Original question**")
            st.write(question)
            st.markdown("**Query strategy**")
            st.write(query_strategy)
            st.markdown("**Explicit topic anchor**")
            st.write(explicit_topic_question or "No explicit topic question found")
            st.markdown("**Topic filter**")
            st.write(topic_filter_note)
            st.markdown("**Query sent to FAISS/BM25**")
            st.write(search_query)
            st.markdown("**Question sent to answer generation**")
            st.write(generation_question)
            if conversation_context:
              st.markdown("**Conversation context used for rewriting**")
              st.text(conversation_context)
            else:
              st.info("No prior conversation context was available; query rewriting was skipped.")
            st.markdown("**Reranking**")
            st.write(rerank_note)
            st.markdown("**Top retrieved passages (after reranking)**")
            if results:
              for result_number, item in enumerate(results[:3], start=1):
                st.markdown(
                  f"**{result_number}. {item.get('source', 'Unknown source')} "
                  f"· page {item.get('page', '?')} · cosine {item.get('score', 'n/a')} · rerank {item.get('rerank_score', 'n/a')}**"
                )
                st.write(str(item.get("text", ""))[:1200])
            else:
              st.warning("The retriever returned no passages.")

          objectives_extracted = False

          if not results:

            answer = (

              "I couldn't find relevant information in the "

              "retrieved PDF passages to answer this question."

            )

          elif (

            page_filter is None

            and max(

              item.get("score", 0.0)

              for item in results

            ) < rag.MIN_RELEVANCE_SCORE

          ):

            answer = (

              "I couldn't find sufficiently relevant "

              "information in the PDFs to answer this question."

            )

          else:

            answer = None

            is_objectives_question = bool(

              re.search(

                r"\b(course\s+)?objectives?\b",

                question,

                flags=re.IGNORECASE,

              )

            )

            if is_objectives_question:

              # Extract only an explicit objectives list from PDF text.

              # Never use the LLM to invent generic course objectives.

              answer = rag.extract_course_objectives(

                rag_data["pages"],

                page_filter,

              )

              objectives_extracted = answer is not None

              if answer is None:

                answer = (

                  "I couldn't find a clearly stated course "

                  "objectives list in the retrieved PDF text."

                )

            else:

              answer = rag.generate_answer(

                generation_question,

                results,

                rag_data["tokenizer"],

                rag_data["llm"],

              )

          if objectives_extracted:

            objective_source = re.search(

              r"\n\nSource: (.+?)\s*\|\s*Page: (\d+)$",

              answer,

            )

            evidence_items = [

              {

                "source": page["source"],

                "page": page["page"],

                "text": page["text"],

              }

              for page in rag_data["pages"]

              if objective_source

              and page["source"] == objective_source.group(1)

              and page["page"] == int(objective_source.group(2))

            ]

          else:

            evidence_items = [

              {

                "source": item["source"],

                "page": item["page"],

                "text": item["text"],

              }

              for item in results

            ]

        uploaded_names = st.session_state.pop(

          "pending_uploads", []

        )

        if uploaded_names:

          answer = (

            f"Added {', '.join(uploaded_names)} to your "

            f"library.\n\n{answer}"

          )

        display_answer_with_sources(answer, evidence_items)

        active_conversation["messages"].append({

          "role": "assistant",

          "content": answer,

          "evidence": evidence_items,

        })

        active_conversation["updated"] = time()

        persist_active_conversation()

      except Exception as exc:

        st.error(f"Question processing failed: {exc}")

        st.exception(exc)

        answer = (

          "I couldn't complete the answer because question "

          f"processing failed: {exc}"

        )

        st.markdown(answer)

        active_conversation["messages"].append({

          "role": "assistant",

          "content": answer,

          "evidence": [],

        })

        active_conversation["updated"] = time()

        persist_active_conversation()

  if not rag_data:

    with st.chat_message(

      "assistant",

      avatar=":material/auto_awesome:",

    ):

      st.markdown(answer)

    active_conversation["messages"].append({

      "role": "assistant",

      "content": answer,

      "evidence": evidence_items,

    })

    active_conversation["updated"] = time()

    persist_active_conversation()

  st.session_state.pop("pending_question", None)

  if answer is not None:

    st.rerun()
