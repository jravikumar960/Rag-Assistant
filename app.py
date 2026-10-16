

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



    return {

        "pages": pages,

        "chunks": chunks,

        "embedding_model": embedding_model,

        "embeddings": embeddings,

        "index": index,

        "bm25": bm25,

        "tokenizer": tokenizer,

        "llm": llm,

        "cache_hit": cache_hit,

    }





# ============================================================

# COMPACT SOURCE CITATIONS

# ============================================================



def display_answer_with_sources(answer, evidence_items):
    clean_answer = answer.strip()
    clean_answer = re.split(
        r"\n\s*\[?Supporting Evidence:?\]?\s*\n",
        clean_answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    source_match = re.search(
        r"\n{1,2}Sources?:\s*(.+)$",
        clean_answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    evidence = None
    if source_match:
        source_text = source_match.group(1).strip()
        clean_answer = clean_answer[:source_match.start()].strip()
        evidence_id = re.search(r"\[Evidence\s+(\d+)\]", source_text, re.I)
        if evidence_id:
            index = int(evidence_id.group(1)) - 1
            if 0 <= index < len(evidence_items):
                evidence = evidence_items[index]
        else:
            source_page = re.search(
                r"(.+?),\s*page\s*(\d+)",
                source_text,
                flags=re.IGNORECASE,
            )
            if source_page:
                evidence = next(
                    (
                        item for item in evidence_items
                        if item.get("source") == source_page.group(1).strip()
                        and item.get("page") == int(source_page.group(2))
                    ),
                    None,
                )

    st.markdown(clean_answer)

    if evidence is None and evidence_items:
        evidence = evidence_items[0]
    if evidence is None:
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



                    results = rag.search_chunks(

                        question=question,

                        model=rag_data["embedding_model"],

                        index=rag_data["index"],

                        embeddings=rag_data["embeddings"],

                        chunks=rag_data["chunks"],

                        bm25=rag_data["bm25"],

                        top_k=rag.TOP_K,

                        page_filter=page_filter,

                    )

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
                                question,
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
