import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"


st.title("RAG Assistant")
st.caption("Streamlit frontend connected to FastAPI")

# Check whether the backend is ready.
try:
    health = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    health.raise_for_status()
    status = health.json()

    if status.get("rag_ready"):
        st.success("RAG backend is ready")
    else:
        st.warning("Backend is running, but RAG is not ready")

except requests.RequestException:
    st.error("Cannot connect to FastAPI. Start the backend first.")
    st.stop()


# Keep the same conversation session across requests.
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about your PDFs")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    payload = {"question": question}

    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id

    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

                st.session_state.session_id = result["session_id"]

                answer = result["answer"]
                st.markdown(answer)

                with st.expander("Sources"):
                    for source in result.get("sources", []):
                        st.markdown(
                            f"**{source['filename']} — "
                            f"page {source['page']}**"
                        )
                        st.caption(source["excerpt"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")