
import hashlib
import re
from pathlib import Path

import streamlit as st

from rag.pipeline import answer_question, ingest_document
from rag.vector_store import VectorStore


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

UPLOAD_DIR = Path("data/uploads")
VECTOR_DB_DIR = "chroma_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# SHARED VECTOR STORE
# --------------------------------------------------

@st.cache_resource
def get_vector_store():
    return VectorStore(persist_directory=VECTOR_DB_DIR)


vector_store = get_vector_store()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Researcher RAG AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Researcher RAG AI")
st.caption(
    "Upload your documents and ask questions grounded in their content."
)


# --------------------------------------------------
# SIDEBAR: DOCUMENT UPLOAD
# --------------------------------------------------

with st.sidebar:
    st.header("Document Library")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_bytes = uploaded_file.getvalue()

                # Identify the exact uploaded file contents.
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                safe_name = re.sub(
                    r"[^A-Za-z0-9._-]+",
                    "_",
                    Path(uploaded_file.name).name,
                )

                upload_path = (
                    UPLOAD_DIR / f"{file_hash[:16]}_{safe_name}"
                ).resolve()

                document_key = str(upload_path)

                if document_key in st.session_state.processed_uploads:
                    continue

                # Save the file so it remains available after reruns.
                if not upload_path.exists():
                    upload_path.write_bytes(file_bytes)

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):
                    result = ingest_document(
                        file_path=str(upload_path),
                        vector_store=vector_store,
                    )

                document_id = result["document_id"]

                st.session_state.documents[document_id] = {
                    "name": uploaded_file.name,
                    "chunk_count": result["chunk_count"],
                }

                st.session_state.processed_uploads.add(document_key)

                st.success(
                    f"Processed {uploaded_file.name}: "
                    f"{result['chunk_count']} chunks"
                )

            except Exception as exc:
                st.error(
                    f"Could not process {uploaded_file.name}: {exc}"
                )

    st.divider()

    st.subheader("Uploaded Documents")

    if st.session_state.documents:
        for document in st.session_state.documents.values():
            st.write(f"📄 {document['name']}")
            st.caption(f"{document['chunk_count']} chunks")
    else:
        st.info("Upload a document to get started.")


# --------------------------------------------------
# RETRIEVAL SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.divider()
    st.subheader("Retrieval Settings")

    strategy = st.selectbox(
        "Retrieval strategy",
        options=["similarity", "multiquery"],
        format_func=lambda value: {
            "similarity": "Similarity Search",
            "multiquery": "Multi-Query Search",
        }[value],
        help=(
            "Similarity searches using your original question. "
            "Multi-Query generates alternative questions using Gemini."
        ),
    )

    document_options = ["All uploaded documents"]
    document_ids = list(st.session_state.documents.keys())

    document_options.extend(document_ids)

    selected_document = st.selectbox(
        "Search scope",
        options=document_options,
        format_func=lambda value: (
            value
            if value == "All uploaded documents"
            else st.session_state.documents.get(
                value, {}
            ).get("name", value)
        ),
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
    )


# --------------------------------------------------
# DISPLAY PREVIOUS CHAT MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for index, chunk in enumerate(
                    message["sources"], start=1
                ):
                    metadata = chunk.get("metadata", {})

                    source = metadata.get("source", "Unknown source")
                    page = metadata.get("page")

                    st.markdown(f"**Source {index}: {Path(str(source)).name}**")

                    if page is not None:
                        st.caption(f"Page: {page}")

                    st.write(chunk.get("text", ""))


# --------------------------------------------------
# QUESTION-ANSWERING INTERFACE
# --------------------------------------------------

if not st.session_state.documents:
    st.info(
        "Upload a PDF, TXT, or DOCX document in the sidebar "
        "before asking a question."
    )

else:
    prompt = st.chat_input(
        "Ask a question about your documents..."
    )

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        document_id = (
            None
            if selected_document == "All uploaded documents"
            else selected_document
        )

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating answer..."):
                    result = answer_question(
                        query=prompt,
                        vector_store=vector_store,
                        top_k=top_k,
                        document_id=document_id,
                        strategy=strategy,
                    )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("View retrieved sources"):
                        for index, chunk in enumerate(
                            sources, start=1
                        ):
                            metadata = chunk.get("metadata", {})
                            source = metadata.get(
                                "source", "Unknown source"
                            )
                            page = metadata.get("page")

                            st.markdown(
                                f"**Source {index}: "
                                f"{Path(str(source)).name}**"
                            )

                            if page is not None:
                                st.caption(f"Page: {page}")

                            st.write(chunk.get("text", ""))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                error_message = (
                    f"Sorry, I couldn't answer your question: {exc}"
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
