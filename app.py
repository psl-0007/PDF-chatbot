"""Streamlit UI for the RAG PDF chatbot."""
import tempfile
from pathlib import Path

import streamlit as st

import config
import ingestion
import retrieval

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

collection = config.get_collection()

with st.sidebar:
    st.header("Ingest PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )
    if st.button("Ingest", disabled=not uploaded_files):
        with st.spinner("Ingesting PDFs..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_paths = []
                for uploaded_file in uploaded_files:
                    tmp_path = Path(tmp_dir) / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.getvalue())
                    tmp_paths.append(str(tmp_path))
                results = ingestion.ingest_pdfs(tmp_paths, collection=collection)
            for filename, num_chunks in results.items():
                st.success(f"{filename}: {num_chunks} chunks added")

    st.divider()
    st.header("Ingested documents")
    try:
        filenames = sorted({m["filename"] for m in collection.get(include=["metadatas"])["metadatas"]})
    except Exception:
        filenames = []

    if not filenames:
        st.caption("No documents ingested yet.")
    for filename in filenames:
        file_col, remove_col = st.columns([4, 1])
        file_col.write(filename)
        if remove_col.button("✕", key=f"remove_{filename}", help=f"Remove {filename}"):
            collection.delete(where={"filename": filename})
            st.rerun()

    st.divider()
    try:
        count = collection.count()
    except Exception:
        count = 0
    st.caption(f"{count} chunks currently indexed")

st.title("📄 RAG PDF Chatbot")
st.caption(
    "Ask questions about the PDFs you've ingested. Answers are grounded only in your documents."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = None
        with st.spinner("Thinking..."):
            try:
                result = retrieval.answer_query(query, collection=collection)
            except RuntimeError as e:
                st.error(str(e))

        if result:
            st.markdown(result["answer"])
            if result["sources"]:
                unique_sources = sorted({(c.filename, c.page) for c in result["sources"]})
                sources_str = ", ".join(f"{fn} (p. {pg})" for fn, pg in unique_sources)
                st.caption(f"📎 Sources: {sources_str}")
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
