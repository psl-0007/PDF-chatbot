"""Shared configuration and Chroma access for the ingestion and retrieval pipelines."""
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_setting(key: str, default: str = None) -> str:
    """Read a setting from the environment (.env locally) or Streamlit secrets
    (st.secrets, used on Streamlit Community Cloud), in that order."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "pdf_chunks"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

TOP_K = 5
MAX_AGGREGATE_CHUNKS = 80  # cap for whole-document questions (counts, "list all", etc.)

GROQ_API_KEY = _get_setting("GROQ_API_KEY")
GROQ_MODEL = _get_setting("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

@st.cache_resource
def get_chroma_client() -> chromadb.ClientAPI:
    """Cached by Streamlit so concurrent script reruns share one client instead
    of racing to construct it (a bare module-global was not thread-safe and
    could trigger a KeyError inside chromadb's internal system-client cache)."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


@st.cache_resource
def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
