"""Ingestion pipeline: load PDFs, chunk text, embed, and store in Chroma."""
from pathlib import Path

import pymupdf
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import config

_embedding_model = None
_tokenizer = tiktoken.get_encoding("cl100k_base")


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model


def _token_length(text: str) -> int:
    return len(_tokenizer.encode(text))


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
    )


def load_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) tuples, pages 1-indexed."""
    doc = pymupdf.open(pdf_path)
    try:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()


def chunk_pdf_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Chunk each page's text, keeping the originating page number.

    Chunking is done per page so every chunk can be attributed to a single
    page number in its metadata. Returns a list of {"text": str, "page": int}.
    """
    splitter = get_text_splitter()
    chunks = []
    for page_num, page_text in pages:
        if not page_text.strip():
            continue
        for chunk_text in splitter.split_text(page_text):
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "page": page_num})
    return chunks


def ingest_pdf(pdf_path: str, collection=None) -> int:
    """Ingest a single PDF file into the Chroma collection. Returns chunk count."""
    collection = collection or config.get_collection()
    filename = Path(pdf_path).name

    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pdf_pages(pages)
    if not chunks:
        return 0

    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [f"{filename}::p{c['page']}::{i}" for i, c in enumerate(chunks)]
    metadatas = [{"filename": filename, "page": c["page"]} for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def ingest_pdfs(pdf_paths: list[str], collection=None) -> dict[str, int]:
    """Ingest multiple PDFs. Returns a dict of filename -> number of chunks added."""
    collection = collection or config.get_collection()
    return {Path(p).name: ingest_pdf(p, collection=collection) for p in pdf_paths}
