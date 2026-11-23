# RAG PDF Chatbot

Chat with your PDFs. Upload documents, they get chunked and embedded into a
local vector store, and questions are answered strictly from retrieved
context, with source page numbers.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
streamlit run app.py
```

## Architecture

- `config.py` — shared settings and the persistent Chroma collection.
- `ingestion.py` — loads PDFs with PyMuPDF, recursively chunks text (500
  tokens, 50 token overlap), embeds with `all-MiniLM-L6-v2`, and upserts into
  Chroma with `filename`/`page` metadata.
- `retrieval.py` — embeds the query with the same model, fetches the top-5
  most similar chunks, builds a context-only prompt, and calls Groq.
- `app.py` — Streamlit chat UI: upload/ingest PDFs in the sidebar, chat in
  the main pane, with source pages shown under each answer.

Chroma data persists in `chroma_db/` between runs.
