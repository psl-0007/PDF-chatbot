"""Retrieval pipeline: embed the query, fetch top chunks, build a grounded prompt, call the LLM."""
import re
from dataclasses import dataclass

from openai import OpenAI

import config
from ingestion import get_embedding_model


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int
    distance: float


def retrieve_chunks(query: str, collection=None, top_k: int = None) -> list[RetrievedChunk]:
    collection = collection or config.get_collection()
    top_k = top_k or config.TOP_K

    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        RetrievedChunk(text=doc, filename=meta["filename"], page=meta["page"], distance=dist)
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


_AGGREGATE_RE = re.compile(
    r"\bhow many\b|\ball\b|\bevery\b|\beach\b|\btotal\b|\bcount\b|\blist\b|\bsummar\w*\b",
    re.IGNORECASE,
)


def is_aggregate_query(query: str) -> bool:
    """Whole-document questions (counts, totals, "list all") need more than top-k
    semantic matches, since the relevant pieces may not be the most similar ones."""
    return bool(_AGGREGATE_RE.search(query))


def retrieve_all_chunks(collection=None, limit: int = None) -> list[RetrievedChunk]:
    """Fetch every chunk in the collection, in original document order."""
    collection = collection or config.get_collection()
    limit = limit or config.MAX_AGGREGATE_CHUNKS
    result = collection.get(include=["documents", "metadatas"], limit=limit)

    def chunk_index(chunk_id: str) -> int:
        suffix = chunk_id.rsplit("::", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    rows = list(zip(result["ids"], result["metadatas"], result["documents"]))
    rows.sort(key=lambda row: (row[1]["filename"], row[1]["page"], chunk_index(row[0])))

    return [
        RetrievedChunk(text=doc, filename=meta["filename"], page=meta["page"], distance=0.0)
        for _id, meta, doc in rows
    ]


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = [
        f"[Source {i} | {c.filename}, page {c.page}]\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)

    return (
        "You are a helpful assistant that answers questions using ONLY the "
        "context provided below. If the answer is not contained in the "
        "context, say you don't know based on the provided documents. Do "
        "not use outside knowledge.\n\n"
        "Formatting rules: write in plain Markdown. For any math, wrap "
        "inline expressions in single dollar signs, e.g. $x = 5$, and "
        "standalone equations in double dollar signs, e.g. $$x + 2 = 7$$. "
        "Never use raw LaTeX delimiters like [ ] or \\[ \\], and never use "
        "\\qquad, \\Longrightarrow, or \\tag — instead connect steps with "
        "plain words like 'so' or 'therefore', each on its own line.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def call_llm(prompt: str) -> str:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def answer_query(query: str, collection=None, top_k: int = None) -> dict:
    """Run the full retrieval + generation pipeline for a query.

    Returns {"answer": str, "sources": list[RetrievedChunk]}.
    """
    collection = collection or config.get_collection()
    if is_aggregate_query(query):
        chunks = retrieve_all_chunks(collection=collection)
    else:
        chunks = retrieve_chunks(query, collection=collection, top_k=top_k)
    if not chunks:
        return {
            "answer": "No documents have been ingested yet, so I have no context to answer from.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)
    answer = call_llm(prompt)
    return {"answer": answer, "sources": chunks}
