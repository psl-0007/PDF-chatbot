"""Streamlit UI for the RAG PDF chatbot ("Editorial" theme)."""
import html
import re
import tempfile
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
import ingestion
import retrieval

STYLE_PATH = Path(__file__).parent / "assets" / "style.css"

STARTER_QUESTIONS = [
    "Give me a short summary of these documents",
    "What are the key points?",
    "List the main topics covered",
]

MAX_FOOTNOTES = 8   # footnote lines shown under an answer
MAX_PASSAGES = 3    # retrieved passages shown in the expander
PASSAGE_CHARS = 600

# Same four brand colours in both modes; only the roles they play change.
THEMES = {
    "light": {
        "bg": "#FAF6EC", "surface": "#F2ECDB", "hair": "rgba(27,42,73,.18)",
        "text": "#1B2A49", "muted": "#566180",
        "accent": "#B3202F", "accent2": "#55622D",
        "side-bg": "#F2ECDB", "side-text": "#1B2A49", "side-muted": "#566180",
        "side-line": "rgba(27,42,73,.2)",
    },
    "dark": {  # neutral near-black (not a tinted navy); ivory ink, lightened crimson / olive accents
        "bg": "#0A0A0A", "surface": "#151515", "hair": "rgba(237,231,216,.15)",
        "text": "#EDE7D8", "muted": "#A9A69C",
        "accent": "#E5707B", "accent2": "#B5C27A",
        "side-bg": "#000000", "side-text": "#EDE7D8", "side-muted": "#8E8B82",
        "side-line": "rgba(237,231,216,.15)",
    },
}
FIXED_COLOURS = {"olive": "#6B7A3A", "olive-dark": "#55622D", "crimson": "#B3202F"}

# One delegated listener (guarded so reruns don't stack duplicates) powers every "Copy" button.
COPY_SCRIPT = """
<script>
if (!window.__copyInit) {
  window.__copyInit = true;
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('.copy-btn');
    if (!btn) return;
    navigator.clipboard.writeText(btn.dataset.text).then(function () {
      var old = btn.textContent;
      btn.textContent = 'Copied'; btn.classList.add('done');
      setTimeout(function () { btn.textContent = old; btn.classList.remove('done'); }, 1500);
    });
  });
}
</script>
"""

st.set_page_config(page_title="PDF Chatbot", page_icon="📖", layout="wide")

# Dark mode is remembered in the URL (?theme=dark) so it survives a refresh and can be bookmarked.
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = st.query_params.get("theme") == "dark"


def remember_theme() -> None:
    if st.session_state.dark_mode:
        st.query_params["theme"] = "dark"
    elif "theme" in st.query_params:
        del st.query_params["theme"]


# Widget state is already populated at the top of a rerun, so the toggle
# (rendered later, in the sidebar) controls the theme of this same run.
mode = "dark" if st.session_state.get("dark_mode", False) else "light"
palette = {**THEMES[mode], **FIXED_COLOURS}
css_vars = "".join(f"--{name}:{value};" for name, value in palette.items())
no_grain = (
    '.stApp,[data-testid="stBottom"],[data-testid="stBottom"]>div{background-image:none !important}'
    if mode == "dark" else ""
)
st.markdown(
    f"<style>:root{{{css_vars}}}{STYLE_PATH.read_text(encoding='utf-8')}{no_grain}</style>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

collection = config.get_collection()

# Session history lives in this browser session only (st.session_state). It is deliberately
# not written to disk: on a shared deployment a file would mix different visitors' chats.
if "sessions" not in st.session_state:
    st.session_state.sessions = {}      # id -> {"created": datetime, "messages": [...]}
    st.session_state.current_id = uuid.uuid4().hex[:8]


# ---------- data helpers ----------

def collect_sources(chunks) -> list[dict]:
    """One entry per (file, page), best-matching passage kept.

    Whole-document questions retrieve every chunk with distance 0, so there is no
    meaningful relevance score there; those entries get score=None.
    """
    aggregate = all(c.distance == 0.0 for c in chunks)
    best = {}
    for c in chunks:
        key = (c.filename, c.page)
        if key not in best or c.distance < best[key].distance:
            best[key] = c
    if aggregate:
        ordered = sorted(best.values(), key=lambda c: (c.filename, c.page))
    else:
        ordered = sorted(best.values(), key=lambda c: (c.distance, c.filename, c.page))
    return [
        {
            "filename": c.filename,
            "page": c.page,
            "text": c.text.strip()[:PASSAGE_CHARS],
            "score": None if aggregate else max(0, round((1 - c.distance) * 100)),
        }
        for c in ordered
    ]


def get_document_counts() -> Counter:
    """Chunks per ingested filename."""
    try:
        metadatas = collection.get(include=["metadatas"])["metadatas"]
    except Exception:
        return Counter()
    return Counter(m["filename"] for m in metadatas)


def conversation_markdown() -> str:
    lines = ["# PDF Chatbot conversation", ""]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines += [f"## {m['content']}", ""]
        else:
            lines += [m["content"], ""]
            if m.get("sources"):
                lines.append("**Sources**")
                lines += [f"- {s['filename']} (p. {s['page']})" for s in m["sources"][:MAX_FOOTNOTES]]
                lines.append("")
    return "\n".join(lines)


def sync_current_session() -> None:
    """Store the live conversation under its id so it shows up in the history list."""
    ss = st.session_state
    if ss.messages:
        prev = ss.sessions.get(ss.current_id)
        ss.sessions[ss.current_id] = {
            "created": prev["created"] if prev else datetime.now(),
            "messages": list(ss.messages),
        }


def start_new_session() -> None:
    sync_current_session()
    st.session_state.current_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []


def open_session(session_id: str) -> None:
    sync_current_session()
    st.session_state.current_id = session_id
    st.session_state.messages = list(st.session_state.sessions[session_id]["messages"])


def delete_session(session_id: str) -> None:
    st.session_state.sessions.pop(session_id, None)
    if session_id == st.session_state.current_id:
        st.session_state.current_id = uuid.uuid4().hex[:8]
        st.session_state.messages = []


def session_title(messages: list[dict], limit: int = 56) -> str:
    first = next((m["content"] for m in messages if m["role"] == "user"), "Untitled")
    first = " ".join(first.split())
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


MD_SPECIAL = set("\\`*_{}[]()#+-.!$<>~|:")


def md_escape(text: str) -> str:
    """Button labels are rendered as markdown; escape so titles show literally."""
    return "".join(("\\" + c) if c in MD_SPECIAL else c for c in text)


# ---------- rendering ----------

def render_masthead(doc_counts: Counter) -> None:
    today = datetime.now().strftime("%A, %d %B %Y")
    issue = sum(1 for m in st.session_state.messages if m["role"] == "user") + 1
    docs = len(doc_counts)
    passages = sum(doc_counts.values())
    st.markdown(
        f'<div class="masthead">'
        f'<div class="dateline"><span>{today}</span><span class="issue">Vol. I &middot; No. {issue}</span>'
        f"<span>{docs} document{'s' if docs != 1 else ''} &middot; {passages} passages indexed</span></div>"
        f'<div class="mast-title">PDF <em>Chatbot</em></div>'
        '<div class="ornament"><span class="rule"></span><span class="fleuron">&#10086;&#xFE0E;</span><span class="rule"></span></div>'
        f'<div class="mast-sub">Ask questions of your documents. Every answer is drawn only '
        f"from what you've uploaded, and cited.</div></div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[Counter, "st.delta_generator.DeltaGenerator"]:
    with st.sidebar:
        st.header("Library")
        uploaded_files = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
        )
        if st.button("Ingest", key="ingest", disabled=not uploaded_files):
            with st.spinner("Reading and indexing your PDFs..."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_paths = []
                    for uploaded_file in uploaded_files:
                        tmp_path = Path(tmp_dir) / uploaded_file.name
                        tmp_path.write_bytes(uploaded_file.getvalue())
                        tmp_paths.append(str(tmp_path))
                    results = ingestion.ingest_pdfs(tmp_paths, collection=collection)
                for filename, num_chunks in results.items():
                    st.success(f"{filename}: {num_chunks} chunks added")

        st.header("Contents")
        doc_counts = get_document_counts()
        if not doc_counts:
            st.caption("No documents ingested yet.")
        else:
            st.markdown('<div class="toc-head"><span>Document</span><span>Passages</span></div>', unsafe_allow_html=True)
        for i, filename in enumerate(sorted(doc_counts), start=1):
            row_col, remove_col = st.columns([8, 1], vertical_alignment="center")
            row_col.markdown(
                f'<div class="doc-row"><span class="num">{i:02d}</span>'
                f'<span class="doc-name">{html.escape(filename)}</span>'
                f'<span class="leader"></span><span class="count">{doc_counts[filename]}</span></div>',
                unsafe_allow_html=True,
            )
            if remove_col.button(":material/close:", key=f"remove_{filename}", help=f"Remove {filename}"):
                collection.delete(where={"filename": filename})
                st.rerun()

        st.header("History")
        # Filled at the end of the run (render_history_panel), once this turn's answer exists;
        # otherwise the buttons below would reflect the conversation as it was before the answer.
        history_slot = st.container()

        st.header("Appearance")
        st.toggle("Dark mode", key="dark_mode", on_change=remember_theme)
    return doc_counts, history_slot


def render_history_panel(slot) -> None:
    """New / export controls plus past conversations as a quiet list: title, time and
    question count; the conversation that is open is ruled in crimson."""
    sync_current_session()
    sessions = st.session_state.sessions
    has_messages = bool(st.session_state.messages)
    with slot:
        st.button("New conversation", key="new_chat", on_click=start_new_session, disabled=not has_messages)
        if not sessions:
            st.caption("Past conversations will appear here.")
        for sid, sess in sorted(sessions.items(), key=lambda kv: kv[1]["created"], reverse=True):
            is_current = sid == st.session_state.current_id
            asked = sum(1 for m in sess["messages"] if m["role"] == "user")
            row, delete = st.columns([8, 1], vertical_alignment="center")
            with row:
                st.button(
                    md_escape(session_title(sess["messages"])),
                    key=f"sess_{'cur_' if is_current else ''}{sid}",
                    on_click=open_session, args=(sid,),
                )
                st.markdown(
                    f'<div class="sess-meta{" is-current" if is_current else ""}">'
                    f'{sess["created"].strftime("%H:%M")} &middot; {asked} question{"s" if asked != 1 else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            delete.button(
                ":material/close:", key=f"remove_session_{sid}", on_click=delete_session, args=(sid,),
                help="Delete this conversation",
            )
        st.download_button(
            "Export this conversation",
            data=conversation_markdown(),
            file_name="conversation.md",
            mime="text/markdown",
            disabled=not has_messages,
        )


def render_sources(sources: list[dict]) -> None:
    """Footnote-style citations, plus the passages the answer was drawn from."""
    if not sources:
        return
    shown = sources[:MAX_FOOTNOTES]
    notes = "".join(
        f'<div class="note"><span class="n">{i}</span>'
        f'<span class="f">{html.escape(s["filename"])}</span>'
        f'<span class="pg">p. {s["page"]}</span></div>'
        for i, s in enumerate(shown, start=1)
    )
    if len(sources) > len(shown):
        notes += f'<div class="note"><span class="n"></span><span class="more">and {len(sources) - len(shown)} more pages</span></div>'
    st.markdown(f'<div class="notes"><div class="notes-h">Sources</div>{notes}</div>', unsafe_allow_html=True)

    scored = [(i, s) for i, s in enumerate(sources, start=1) if s["score"] is not None][:MAX_PASSAGES]
    if scored:
        with st.expander(f"Read the passage{'s' if len(scored) != 1 else ''}"):
            for i, s in scored:
                st.markdown(
                    f'<div class="passage"><div class="q">&ldquo;{html.escape(s["text"])}&rdquo;</div>'
                    f'<div class="meta">{i} &middot; {html.escape(s["filename"])} &middot; p. {s["page"]} '
                    f'&middot; {s["score"]}% match</div></div>',
                    unsafe_allow_html=True,
                )


def render_copy_button(text: str) -> None:
    st.html(
        f'<div class="actions"><button class="copy-btn" type="button" '
        f'data-text="{html.escape(text, quote=True)}">Copy answer</button></div>',
        unsafe_allow_javascript=True,
    )


def render_empty_state(has_documents: bool) -> str | None:
    """Welcome screen. Returns a starter question if the user clicked one."""
    if not has_documents:
        st.markdown(
            '<div class="welcome"><h2>Ask your documents <em>anything.</em></h2>'
            "<p>Three steps to your first answer.</p>"
            '<ol class="steps">'
            "<li><span><b>Upload</b> one or more PDFs in the library on the left.</span></li>"
            "<li><span><b>Ingest</b> them so they can be searched.</span></li>"
            "<li><span><b>Ask</b> a question below and read the cited answer.</span></li>"
            "</ol></div>",
            unsafe_allow_html=True,
        )
        return None

    st.markdown(
        '<div class="welcome"><h2>What would you like <em>to know?</em></h2>'
        "<p>Type a question below, or begin with one of these.</p></div>"
        '<div class="inquiry-h">Lines of inquiry</div>',
        unsafe_allow_html=True,
    )
    for i, question in enumerate(STARTER_QUESTIONS):
        if st.button(question, key=f"starter_{i}"):
            return question
    return None


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                render_copy_button(message["content"])


def answer(query: str) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = None
        with st.spinner("Searching your documents..."):
            try:
                result = retrieval.answer_query(query, collection=collection)
            except RuntimeError as e:
                st.error(str(e))

        if result:
            st.markdown(result["answer"])
            sources = collect_sources(result["sources"]) if result["sources"] else []
            render_sources(sources)
            render_copy_button(result["answer"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": sources}
            )


# ---------- page ----------

doc_counts, history_slot = render_sidebar()
render_masthead(doc_counts)
st.html(COPY_SCRIPT, unsafe_allow_javascript=True)

starter = None
empty_slot = st.empty()
if st.session_state.messages:
    render_history()
else:
    with empty_slot.container():
        starter = render_empty_state(has_documents=bool(doc_counts))

query = st.chat_input("Ask a question about your PDFs...") or starter
if query:
    empty_slot.empty()  # clear the welcome screen while the answer is generated
    answer(query)

render_history_panel(history_slot)
