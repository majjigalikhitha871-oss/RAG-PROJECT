import hashlib
import logging

import numpy as np
import streamlit as st

from core.chunker import Chunk, create_chunks
from core.document_loader import extract_document
from core.embeddings import embed_texts
from core.ollama_client import (
    DEFAULT_OLLAMA_URL,
    OllamaError,
    detect_ollama_models,
    generate_answer,
    resolve_model,
)
from core.rag_pipeline import answer_question
from ui.styles import APP_CSS


st.set_page_config(page_title="Fieldnotes | Document Intelligence", page_icon="🌱", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

logger = logging.getLogger(__name__)


@st.cache_data(ttl=10, show_spinner=False)
def get_model_status(base_url: str) -> tuple[list[str], str | None]:
    models = detect_ollama_models(base_url)
    selection = resolve_model(models)
    return models, selection.name if selection else None


def index_signature(files, chunk_size: int, overlap: int) -> str:
    digest = hashlib.sha256()
    for uploaded in files:
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(uploaded.getvalue())
    digest.update(f"{chunk_size}:{overlap}".encode("ascii"))
    return digest.hexdigest()


def initialize_state() -> None:
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("vectors", np.empty((0, 0), dtype=np.float32))
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("index_signature", None)
    st.session_state.setdefault("document_count", 0)
    st.session_state.setdefault("character_count", 0)
    st.session_state.setdefault("page_count", 0)


initialize_state()

st.title("Fieldnotes")
st.markdown('<div class="brand-rule"></div>', unsafe_allow_html=True)
st.markdown('<p class="status-line">Chat with your documents. Locally indexed, locally answered.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Workspace")
    ollama_url = st.text_input("Ollama address", value=DEFAULT_OLLAMA_URL)
    try:
        installed_models, selected_model = get_model_status(ollama_url)
        if selected_model:
            family = resolve_model(installed_models).family
            st.success(f"Ollama connected · {family}: {selected_model}")
        else:
            st.warning("Ollama connected · no Qwen or Llama model found")
            st.caption("Install one model using the commands in README.md.")
    except OllamaError as exc:
        installed_models, selected_model = [], None
        st.error("Ollama not connected")
        st.caption(str(exc))

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=8, value=3)
    min_similarity = st.slider("Minimum cosine similarity", 0.0, 0.8, 0.2, 0.05)
    chunk_size = st.slider("Chunk size (words)", min_value=60, max_value=500, value=180, step=20)
    overlap = st.slider(
        "Chunk overlap (words)", min_value=0, max_value=min(100, chunk_size - 1), value=min(35, chunk_size - 1), step=5
    )
    temperature = st.slider("Answer temperature", 0.0, 1.0, 0.2, 0.1)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []

    with st.expander("How retrieval works"):
        st.write("Embeddings represent meaning as vectors. Cosine similarity compares their direction; the closest chunks are supplied to the local LLM as context.")
        st.latex(r"\text{cosine}(q,d)=\frac{q\cdot d}{\|q\|\|d\|}")
        st.caption("Uploaded text is treated as untrusted reference material, never as instructions.")

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.subheader("1 · Add source documents")
    uploads = st.file_uploader(
        "PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True,
        help="Files stay in this app session and are not written to disk.",
    )
    if uploads:
        for uploaded in uploads:
            st.caption(f"{uploaded.name} · {uploaded.size / 1024:.1f} KB")
    build_clicked = st.button("Build knowledge base", type="primary", disabled=not uploads)
    current_signature = index_signature(uploads, chunk_size, overlap) if uploads else None
    index_is_current = bool(
        st.session_state.chunks
        and current_signature
        and current_signature == st.session_state.index_signature
    )
    if st.session_state.chunks and current_signature and not index_is_current:
        st.warning("The selected files or chunk settings differ from the index. Rebuild the knowledge base before chatting.")

    if build_clicked and uploads:
        if current_signature == st.session_state.index_signature:
            st.info("These documents and chunk settings are already indexed.")
        else:
            new_chunks: list[Chunk] = []
            characters = 0
            page_count = 0
            try:
                with st.spinner("Extracting, chunking, and embedding documents on this device..."):
                    for uploaded in uploads:
                        pages = extract_document(uploaded.getvalue(), uploaded.name)
                        page_count += sum(page.page_number is not None for page in pages)
                        characters += sum(len(page.text) for page in pages)
                        new_chunks.extend(create_chunks(pages, uploaded.name, chunk_size, overlap))
                    if not new_chunks:
                        raise ValueError("No text was available to index in the selected documents.")
                    new_vectors = embed_texts([chunk.text for chunk in new_chunks])
                st.session_state.chunks = new_chunks
                st.session_state.vectors = new_vectors
                st.session_state.index_signature = current_signature
                st.session_state.document_count = len(uploads)
                st.session_state.character_count = characters
                st.session_state.page_count = page_count
                st.session_state.history = []
                st.success(f"Indexed {len(uploads)} document(s) into {len(new_chunks)} chunks.")
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("Unexpected error while building the knowledge base")
                st.error("The knowledge base could not be built. Check the file and local embedding setup, then try again.")

with right:
    st.subheader("Knowledge base")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Documents", st.session_state.document_count)
    metric_columns[1].metric("Chunks", len(st.session_state.chunks))
    metric_columns[2].metric("Characters", f"{st.session_state.character_count:,}")
    if st.session_state.index_signature:
        page_label = f" · {st.session_state.page_count} PDF pages" if st.session_state.page_count else ""
        st.caption(f"Indexed and ready{page_label}. Index is held in memory for this session.")
    else:
        st.caption("No documents indexed yet.")

st.divider()
st.subheader("2 · Ask your documents")
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Retrieved sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    chunk = source.chunk
                    page = f" · page {chunk.page_number}" if chunk.page_number else ""
                    st.markdown(f"**{chunk.chunk_id}** · {chunk.source}{page} · cosine {source.score:.3f}")
                    st.write(chunk.text)

question = st.chat_input("Ask a question about the indexed documents", disabled=not index_is_current)
if question:
    st.session_state.history.append({"role": "user", "content": question})
    if not selected_model:
        assistant_text = "No supported local model is available. Install Qwen or Llama in Ollama, then refresh this page."
        sources = []
    else:
        try:
            with st.spinner(f"Retrieving sources and asking {selected_model}..."):
                result = answer_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.vectors,
                    embed_texts,
                    lambda clean_question, context: generate_answer(
                        selected_model, clean_question, context, ollama_url, temperature
                    ),
                    top_k,
                    min_similarity,
                )
            assistant_text, sources = result.answer, result.sources
        except (ValueError, RuntimeError, OllamaError) as exc:
            assistant_text, sources = f"I could not complete that request: {exc}", []
        except Exception:
            logger.exception("Unexpected error while answering a document question")
            assistant_text = "I could not complete that request because of an unexpected local processing error. Please try again."
            sources = []
    st.session_state.history.append({"role": "assistant", "content": assistant_text, "sources": sources})
    st.rerun()