"""
Hybrid-Search RAG Demo — single-file Streamlit app.


Deploy on Render (or any host) as a single Streamlit web service:
    Start command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Required environment variables (set as Render secrets / .env locally):
    PINECONE_API_KEY
    GROQ_API_KEY
Optional:
    PINECONE_INDEX_NAME    (default: rag-demo-hybrid)
    PINECONE_CLOUD         (default: aws)
    PINECONE_REGION        (default: us-east-1)
    DENSE_EMBEDDING_MODEL  (default: sentence-transformers/all-MiniLM-L6-v2)
    DENSE_EMBEDDING_DIM    (default: 384)
    GROQ_CHAT_MODEL        (default: llama-3.3-70b-versatile)

on free-tier limits: Groq's free tier for llama-3.3-70b-versatile is
rate-limited at roughly 30 requests/minute, 1,000 requests/day, 12,000
tokens/minute, and 100,000 tokens/day (check your own console for current
numbers — limits change). For RAG, token usage per query (system prompt +
retrieved chunks + question + answer) is usually the binding constraint
before request count is, since each call can easily use 2,000-3,000 tokens.
Lower top_k or chunk_size if you're hitting 429s.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from groq import Groq

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-demo-hybrid")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Dense embeddings run locally via sentence-transformers — free, no API key,
# no rate limit, no quota to exhaust.
DENSE_EMBEDDING_MODEL = os.getenv("DENSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_EMBEDDING_DIM = int(os.getenv("DENSE_EMBEDDING_DIM", "384"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.5"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the provided "
    "context excerpts. If the answer isn't contained in the context, say you "
    "don't have enough information rather than guessing. Be concise."
)


# ============================================================================
# Cached singletons (loaded once per server process, not per rerun)
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_dense_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Add it as an environment variable.")
    return Pinecone(api_key=PINECONE_API_KEY)


@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it as an environment variable.")
    return Groq(api_key=GROQ_API_KEY)


def ensure_index_exists() -> None:
    pc = get_pinecone_client()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DENSE_EMBEDDING_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def get_index():
    ensure_index_exists()
    return get_pinecone_client().Index(PINECONE_INDEX_NAME)


# BM25 is stateful (fit on the corpus) and changes every time documents are
# (re)processed, so it lives in session_state rather than st.cache_resource.
def get_bm25_encoder() -> Optional[BM25Encoder]:
    return st.session_state.get("bm25_encoder")


def set_bm25_encoder(encoder: BM25Encoder) -> None:
    st.session_state["bm25_encoder"] = encoder



# ============================================================================
# Document loading
# ============================================================================

def load_single_file(file_path: str, original_filename: str) -> List[Document]:
    ext = os.path.splitext(original_filename)[1].lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    for d in docs:
        d.metadata["source"] = original_filename
        d.metadata.setdefault("page", d.metadata.get("page", None))
    return docs


def load_uploaded_files(uploaded_files) -> tuple[List[Document], List[str]]:
    """Save Streamlit UploadedFile objects to a temp dir, load, then clean up."""
    all_docs: List[Document] = []
    errors: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            ext = os.path.splitext(f.name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"{f.name}: unsupported file type {ext}")
                continue
            tmp_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{ext}")
            with open(tmp_path, "wb") as out:
                out.write(f.getvalue())
            try:
                all_docs.extend(load_single_file(tmp_path, f.name))
            except Exception as e:
                errors.append(f"{f.name}: {e}")

    return all_docs, errors


# ============================================================================
# Chunking
# ============================================================================

def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_index"] = i

    return chunks


# ============================================================================
# Hybrid vector store operations
# ============================================================================

def delete_all_vectors() -> None:
    index = get_index()
    try:
        index.delete(delete_all=True)
    except Exception:
        # Raised by Pinecone if the namespace is already empty - safe to ignore.
        pass


def _scale_dense(dense: List[float], alpha: float) -> List[float]:
    return [v * alpha for v in dense]


def _scale_sparse(sparse: Dict[str, List], alpha: float) -> Dict[str, List]:
    return {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]}


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Pinecone metadata values must be string, number, boolean, or list of
    strings — None/null is rejected. Drop any None-valued keys entirely
    (e.g. "page" for .txt/.docx files, which have no page numbers).
    """
    return {k: v for k, v in metadata.items() if v is not None}


def upsert_chunks(chunks: List[Document], batch_size: int = 64) -> int:
    if not chunks:
        return 0

    index = get_index()
    dense_model = get_dense_embeddings()
    bm25 = get_bm25_encoder()
    if bm25 is None:
        raise RuntimeError("BM25 encoder has not been fit yet.")

    texts = [c.page_content for c in chunks]
    dense_vectors = dense_model.embed_documents(texts)
    sparse_vectors = bm25.encode_documents(texts)

    total = 0
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch = []
        for chunk, dvec, svec in zip(chunks[start:end], dense_vectors[start:end], sparse_vectors[start:end]):
            batch.append({
                "id": chunk.metadata["chunk_id"],
                "values": dvec,
                "sparse_values": svec,
                "metadata": _clean_metadata({
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                    "page": chunk.metadata.get("page"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                }),
            })
        index.upsert(vectors=batch)
        total += len(batch)

    return total


def hybrid_search(query: str, alpha: float, top_k: int) -> List[Dict[str, Any]]:
    alpha = max(0.0, min(1.0, alpha))

    index = get_index()
    dense_model = get_dense_embeddings()
    bm25 = get_bm25_encoder()
    if bm25 is None:
        return []

    dense_query = dense_model.embed_query(query)
    sparse_query = bm25.encode_queries(query)

    scaled_dense = _scale_dense(dense_query, alpha)
    scaled_sparse = _scale_sparse(sparse_query, alpha)

    results = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )

    matches = []
    for m in results.get("matches", []):
        meta = m.get("metadata", {})
        matches.append({
            "id": m["id"],
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
        })
    return matches


# ============================================================================
# LLM answer generation (Groq, single model, free tier)
# ============================================================================

def build_context_block(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = c.get("page")
        loc = c.get("source", "unknown") + (f" (page {page + 1})" if page is not None else "")
        parts.append(f"[Chunk {i} — {loc}]\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    context = build_context_block(retrieved_chunks)
    client = get_groq_client()
    completion = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return completion.choices[0].message.content


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Hybrid RAG", page_icon="🔎", layout="wide")

if "vector_weight" not in st.session_state:
    st.session_state.vector_weight = DEFAULT_ALPHA
if "keyword_weight" not in st.session_state:
    st.session_state.keyword_weight = round(1 - DEFAULT_ALPHA, 2)
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bm25_encoder" not in st.session_state:
    st.session_state.bm25_encoder = None


def _on_vector_change():
    st.session_state.keyword_weight = round(1 - st.session_state.vector_weight, 2)


def _on_keyword_change():
    st.session_state.vector_weight = round(1 - st.session_state.keyword_weight, 2)


# --- Sidebar: retrieval configuration -------------------------------------

with st.sidebar:
    st.header("⚙️ Retrieval Settings")

    missing_keys = []
    if not PINECONE_API_KEY:
        missing_keys.append("PINECONE_API_KEY")
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if missing_keys:
        st.error(f"Missing environment variable(s): {', '.join(missing_keys)}")

    st.caption(f"LLM: **Groq** — {GROQ_CHAT_MODEL}")

    st.subheader("Hybrid Search Weights")
    st.slider(
        "Vector (semantic) search weight", 0.0, 1.0, step=0.05,
        key="vector_weight", on_change=_on_vector_change,
        help="Dense/semantic similarity weight. Keyword weight auto-adjusts to (1 - this).",
    )
    st.slider(
        "Keyword (BM25) search weight", 0.0, 1.0, step=0.05,
        key="keyword_weight", on_change=_on_keyword_change,
        help="Exact keyword-matching weight. Vector weight auto-adjusts to (1 - this).",
    )
    st.caption(
        f"Current split → Vector: **{st.session_state.vector_weight:.2f}** / "
        f"Keyword: **{st.session_state.keyword_weight:.2f}**"
    )

    st.divider()
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 200, 4000, DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.slider(
        "Chunk overlap (characters)", 0, min(2000, chunk_size - 50),
        min(DEFAULT_CHUNK_OVERLAP, chunk_size - 50), step=50,
    )

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 15, DEFAULT_TOP_K)

    st.divider()
    if st.button("🗑️ Reset everything (clear index)", use_container_width=True):
        try:
            delete_all_vectors()
            st.session_state.bm25_encoder = None
            st.session_state.documents_processed = False
            st.session_state.chat_history = []
            st.success("Reset complete. Upload new documents to start again.")
        except Exception as e:
            st.error(f"Reset failed: {e}")


# --- Main layout ------------------------------------------------------------

st.title("🔎 Hybrid Search RAG")
st.caption("LangChain + Pinecone (hybrid dense/sparse) +  local embeddings +  Groq LLM")

upload_col, chat_col = st.columns([1, 1.4], gap="large")

with upload_col:
    st.subheader("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Supported formats: .txt, .pdf, .docx",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

    st.subheader("2. Process documents")
    process_clicked = st.button(
        "🚀 Process documents", type="primary", use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_clicked and uploaded_files:
        with st.spinner("Loading and chunking documents..."):
            try:
                raw_docs, load_errors = load_uploaded_files(uploaded_files)
            except Exception as e:
                st.error(f"Failed to load files: {e}")
                raw_docs, load_errors = [], []

        if not raw_docs:
            st.error(f"None of the uploaded files could be parsed. Errors: {load_errors}")
        else:
            if load_errors:
                st.warning(f"Some files had load errors: {load_errors}")

            chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap)

            with st.spinner("Fitting keyword index and embedding chunks (this may take a minute)..."):
                try:
                    delete_all_vectors()
                    bm25 = BM25Encoder()
                    bm25.fit([c.page_content for c in chunks])
                    set_bm25_encoder(bm25)

                    upserted = upsert_chunks(chunks)

                    st.session_state.documents_processed = True
                    st.success(
                        f"Processed {len(uploaded_files)} file(s) → "
                        f"{len(raw_docs)} raw doc(s) → "
                        f"{len(chunks)} chunk(s) → "
                        f"{upserted} vector(s) indexed."
                    )
                except Exception as e:
                    st.error(f"Processing failed: {e}")

    if st.session_state.documents_processed:
        st.info(" Documents are indexed. Ask questions on the right →")

with chat_col:
    st.subheader("3. Ask questions")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What does the document say about...?",
        disabled=not st.session_state.documents_processed,
    )
    ask_clicked = st.button(
        "Ask", type="primary",
        disabled=not st.session_state.documents_processed or not question,
    )

    if not st.session_state.documents_processed:
        st.caption("Upload and process documents on the left before asking questions.")

    if ask_clicked and question:
        with st.spinner("Retrieving chunks and generating answer..."):
            try:
                retrieved = hybrid_search(question, alpha=st.session_state.vector_weight, top_k=top_k)
                if not retrieved:
                    answer = "No relevant chunks were found. Have you processed any documents yet?"
                else:
                    answer = generate_answer(question, retrieved)

                st.session_state.chat_history.insert(0, {
                    "question": question,
                    "answer": answer,
                    "chunks": retrieved,
                    "vector_weight": st.session_state.vector_weight,
                    "keyword_weight": st.session_state.keyword_weight,
                })
            except Exception as e:
                st.error(f"Query failed: {e}")

    st.divider()

    for turn in st.session_state.chat_history:
        st.markdown(f"**Question:** {turn['question']}")
        st.markdown(f"**Answer:** {turn['answer']}")
        st.caption(
            f"Search weights used → vector: {turn['vector_weight']:.2f}, "
            f"keyword: {turn['keyword_weight']:.2f}"
        )

        with st.expander(f"Retrieved chunks ({len(turn['chunks'])})", expanded=False):
            if not turn["chunks"]:
                st.write("No chunks retrieved.")
            for i, chunk in enumerate(turn["chunks"], start=1):
                page_info = f", page {chunk['page'] + 1}" if chunk.get("page") is not None else ""
                st.markdown(f"**Chunk {i}** — *{chunk['source']}{page_info}*")
                st.text(chunk["text"])
                st.markdown("---")

        st.divider()
