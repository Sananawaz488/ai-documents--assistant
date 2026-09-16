import io
import os
import re
import shutil
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# Simple AI Document Assistant
# Streamlit + Sentence Transformers + FAISS + Grok
# ============================================================

APP_TITLE = "AI Document Assistant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROK_MODEL = "grok-4.6"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_signature" not in st.session_state:
    st.session_state.processed_signature = None


# -----------------------------
# Cached embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# File extraction functions
# -----------------------------
def extract_pdf(file_bytes, file_name):
    """Extract text from PDF and keep page numbers."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "file_name": file_name,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def extract_docx(file_bytes, file_name):
    """Extract text from a DOCX file."""
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [
        {
            "file_name": file_name,
            "page": None,
            "text": text,
        }
    ]


def extract_text(file_bytes, file_name):
    """Extract text from TXT."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        return []

    return [
        {
            "file_name": file_name,
            "page": None,
            "text": text,
        }
    ]


def extract_markdown(file_bytes, file_name):
    """Extract text from Markdown."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        return []

    return [
        {
            "file_name": file_name,
            "page": None,
            "text": text,
        }
    ]


def extract_document(file_bytes, file_name):
    """Choose the correct extractor from the file extension."""
    extension = Path(file_name).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, file_name)

    if extension == ".docx":
        return extract_docx(file_bytes, file_name)

    if extension == ".txt":
        return extract_text(file_bytes, file_name)

    if extension == ".md":
        return extract_markdown(file_bytes, file_name)

    return []


# -----------------------------
# Chunking
# -----------------------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def create_chunks(extracted_pages):
    """Create chunks while preserving file and page metadata."""
    all_chunks = []

    for item in extracted_pages:
        pieces = split_text(item["text"])

        for piece in pieces:
            all_chunks.append(
                {
                    "text": piece,
                    "file_name": item["file_name"],
                    "page": item["page"],
                }
            )

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_vector_store(chunks):
    """Create embeddings once and build a FAISS index."""
    if not chunks:
        return None, None

    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index


# -----------------------------
# Keyword search
# -----------------------------
def important_words(question):
    """Return simple keywords from the question."""
    stop_words = {
        "what", "when", "where", "who", "why", "how",
        "is", "are", "was", "were", "the", "a", "an",
        "of", "to", "in", "on", "for", "and", "or",
        "does", "do", "can", "could", "would", "should",
        "please", "tell", "me", "about"
    }

    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    return [word for word in words if word not in stop_words and len(word) > 2]


def keyword_score(question, text):
    """Score a chunk using keyword overlap."""
    keywords = important_words(question)

    if not keywords:
        return 0.0

    lower_text = text.lower()
    matches = sum(1 for word in keywords if word in lower_text)

    return matches / len(keywords)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, chunks, index, embeddings, top_k=TOP_K):
    """
    Combine semantic similarity and keyword matching.
    Semantic search uses FAISS.
    Keyword search checks important question words.
    """
    if not chunks or index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_embedding = np.asarray(question_embedding, dtype="float32")

    semantic_scores, semantic_indices = index.search(
        question_embedding,
        min(len(chunks), max(top_k * 3, 10)),
    )

    candidates = {}

    for score, idx in zip(semantic_scores[0], semantic_indices[0]):
        if idx < 0:
            continue

        candidates[int(idx)] = float(score)

    # Add keyword-based candidates too.
    for idx, chunk in enumerate(chunks):
        score = keyword_score(question, chunk["text"])
        if score > 0:
            candidates.setdefault(idx, 0.0)

    ranked = []

    for idx, semantic_score in candidates.items():
        keyword = keyword_score(question, chunks[idx]["text"])

        # Both scores are normalized to roughly 0-1.
        hybrid_score = (0.75 * max(semantic_score, 0.0)) + (0.25 * keyword)

        ranked.append(
            {
                "index": idx,
                "hybrid_score": hybrid_score,
                "semantic_score": float(semantic_score),
                "keyword_score": float(keyword),
                "chunk": chunks[idx],
            }
        )

    ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return ranked[:top_k]


# -----------------------------
# Grok
# -----------------------------
def get_grok_client():
    """Read the Grok/xAI API key from Streamlit secrets."""
    api_key = st.secrets.get("GROK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROK_API_KEY is missing. Add it to Streamlit Secrets."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )


def ask_grok(question, retrieved_chunks):
    """Ask Grok to answer only from retrieved context."""
    context_parts = []

    for number, item in enumerate(retrieved_chunks, start=1):
        chunk = item["chunk"]
        page = chunk["page"]
        page_text = f"Page {page}" if page else "Page not available"

        context_parts.append(
            f"[Source {number}]\n"
            f"File: {chunk['file_name']}\n"
            f"{page_text}\n"
            f"Text: {chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Rules:
1. Answer ONLY from the supplied document context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I could not find this information in the uploaded documents."
4. Do not invent facts, page numbers, or sources.
5. Give a clear and concise answer.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer the question using only the document context above.
"""

    client = get_grok_client()

    response = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# -----------------------------
# Google Drive
# -----------------------------
def download_google_drive(url):
    """
    Download a public Google Drive file or folder.

    Folder links are downloaded into a temporary directory.
    Public access is required.
    """
    if "docs.google.com/document/d/" in url:
        match = re.search(r"/document/d/([^/]+)", url)
        if match:
            file_id = match.group(1)
            export_url = (
                f"https://docs.google.com/document/d/{file_id}/export?format=docx"
            )
            import requests

            response = requests.get(export_url, timeout=30)
            response.raise_for_status()

            return [
                (
                    f"google_drive_document_{file_id}.docx",
                    response.content,
                )
            ]

    temp_dir = tempfile.mkdtemp(prefix="drive_docs_")

    try:
        if "/folders/" in url:
            downloaded = gdown.download_folder(
                url,
                output=temp_dir,
                quiet=True,
                use_cookies=False,
            )

            files = []

            if downloaded:
                for path in downloaded:
                    path = Path(path)
                    if path.is_file():
                        files.append((path.name, path.read_bytes()))

            return files

        output_file = os.path.join(temp_dir, "drive_file")
        downloaded = gdown.download(
            url=url,
            output=output_file,
            quiet=True,
            fuzzy=True,
        )

        if downloaded and Path(downloaded).is_file():
            path = Path(downloaded)

            # gdown may not preserve an extension in some cases.
            # Try to infer it from the URL if needed.
            name = path.name
            url_name = url.split("?")[0].rstrip("/").split("/")[-1]

            if Path(url_name).suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
                name = url_name

            return [(name, path.read_bytes())]

        return []

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------
# Processing pipeline
# -----------------------------
def process_documents(file_items):
    """Extract -> chunk -> embed -> FAISS."""
    extracted = []

    for file_name, file_bytes in file_items:
        pages = extract_document(file_bytes, file_name)

        if pages:
            extracted.extend(pages)

    chunks = create_chunks(extracted)

    embeddings, index = build_vector_store(chunks)

    return extracted, chunks, embeddings, index


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Assistant")
st.caption(
    "Upload documents, search them semantically and by keywords, "
    "then ask questions using Grok."
)

st.sidebar.header("Document Sources")

source_type = st.sidebar.radio(
    "Choose source",
    ["Local Upload", "Google Drive"],
)

file_items = []

if source_type == "Local Upload":
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        file_items = [
            (uploaded.name, uploaded.getvalue())
            for uploaded in uploaded_files
        ]

else:
    drive_url = st.sidebar.text_input(
        "Paste a public Google Drive file or folder link"
    )

    if st.sidebar.button("Load from Google Drive"):
        if not drive_url.strip():
            st.warning("Please paste a Google Drive link.")
        else:
            with st.spinner("Loading files from Google Drive..."):
                try:
                    file_items = download_google_drive(drive_url.strip())

                    if not file_items:
                        st.error(
                            "No supported files were found. "
                            "Make sure the Drive item is public."
                        )
                    else:
                        st.session_state.drive_files = file_items
                        st.success(
                            f"Loaded {len(file_items)} file(s) from Google Drive."
                        )

                except Exception as exc:
                    st.error(f"Google Drive loading failed: {exc}")

    file_items = st.session_state.get("drive_files", [])

if file_items:
    st.subheader("Selected Documents")

    for file_name, file_bytes in file_items:
        st.write(f"📄 **{file_name}** — {len(file_bytes):,} bytes")

    # A stable signature prevents re-embedding the same documents
    # every time Streamlit reruns the script.
    signature = tuple(
        (name, len(data), hash(data))
        for name, data in file_items
    )

    if signature != st.session_state.processed_signature:
        with st.spinner("Extracting, chunking, and creating embeddings..."):
            try:
                extracted, chunks, embeddings, index = process_documents(
                    file_items
                )

                st.session_state.extracted = extracted
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.faiss_index = index
                st.session_state.processed_signature = signature

            except Exception as exc:
                st.error(f"Document processing failed: {exc}")

# -----------------------------
# Document information
# -----------------------------
if st.session_state.get("extracted"):
    st.subheader("📄 Document Information")

    extracted = st.session_state.extracted
    chunks = st.session_state.chunks

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(set(x["file_name"] for x in extracted)))
    col2.metric("Extracted Sections", len(extracted))
    col3.metric("Created Chunks", len(chunks))

    for file_name in sorted(set(x["file_name"] for x in extracted)):
        matching = [x for x in extracted if x["file_name"] == file_name]

        with st.expander(f"📄 {file_name}"):
            st.write(f"Extracted sections/pages: {len(matching)}")

            for item in matching:
                page = (
                    f"Page {item['page']}"
                    if item["page"] is not None
                    else "Page not available"
                )

                st.markdown(f"**{page}**")
                st.write(item["text"][:2000])

    st.divider()
    st.subheader("🧩 Chunk Information")
    st.write(f"**Total chunks:** {len(chunks)}")
    st.caption(
        f"Chunk size: {CHUNK_SIZE} characters | "
        f"Overlap: {CHUNK_OVERLAP} characters"
    )

    if chunks:
        with st.expander("Preview chunks"):
            for i, chunk in enumerate(chunks[:10], start=1):
                page = (
                    f"Page {chunk['page']}"
                    if chunk["page"] is not None
                    else "Page not available"
                )
                st.markdown(
                    f"**Chunk {i} — {chunk['file_name']} — {page}**"
                )
                st.write(chunk["text"])


# -----------------------------
# Question answering
# -----------------------------
st.divider()
st.subheader("💬 Ask Your Documents")

question = st.text_input(
    "Ask a question about the uploaded documents",
    placeholder="Example: What is the main purpose of this document?",
)

if st.button("Ask", type="primary"):
    if not st.session_state.chunks:
        st.warning("Please upload or load at least one document first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching documents..."):
            results = hybrid_search(
                question,
                st.session_state.chunks,
                st.session_state.faiss_index,
                st.session_state.embeddings,
            )

        if not results:
            st.warning(
                "I could not find relevant document chunks for this question."
            )

        else:
            with st.spinner("Generating answer with Grok..."):
                try:
                    answer = ask_grok(question, results)

                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### 🔎 Retrieved Sources")

                    for number, result in enumerate(results, start=1):
                        chunk = result["chunk"]
                        page = (
                            f"Page {chunk['page']}"
                            if chunk["page"] is not None
                            else "Page not available"
                        )

                        with st.expander(
                            f"Source {number}: {chunk['file_name']} — {page}"
                        ):
                            st.write(
                                f"Hybrid score: {result['hybrid_score']:.3f}"
                            )
                            st.write(
                                f"Semantic score: {result['semantic_score']:.3f}"
                            )
                            st.write(
                                f"Keyword score: {result['keyword_score']:.3f}"
                            )
                            st.markdown("**Retrieved text:**")
                            st.write(chunk["text"])

                except Exception as exc:
                    st.error(f"Grok request failed: {exc}")


st.sidebar.divider()
st.sidebar.caption(
    "Supported: PDF, DOCX, TXT, MD + public Google Drive files/folders."
)
