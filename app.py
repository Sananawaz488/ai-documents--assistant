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
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# AI DOCUMENT ASSISTANT
# RAG + FAISS + Sentence Transformers + GroqCloud
# OpenAI-compatible client
# ============================================================

APP_TITLE = "AI Document Assistant"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# GroqCloud model
# This is an OpenAI model hosted through GroqCloud.
CHAT_MODEL = "openai/gpt-oss-120b"

# Groq OpenAI-compatible endpoint
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Chunk settings
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Number of retrieved chunks
TOP_K = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_signature" not in st.session_state:
    st.session_state.processed_signature = None


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf(file_bytes, file_name):
    """
    Extract text from PDF.
    Keeps filename and page number.
    """

    results = []

    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()

        if text:
            results.append(
                {
                    "file_name": file_name,
                    "page": page_number,
                    "text": text,
                }
            )

    return results


def extract_docx(file_bytes, file_name):
    """
    Extract text from DOCX.
    DOCX does not reliably provide page numbers through python-docx.
    """

    results = []

    document = Document(io.BytesIO(file_bytes))

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    if full_text.strip():
        results.append(
            {
                "file_name": file_name,
                "page": None,
                "text": full_text,
            }
        )

    return results


def extract_text(file_bytes, file_name):
    """
    Extract TXT file.
    """

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore")

    text = text.strip()

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
    """
    Extract Markdown file.
    """

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore")

    text = text.strip()

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
    """
    Detect file type and extract text.
    """

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


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character chunks.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_chunks(extracted_documents):
    """
    Create chunks while preserving:
    - filename
    - page number
    """

    all_chunks = []

    for document in extracted_documents:

        file_name = document["file_name"]
        page = document["page"]
        text = document["text"]

        text_chunks = split_text(text)

        for chunk_number, chunk_text in enumerate(
            text_chunks,
            start=1
        ):

            all_chunks.append(
                {
                    "file_name": file_name,
                    "page": page,
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                }
            )

    return all_chunks


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

def build_vector_store(chunks):
    """
    Create embeddings and FAISS index.
    """

    if not chunks:
        return None, None

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return embeddings, index


# ============================================================
# KEYWORD SEARCH
# ============================================================

STOP_WORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "this",
    "that",
    "these",
    "those",
    "can",
    "could",
    "should",
    "would",
    "please",
}


def important_words(question):
    """
    Extract important words from user question.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]{3,}\b",
        question.lower()
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def keyword_score(question, text):
    """
    Calculate simple keyword matching score.
    """

    keywords = important_words(question)

    if not keywords:
        return 0.0

    text_lower = text.lower()

    matches = 0

    for keyword in keywords:
        if keyword in text_lower:
            matches += 1

    return matches / len(keywords)


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(question, top_k=TOP_K):
    """
    Combine:
    1. Semantic FAISS search
    2. Keyword search
    """

    chunks = st.session_state.chunks
    index = st.session_state.faiss_index

    if not chunks or index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Search more candidates first
    candidate_count = min(
        max(top_k * 3, 10),
        len(chunks)
    )

    semantic_scores, indices = index.search(
        question_embedding,
        candidate_count
    )

    candidates = {}

    # Semantic candidates
    for score, index_position in zip(
        semantic_scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        candidates[int(index_position)] = float(score)

    # Keyword candidates
    for index_position, chunk in enumerate(chunks):

        score = keyword_score(
            question,
            chunk["text"]
        )

        if score > 0:
            if index_position in candidates:
                candidates[index_position] = max(
                    candidates[index_position],
                    0.0
                )
            else:
                candidates[index_position] = 0.0

    ranked_results = []

    for index_position, semantic_score in candidates.items():

        chunk = chunks[index_position]

        keyword_match = keyword_score(
            question,
            chunk["text"]
        )

        # Hybrid ranking
        hybrid_score = (
            0.75 * semantic_score
            + 0.25 * keyword_match
        )

        result = {
            **chunk,
            "semantic_score": semantic_score,
            "keyword_score": keyword_match,
            "hybrid_score": hybrid_score,
        }

        ranked_results.append(result)

    ranked_results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return ranked_results[:top_k]


# ============================================================
# GROQ + OPENAI CLIENT
# ============================================================

def get_chat_client():
    """
    Uses GroqCloud API through OpenAI-compatible client.
    """

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    api_key = api_key.strip()

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )


# ============================================================
# ASK CHAT MODEL
# ============================================================

def ask_chat_model(question, retrieved_chunks):
    """
    Send only retrieved document context to the model.
    """

    if not retrieved_chunks:
        return (
            "I could not find this information in the "
            "uploaded documents."
        )

    context_parts = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"""
SOURCE {number}
File: {chunk['file_name']}
{page_text}
Chunk: {chunk['chunk_number']}

Content:
{chunk['text']}
"""
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an AI Document Assistant.

Your job is to answer questions using ONLY the
document context provided by the application.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not supported by the provided
   document context, say:

   "I could not find this information in the uploaded documents."

4. Give a clear and concise answer.
5. When useful, mention the relevant document or page.
6. Treat the supplied context as the only source of truth.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question using only the document context.
"""

    client = get_chat_client()

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# GOOGLE DRIVE
# ============================================================

def download_google_drive(url):
    """
    Download Google Drive file/folder.
    Also supports public Google Docs.
    """

    temp_dir = tempfile.mkdtemp(
        prefix="ai_document_assistant_"
    )

    # Google Docs document
    if "docs.google.com/document/d/" in url:

        match = re.search(
            r"/document/d/([a-zA-Z0-9_-]+)",
            url
        )

        if match:

            document_id = match.group(1)

            export_url = (
                "https://docs.google.com/document/d/"
                f"{document_id}/export?format=docx"
            )

            output_path = Path(temp_dir) / "google_document.docx"

            import requests

            response = requests.get(
                export_url,
                timeout=60
            )

            response.raise_for_status()

            output_path.write_bytes(
                response.content
            )

            return temp_dir

    # Google Drive file/folder
    gdown.download_folder(
        url=url,
        output=temp_dir,
        quiet=True,
        use_cookies=False,
    )

    return temp_dir


def read_drive_files(folder_path):
    """
    Read supported files downloaded from Google Drive.
    """

    supported_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    }

    extracted_documents = []

    folder = Path(folder_path)

    for file_path in folder.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            file_bytes = file_path.read_bytes()

            extracted = extract_document(
                file_bytes,
                file_path.name
            )

            extracted_documents.extend(
                extracted
            )

        except Exception as error:
            st.warning(
                f"Could not read {file_path.name}: {error}"
            )

    return extracted_documents


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(document_files):
    """
    Full RAG pipeline:

    Extraction
    ↓
    Chunking
    ↓
    Embeddings
    ↓
    FAISS
    """

    extracted_documents = []

    for file_name, file_bytes in document_files:

        extracted = extract_document(
            file_bytes,
            file_name
        )

        extracted_documents.extend(
            extracted
        )

    chunks = create_chunks(
        extracted_documents
    )

    if not chunks:
        return [], None, None

    embeddings, index = build_vector_store(
        chunks
    )

    return chunks, embeddings, index


# ============================================================
# APP HEADER
# ============================================================

st.title("📚 AI Document Assistant")

st.markdown(
    """
Upload documents and ask questions about them.

The application uses **RAG + FAISS + Sentence Transformers**
and an **OpenAI GPT-OSS 120B chat model through GroqCloud**.
"""
)


# ============================================================
# DOCUMENT SOURCE
# ============================================================

st.sidebar.header("📂 Document Sources")

source_type = st.sidebar.radio(
    "Choose document source:",
    [
        "Local Upload",
        "Google Drive",
    ]
)


# ============================================================
# LOCAL UPLOAD
# ============================================================

document_files = []

if source_type == "Local Upload":

    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()

            document_files.append(
                (
                    uploaded_file.name,
                    file_bytes,
                )
            )


# ============================================================
# GOOGLE DRIVE
# ============================================================

else:

    drive_url = st.sidebar.text_input(
        "Google Drive file/folder link"
    )

    if st.sidebar.button(
        "Load from Google Drive"
    ):

        if not drive_url.strip():

            st.sidebar.error(
                "Please enter a Google Drive URL."
            )

        else:

            with st.spinner(
                "Downloading files from Google Drive..."
            ):

                try:

                    drive_folder = download_google_drive(
                        drive_url.strip()
                    )

                    extracted_documents = read_drive_files(
                        drive_folder
                    )

                    chunks = create_chunks(
                        extracted_documents
                    )

                    if not chunks:

                        st.error(
                            "No supported documents were found."
                        )

                    else:

                        embeddings, index = (
                            build_vector_store(chunks)
                        )

                        st.session_state.chunks = chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.faiss_index = index

                        st.success(
                            f"Loaded {len(chunks)} chunks "
                            "from Google Drive."
                        )

                    shutil.rmtree(
                        drive_folder,
                        ignore_errors=True
                    )

                except Exception as error:

                    st.error(
                        f"Google Drive error: {error}"
                    )


# ============================================================
# PROCESS LOCAL DOCUMENTS
# ============================================================

if document_files:

    # Create a stable signature
    signature_parts = []

    for file_name, file_bytes in document_files:

        signature_parts.append(
            (
                file_name,
                len(file_bytes),
                hash(file_bytes),
            )
        )

    current_signature = tuple(
        signature_parts
    )

    if (
        st.session_state.processed_signature
        != current_signature
    ):

        with st.spinner(
            "Processing documents..."
        ):

            chunks, embeddings, index = (
                process_documents(
                    document_files
                )
            )

            if chunks:

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.faiss_index = index

                st.session_state.processed_signature = (
                    current_signature
                )

                st.success(
                    f"Documents processed successfully. "
                    f"{len(chunks)} chunks created."
                )

            else:

                st.error(
                    "No readable text was found "
                    "in the uploaded documents."
                )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.chunks:

    st.subheader("📊 Document Information")

    unique_files = sorted(
        set(
            chunk["file_name"]
            for chunk in st.session_state.chunks
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Documents",
            len(unique_files)
        )

    with col2:

        st.metric(
            "Total Chunks",
            len(st.session_state.chunks)
        )

    with st.expander(
        "View processed documents"
    ):

        for file_name in unique_files:

            st.write(
                f"📄 {file_name}"
            )


# ============================================================
# ASK YOUR DOCUMENTS
# ============================================================

st.divider()

st.subheader("💬 Ask Your Documents")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What is the main purpose of this document?"
    ),
    height=120,
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔎 Ask",
    type="primary",
):

    if not st.session_state.chunks:

        st.warning(
            "Please upload or load a document first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                retrieved_chunks = hybrid_search(
                    question.strip(),
                    TOP_K
                )

                answer = ask_chat_model(
                    question.strip(),
                    retrieved_chunks
                )

                st.subheader("🤖 Answer")

                st.write(answer)

                # ==========================================
                # SOURCES
                # ==========================================

                st.subheader(
                    "📚 Retrieved Sources"
                )

                if retrieved_chunks:

                    for number, source in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        page_info = (
                            f"Page {source['page']}"
                            if source["page"] is not None
                            else "Page not available"
                        )

                        with st.expander(
                            f"Source {number}: "
                            f"{source['file_name']} "
                            f"({page_info})"
                        ):

                            st.write(
                                f"**File:** "
                                f"{source['file_name']}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{page_info}"
                            )

                            st.write(
                                f"**Chunk:** "
                                f"{source['chunk_number']}"
                            )

                            st.write(
                                f"**Semantic Score:** "
                                f"{source['semantic_score']:.4f}"
                            )

                            st.write(
                                f"**Keyword Score:** "
                                f"{source['keyword_score']:.4f}"
                            )

                            st.write(
                                f"**Hybrid Score:** "
                                f"{source['hybrid_score']:.4f}"
                            )

                            st.markdown(
                                "**Retrieved Text:**"
                            )

                            st.write(
                                source["text"]
                            )

            except Exception as error:

                st.error(
                    f"Chat request failed: {error}"
                )


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.markdown(
    """
### Supported Files

- PDF
- DOCX
- TXT
- Markdown

### RAG Pipeline

Document → Extraction → Chunking → Embeddings → FAISS → Hybrid Search → AI Answer

### Chat Model

`openai/gpt-oss-120b`

### Provider

GroqCloud
"""
)
