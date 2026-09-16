📚 AI Document Assistant

A simple Streamlit RAG document assistant that can read local documents or supported public Google Drive files, create reusable embeddings, search with FAISS + keywords, and answer questions with Grok.

Features

PDF, DOCX, TXT and MD upload

Google Drive file/folder source

Text extraction with file names and PDF page numbers

Overlapping text chunking

Sentence Transformers embeddings

FAISS semantic search

Keyword search

Hybrid document search

Reuses embeddings instead of embedding documents for every question

Grok answers only from retrieved context

Retrieved source text shown after every answer

Streamlit session state and caching

Files

app.py
requirements.txt
README.md

Run

pip install -r requirements.txt
streamlit run app.py

Streamlit Secrets

Add your Grok/xAI API key in Streamlit Secrets:

GROK_API_KEY = "your_key_here"

The key is read from st.secrets and is never hard-coded in app.py.

Google Drive

Use a public Google Drive file/folder link. Supported document types are PDF, DOCX, TXT and MD. A native Google Docs document can also be loaded through its export link.

RAG Flow

Document
   ↓
Text Extraction
   ↓
Overlapping Chunks + Metadata
   ↓
Sentence Transformer Embeddings
   ↓
FAISS Index
   ↓
User Question
   ↓
Question Embedding + Keyword Search
   ↓
Hybrid Ranking
   ↓
Top Document Chunks
   ↓
Grok
   ↓
Answer + Sources
