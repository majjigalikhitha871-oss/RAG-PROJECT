# Fieldnotes: Local RAG Document Intelligence

A workshop-ready Streamlit application for asking questions about PDF and DOCX documents. Extraction, chunking, embeddings, cosine retrieval, and answer generation run locally. No API keys or paid services are used.

## What happens to a question?

```text
PDF/DOCX -> extract text -> clean -> chunk -> local embeddings
          -> cosine similarity -> top-K sources -> context -> Ollama -> answer
```

RAG does not mean the LLM automatically reads an uploaded file. The application first finds relevant chunks, then supplies those chunks and the question to the local LLM. Embeddings represent semantic meaning as vectors. Cosine similarity measures the angle between vectors; a larger score means the vectors point in a more similar direction. The app normalizes vectors and calculates cosine scores with a NumPy dot product, so students can inspect the actual Top-K results.

## Requirements

- Windows, macOS, or Linux; Python 3.11 or 3.12 is recommended.
- Ollama installed and running for chat generation.
- About 1 GB of free disk space for Python packages and the small embedding model. The Ollama model requires additional space.
- CPU is supported. No GPU is required.

Python 3.14 may not yet have compatible prebuilt wheels for all ML dependencies. Use Python 3.11/3.12 if installation fails.

## Setup

Create and activate a virtual environment from this directory:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install and start Ollama from [ollama.com/download](https://ollama.com/download). Ollama exposes its local API at `http://localhost:11434` by default. Choose one model for the workshop; the app selects Qwen if present, otherwise Llama:

```powershell
ollama pull qwen2.5:3b
# Or, if your laptop has less memory:
ollama pull llama3.2:1b
```

The app does not download an LLM. It checks installed Ollama model tags at runtime and displays setup guidance when neither Qwen nor Llama is installed.

The embedding model is also kept local-only by the app. Download the small `all-MiniLM-L6-v2` model explicitly once while online, after installing the Python packages:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Then start the app:

```powershell
streamlit run app.py
```

## Workshop demo

1. Start Ollama and confirm the chosen model appears with `ollama list`.
2. Open the app, upload a syllabus or policy PDF/DOCX, and select **Build knowledge base**.
3. Review document, character, and chunk counts.
4. Ask: “What are the eligibility requirements mentioned in this document?”
5. Expand **Retrieved sources** to inspect chunk text, source/page, and cosine scores.
6. Ask something unrelated to demonstrate that a high enough similarity threshold prevents unrelated context from being sent to the LLM.
7. If Qwen is installed alongside Llama, observe that Qwen is selected; with only Llama installed, observe the fallback.

Example questions:

- What are the eligibility requirements?
- What documents must applicants submit?
- When does registration close?
- What is the attendance requirement?
- What does this document say about marine biology? (Use a document that does not discuss this topic.)

## Project map

- `app.py`: Streamlit layout, upload/index flow, settings, status, and chat history.
- `core/document_loader.py`: local PDF/DOCX extraction and friendly handling of empty/image-only PDFs.
- `core/text_cleaner.py`: whitespace/control-character cleanup while preserving paragraph breaks.
- `core/chunker.py`: overlapping word windows with source and page metadata.
- `core/embeddings.py`: CPU Sentence Transformers embeddings, normalized and loaded only from the local cache.
- `core/retriever.py`: NumPy cosine similarity, thresholding, Top-K ranking, and duplicate suppression.
- `core/ollama_client.py`: Ollama health/model detection, Qwen-first fallback, and grounded chat requests.
- `core/rag_pipeline.py`: joins query embedding, retrieval, context construction, and answer generation.
- `tests/`: extraction, cleaning, chunking, cosine retrieval, pipeline, and model selection checks.

## Design and safety notes

- Uploaded files are processed in memory and not written to disk. The index is in Streamlit session state; restarting the app or session clears it.
- PDF page numbers are retained. DOCX does not have stable page boundaries, so its source is retained without page numbers. Image-only PDFs need OCR, intentionally not bundled here.
- The embedding model is CPU-only and local-cache-only. It is separate from the Ollama generation model.
- Retrieved document text is untrusted data. It is placed in the user message as reference context and cannot override the system instruction. This reduces prompt-injection risk but does not make generated answers infallible; inspect the cited sources.
- Similarity is a ranking signal, not a truth score. Tune the minimum threshold for each document and embedding model.
- Very large uploads can consume laptop memory. Start with a few documents and a small Top-K.

## Troubleshooting

- **Ollama not connected:** open the Ollama app or run `ollama serve`; check the address in the sidebar.
- **No supported model:** run `ollama pull qwen2.5:3b` or `ollama pull llama3.2:1b`, then refresh the page.
- **Embedding model is not ready:** install requirements and run the explicit model download command above while online.
- **No PDF text found:** this is likely a scanned/image-only PDF; use a text-based PDF or OCR it before upload.
- **Slow answers:** choose a smaller Ollama model, reduce Top-K, or use shorter questions.
- **No relevant sources:** ask a document-specific question or lower the minimum cosine similarity in the sidebar.
- **Python package installation fails:** use Python 3.11 or 3.12 in a fresh virtual environment.

## Tests

The tests do not call Ollama or download embedding models. They use small known vectors and generated in-memory PDF/DOCX fixtures:

```powershell
python -m pytest
```