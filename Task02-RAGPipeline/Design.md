# Design Document — Legal Contract RAG System

## Overview

This system is a Retrieval-Augmented Generation (RAG) pipeline designed for question answering over a corpus of legal contracts. Given a natural-language question that references a specific contract (e.g. *"What is the financing term in Contract_11?"*), the system retrieves the most relevant clauses from the correct document and generates a grounded, cited answer using an LLM.

The primary use case is legal document Q&A where contracts share near-identical boilerplate text, making document disambiguation the central technical challenge.

---

## Architecture

```
PDF Files
    │
    ▼
[helper.py] load_pdfs()
    │  Per-page text extraction (PyPDF2)
    ▼
[rag.py] chunk_text()
    │  Legal-aware clause splitting (CLAUSE / SECTION / ARTICLE boundaries)
    ▼
[rag.py] ingest()
    │  Document-prefixed embedding via all-mpnet-base-v2
    │  Stored in FAISS IndexFlatL2
    ▼
[rag.py] retrieve()
    │  Top-K ANN search (default K=5)
    ▼
[rag.py] rerank()
    │  Keyword overlap + document-match bonus
    │  Returns top rerank_k=3 chunks
    ▼
[prompt.py] build_prompt()
    │  Assembles context + question into LLM prompt
    ▼
[llm_endpoint.py] generate_answer()
    │  Primary: Gemini 2.5 Flash
    │  Fallback: GPT-4o
    ▼
QueryResponse { answer, sources, confidence }
```

---

## Module Breakdown

### `main.py`
Entry point for interactive CLI use. Loads config, builds and ingests the pipeline, then runs a REPL loop accepting questions from stdin and printing the answer, confidence score, and source citations.

### `evaluation.py`
Offline evaluation harness. Runs a fixed 10-question test set and computes **Precision@3** — whether the ground-truth contract appears in the top-3 retrieved chunks. Used to benchmark retrieval quality during development.

### `rag.py` — `RAGPipeline`
The core pipeline class. Handles all five stages:

**1. Chunking (`chunk_text`)**
Splits each page's text at legal clause boundaries using a regex that matches `CLAUSE <name>:`, `SECTION <n>:`, and `ARTICLE <n>:` headers. This produces semantically coherent units aligned to how lawyers read contracts, rather than arbitrary fixed-size windows.

**2. Ingestion (`ingest`)**
For each chunk, the document filename (minus `.pdf`) is prepended before embedding — e.g. `"Contract_11: CLAUSE TWO – TERM: ..."`. This encodes document identity into the vector, solving the disambiguation problem that arises when contracts share identical boilerplate. Embeddings are stored in a FAISS `IndexFlatL2` (exact L2 search; appropriate for corpora up to ~100k chunks).

**3. Retrieval (`retrieve`)**
Embeds the raw question and runs an ANN search for the top `top_k` (default 5) nearest chunks. Because the query naturally contains the contract name (e.g. *"in Contract_11"*), it aligns well with the document-prefixed chunk embeddings.

**4. Reranking (`rerank`)**
A lightweight hybrid reranker that combines:
- **Keyword overlap score** — word-level intersection between question tokens and chunk tokens.
- **Document-match bonus** — a hard `+100` bonus for chunks whose document matches the contract explicitly named in the query (extracted via `extract_target_doc`).

This ensures that even if the semantic search returns chunks from the wrong contract, the reranker surfaces the correct one. Returns the top `rerank_k` (default 3) chunks.

**5. Confidence scoring (`compute_confidence`)**
A lightweight heuristic: average keyword overlap across the reranked chunks, normalised to `[0, 1]`. Not a calibrated probability — used as a rough quality signal surfaced to the caller.

### `llm_endpoint.py`
Unified LLM interface. Uses `all-mpnet-base-v2` (via `sentence-transformers`) for local embeddings — no API calls needed for the vector search. For generation it tries **Gemini 2.5 Flash** first (free tier), falling back to **GPT-4o** on failure. Both clients are initialised at module load from environment variables.

### `prompt.py`
Constructs the final LLM prompt. Uses a strict system prompt (`SYSTEM_PROMPT_V1`) that instructs the model to answer only from provided context, cite document name and page, and respond with *"Insufficient context to answer"* when the answer is not present.

### `helper.py`
Utility functions: `load_config` reads `config.yaml`, and `load_pdfs` walks the data directory and extracts per-page text from every PDF using PyPDF2.

### `schema.py`
Pydantic models: `QueryRequest` (validates question length ≥ 3), `QueryResponse` (answer, sources list, confidence in `[0,1]`), and `Source` (document, page, chunk text).

### `log.py`
Module-level logger configured with INFO level and timestamp formatting.

### `config.yaml`
Runtime configuration: model names, chunking parameters (`chunk_size`, `chunk_overlap` — reserved for future fixed-size fallback), retrieval parameters (`top_k`, `rerank_k`), FAISS index path, and data directory paths.

---

## Key Design Decisions

###  Document-prefixed embeddings

The corpus consists of contracts that are structurally identical — same clause names and boilerplate wording, with differences primarily in entity names, amounts, and dates. Embedding clause text alone leads to highly similar vectors across documents, which degrades retrieval precision at the document level.

Injecting the document identifier directly into the embedding input is a deliberate biasing strategy: it encodes document-level separation into the vector space itself rather than relying on downstream filtering. This avoids an additional dependency on metadata-aware retrieval logic and ensures that document identity influences similarity scoring during the initial retrieval step, not after.

This approach prioritizes retrieval correctness under high textual overlap, which is the dominant challenge in this corpus.

### Legal-aware chunking over fixed-size sliding windows

Legal documents are structured around clauses that represent complete semantic and contractual units. Fixed-size chunking introduces boundary fragmentation—splitting obligations across chunks or merging unrelated clauses—which reduces both retrieval precision and answer faithfulness.

Clause-aware chunking aligns the retrieval unit with how legal reasoning is performed. Each chunk corresponds to a self-contained provision, improving:

- semantic coherence during embedding
- retrieval alignment with query intent
- citation quality in generated answers

The trade-off in chunk size variability is acceptable because embedding models handle moderate length variance well, while broken semantics are significantly harder to recover from downstream.

### Keyword + document-match reranker over a neural cross-encoder

A cross-encoder reranker would improve semantic ranking in a general-purpose setting, but it introduces additional latency and computational overhead per query. In this system, queries explicitly specify the target contract (e.g., "in Contract_11"), which creates a strong structured signal.

The reranker is designed to exploit this structure:

- keyword overlap captures clause-level relevance
- document-match scoring enforces alignment with the explicitly requested document

This is not a simplification but a constraint-aware optimization: it leverages deterministic signals already present in the query to achieve high precision without introducing unnecessary model inference steps.

### Gemini-primary / GPT-4o-fallback generation

The generation layer is structured to separate latency-sensitive development workflows from quality-sensitive production needs.

Gemini 2.5 Flash is used as the primary model due to its low latency and cost efficiency, enabling rapid iteration and evaluation.
GPT-4o serves as a fallback to maintain output reliability when higher-quality reasoning or robustness is required.

This dual-layer design ensures that:

- experimentation remains cost-effective
- system behavior remains stable under variability in model performance

Rather than optimizing for a single axis (cost or quality), the system explicitly balances both through controlled model routing.

### Retrieval-first design (grounding over generation)

The system is intentionally designed to make retrieval the primary source of correctness, with generation acting as a constrained summarization layer rather than a reasoning engine.

Key implications:

- The LLM is not relied upon to infer missing information
- Answers are strictly derived from retrieved context
- Prompting enforces refusal when context is insufficient

This reduces hallucination risk and shifts system reliability to components that are:

- deterministic (retrieval, reranking)
- observable and testable (Precision@k, retrieval logs)

The trade-off is reduced flexibility for open-ended or cross-document reasoning, but this is acceptable given the requirement that hallucinated answers are unacceptable.

---

## Evaluation

Retrieval quality is measured with **Precision@3**: for each test question, a score of 1 is awarded if the ground-truth contract appears in the top-3 retrieved chunks, 0 otherwise. The test set (`evaluation.py`) contains 10 questions spanning contract terms, payment clauses, guarantees, insurance, default, and termination across three contracts (Contract_1, Contract_10, Contract_11).

**Baseline (before fix):** Precision@3 = 0.3 — the retriever consistently surfaced Contract_1 chunks regardless of the query because identical boilerplate produced identical vectors.

**After fix:** Precision@3 expected ≈ 1.0 — document-prefixed embeddings and the document-match reranker together route queries to the correct contract.

---

## Limitations and Future Work

- **FAISS IndexFlatL2** performs exact search and does not scale beyond ~500k vectors. For larger corpora, replace with `IndexIVFFlat` or `IndexHNSW`.
- **No persistence** — the FAISS index is rebuilt from scratch on every run. Adding `faiss.write_index` / `faiss.read_index` calls in `ingest` and a load-if-exists check would eliminate re-ingestion cost.
- **Confidence scoring** is a heuristic keyword overlap ratio, not a calibrated probability. A future version could use the LLM's own log-probabilities or a trained confidence head.
- **Single-document queries only** — the current reranker assumes one contract is named per query. Cross-contract comparison queries (e.g. *"Compare the default clauses in Contract_1 and Contract_11"*) would require multi-target retrieval logic.
- **PyPDF2 extraction quality** degrades on scanned or image-based PDFs. Replacing with `pdfplumber` or `pymupdf` would improve text fidelity on such documents.