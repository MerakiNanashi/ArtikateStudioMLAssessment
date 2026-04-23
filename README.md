# Setup Guide — Legal Contract RAG System (For folder Task02-RAGPipeline)
---

## Prerequisites

- Python 3.9 or higher
- `pip` package manager
- An OpenAI API key (for GPT-4o fallback generation)
- A Google Gemini API key (for primary generation, free tier)
- A directory of PDF contracts to query

---

## 1. Clone / Obtain the Project

Place all project files in a single directory:

```
project/
├── config.yaml
├── main.py
├── evaluation.py
├── rag.py
├── helper.py
├── llm_endpoint.py
├── prompt.py
├── schema.py
├── log.py
└── data/
    └── Contracts_30_English/
        ├── Contract_1.pdf
        ├── Contract_2.pdf
        └── ...
```

The `data/` directory and the subdirectory name are configured in `config.yaml` (see step 4).

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install the packages directly:

```bash
pip install \
  faiss-cpu \
  numpy \
  sentence-transformers \
  openai \
  google-generativeai \
  PyPDF2 \
  pydantic \
  pyyaml \
  python-dotenv
```

> **Note:** Use `faiss-gpu` instead of `faiss-cpu` if you have a CUDA-capable GPU and want faster index search on large corpora.

The `sentence-transformers` package will download the `all-mpnet-base-v2` model (~420 MB) on first run and cache it locally under `~/.cache/torch/sentence_transformers/`.

---

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

`llm_endpoint.py` calls `load_dotenv()` at import time, so these will be picked up automatically. Alternatively, export them in your shell:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
```

---

## 5. Configure `config.yaml`

```yaml
embedding_model: text-embedding-3-large   # reserved / informational; local model is used
llm_model: gpt-4o                         # GPT-4o fallback model name

chunk_size: 512                           # reserved for future fixed-size chunking
chunk_overlap: 64                         # reserved for future fixed-size chunking

top_k: 5                                  # number of FAISS nearest-neighbour results
rerank_k: 3                               # number of chunks passed to the LLM after reranking

faiss_index_path: "faiss.index"           # reserved for future index persistence
data_path: "data"                         # relative path to the data root directory
subdata_path: "Contracts_30_English"      # subdirectory inside data_path containing PDFs
```

Adjust `data_path` and `subdata_path` to match where your PDF contracts live.

---

## 6. Add Your PDF Documents

Place your PDF contracts in the configured data directory:

```
data/
└── Contracts_30_English/
    ├── Contract_1.pdf
    ├── Contract_2.pdf
    └── ...
```

The directory is scanned recursively for `*.pdf` files. Each page is extracted and chunked separately, so multi-page contracts are handled automatically.

---

## 7. Run the Interactive CLI

```bash
python main.py
```

On first run, the pipeline will:
1. Load and extract text from all PDFs (this can take 30–60 seconds for large corpora).
2. Build the FAISS index (embeddings are computed locally via `all-mpnet-base-v2`).
3. Enter a REPL loop for question answering.

Example session:

```
Enter question: What is the financing term in Contract_11?

ANSWER:
 According to Contract_11.pdf (Page 1), the financing period is up to 60 months,
 unless otherwise agreed upon by the parties [Contract_11.pdf - Page 1].

CONFIDENCE: 0.72

SOURCES:
{'document': 'Contract_11.pdf', 'page': 1, 'chunk': 'CLAUSE TWO – TERM: ...'}
```

Press `Ctrl+C` or `Ctrl+D` to exit.

---

## 8. Run the Evaluation

To benchmark retrieval quality (Precision@3) against the built-in 10-question test set:

```bash
python evaluation.py
```

This ingests the pipeline, runs all test questions, prints per-question results, and outputs a final `Precision@3` score. Expected output with the fixed `rag.py`:

```
Final Precision@3: 1.0
```

---

## Project Structure Reference

| File | Purpose |
|---|---|
| `main.py` | Interactive CLI entry point |
| `evaluation.py` | Offline Precision@3 evaluation harness |
| `rag.py` | Core RAG pipeline (chunking, ingestion, retrieval, reranking, generation) |
| `helper.py` | PDF loading and config utilities |
| `llm_endpoint.py` | Embedding model and LLM clients (Gemini + OpenAI) |
| `prompt.py` | System prompt and prompt builder |
| `schema.py` | Pydantic request/response models |
| `log.py` | Shared logger |
| `config.yaml` | Runtime configuration |
| `.env` | API keys (not committed to version control) |

---

## Troubleshooting

**`No documents loaded` error**
Check that `data_path` and `subdata_path` in `config.yaml` resolve to a directory that exists relative to the project root and contains at least one `.pdf` file.

**Empty or garbled text from PDFs**
PyPDF2 struggles with scanned or image-based PDFs. Install `pdfplumber` and update `helper.py` to use it as an alternative extractor.

**`OPENAI_API_KEY` / `GEMINI_API_KEY` not found**
Ensure `.env` exists in the same directory you run `python main.py` from, or that the keys are exported in your shell environment.

**Slow first run**
The `all-mpnet-base-v2` model is downloaded on first use (~420 MB). Subsequent runs use the local cache and start much faster.

**`faiss` import error on Apple Silicon**
Install via conda instead of pip: `conda install -c conda-forge faiss-cpu`.


---

# Setup Guide — Ticket Classification (Section 03)

---

## Overview

This project implements a real-time customer support ticket classifier under strict production constraints defined in the assessment :contentReference[oaicite:0]{index=0}.

The system classifies tickets into:
- billing  
- technical_issue  
- feature_request  
- complaint  
- other  

The solution uses a fine-tuned DistilBERT model optimized for:
- low latency (<500ms on CPU)  
- deterministic outputs  
- local inference (no API dependency)  

---

## Project Structure

project/  
├── data/  
│ └── customer_support_tickets_200k.csv  
├── fine_tuned_model_v1/  
├── main.ipynb     
└── README.md

---

## Set Up:

1.  Install Dependencies

pip install torch transformers scikit-learn pandas numpy tqdm jupyter
or 
Uncomment line 1 in main.ipynb

2. Running the Notebook
Step 1 — Launch Jupyter

3. Step 2 — Execute Cells Sequentially from code block 1-3, skip code block 4, then the rest sequentially again.

Note: *No need to generate labelled data again.
