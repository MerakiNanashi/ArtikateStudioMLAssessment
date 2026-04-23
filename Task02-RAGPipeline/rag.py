import faiss
import numpy as np
from helper import load_pdfs
from llm_endpoint import get_embedding, generate_answer
from prompt import build_prompt
from log import logger
import re

_embed_model = None  # global singleton

class RAGPipeline:

    def __init__(self, config):
        self.config = config
        self.index = None
        self.metadata = []

    # -------------------------
    # 1. Legal-aware chunking
    # -------------------------
    def chunk_text(self, text):
        pattern = r"(CLAUSE\s+\w+.*?:|SECTION\s+\d+.*?:|ARTICLE\s+\d+.*?:)"
        splits = re.split(pattern, text, flags=re.IGNORECASE)

        chunks = []
        current = ""

        for part in splits:
            if re.match(pattern, part, flags=re.IGNORECASE):
                if current.strip():
                    chunks.append(current.strip())
                current = part
            else:
                current += " " + part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # -------------------------
    # 2. Ingestion
    # -------------------------
    def ingest(self):
        docs = load_pdfs(self.config["data_path"])

        if not docs:
            raise ValueError("No documents loaded")

        embeddings = []

        for doc in docs:
            chunks = self.chunk_text(doc["text"])

            for chunk in chunks:
                if not chunk.strip():
                    continue

                # FIX 1: Prepend document name (without .pdf) to each chunk
                # so the embedding encodes document identity alongside content.
                doc_label = doc["document"].replace(".pdf", "")
                enriched_chunk = f"{doc_label}: {chunk}"

                emb = get_embedding(enriched_chunk)
                embeddings.append(emb)

                self.metadata.append({
                    "text": chunk,           # original text kept for prompts
                    "enriched": enriched_chunk,
                    "page": doc["page"],
                    "document": doc["document"]
                })

        if not embeddings:
            raise ValueError("No embeddings generated")

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        logger.info(f"FAISS index built with {len(embeddings)} vectors")

    # -------------------------
    # 3. Retrieval
    # -------------------------
    def retrieve(self, question):
        # FIX 2: The query already contains the contract name (e.g. "Contract_11"),
        # which now aligns with the enriched chunk embeddings.
        q_emb = get_embedding(question)

        D, I = self.index.search(
            np.array([q_emb]).astype("float32"),
            self.config["top_k"]
        )

        results = [self.metadata[i] for i in I[0]]

        logger.info(f"Top-{self.config['top_k']} retrieved chunks:")
        for i, r in enumerate(results):
            logger.info(f"{i+1}. {r['document']} (p{r['page']}) -> {r['text'][:120]}")

        return results

    # -------------------------
    # 4. Hybrid-lite reranking
    # -------------------------
    def extract_target_doc(self, question):
        """Extract the contract filename explicitly mentioned in the question."""
        match = re.search(r"Contract_(\d+)", question, re.IGNORECASE)
        if match:
            return f"Contract_{match.group(1)}.pdf"
        return None

    def keyword_score(self, question, text):
        q_words = set(re.findall(r"\w+", question.lower()))
        t_words = set(re.findall(r"\w+", text.lower()))
        return len(q_words.intersection(t_words))

    def rerank(self, question, chunks):
        # FIX 3: Heavy document-match bonus when the query names a specific contract.
        target_doc = self.extract_target_doc(question)

        def score(chunk):
            kw = self.keyword_score(question, chunk["text"])
            doc_match_bonus = 100 if (target_doc and chunk["document"] == target_doc) else 0
            return kw + doc_match_bonus

        ranked = sorted(chunks, key=score, reverse=True)

        logger.info("After reranking:")
        for i, r in enumerate(ranked[:self.config["rerank_k"]]):
            logger.info(f"{i+1}. {r['document']} (p{r['page']}) -> score={score(r)}")

        return ranked[:self.config["rerank_k"]]

    # -------------------------
    # 5. Confidence scoring
    # -------------------------
    def compute_confidence(self, question, chunks):
        if not chunks:
            return 0.0

        scores = [self.keyword_score(question, c["text"]) for c in chunks]
        avg_score = sum(scores) / len(scores)

        return min(1.0, avg_score / 5 + 0.3)

    # -------------------------
    # 6. Query pipeline
    # -------------------------
    def query(self, question):
        retrieved = self.retrieve(question)
        reranked = self.rerank(question, retrieved)

        prompt = build_prompt(question, reranked)
        answer = generate_answer(prompt, self.config["llm_model"])

        if "Insufficient context" in answer:
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.0
            }

        confidence = self.compute_confidence(question, reranked)

        sources = [
            {
                "document": c["document"],
                "page": c["page"],
                "chunk": c["text"]
            }
            for c in reranked
        ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }