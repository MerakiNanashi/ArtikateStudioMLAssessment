# DESIGN.md — Ticket Classification System

## 1. Problem Overview

We are tasked with building a real-time classifier for customer support tickets into five categories:

- billing  
- technical_issue  
- feature_request  
- complaint  
- other  

### Constraints
- Training data: 1,000 labeled examples (200 per class)
- Inference latency: **< 500ms per ticket**
- Throughput: **1 ticket every 30 seconds (~2,880/day)**
- Infrastructure: **single CPU machine**
- Evaluation: accuracy, per-class F1, confusion matrix
- Reliability: deterministic outputs + strict label space

This is explicitly a **production-oriented system design problem**, not just a modeling task :contentReference[oaicite:0]{index=0}.

---

## 2. Approach Selection

### Chosen Approach: Fine-tuned Small Transformer (DistilBERT)

We fine-tune a lightweight transformer (`DistilBERT`) instead of using an LLM API.

---

## 3. Why Fine-Tuning Over Prompt Engineering

### 3.1 Latency Constraints

| Approach | Expected Latency |
|----------|----------------|
| DistilBERT (CPU) | ~50–150 ms |
| LLM API (GPT / Gemini) | ~800ms – 2s+ |

- API-based approaches violate the **500ms hard constraint**
- Network latency alone (~200–400ms) makes compliance unreliable

---

### 3.2 Throughput and Stability

- Required throughput is low (~1 request / 30 sec), but **latency must be deterministic**
- LLM APIs introduce:
  - rate limits
  - cold starts
  - unpredictable tail latency (p95 spikes)

Local inference:
- Stable performance
- No dependency on external systems

---

### 3.3 Cost Trade-off

| Approach | Cost |
|----------|------|
| DistilBERT (local) | ~0 |
| LLM API | recurring (per token) |

Even at low volume (~2.8k/day), API usage creates unnecessary operational cost.

---

### 3.4 Data Regime Consideration

We only have **1,000 samples**, which is relatively small.

- LLM few-shot prompting performs well in low-data regimes
- However:
  - prompts must be engineered carefully
  - outputs require parsing + validation
  - consistency issues arise across runs

Fine-tuning still works because:
- Classes are **coarse-grained**
- Language patterns are repetitive (support tickets)
- Transfer learning from pretrained BERT compensates for small data

---

## 4. Data Strategy

### 4.1 Initial Dataset Limitation

The provided dataset (1,000 samples) is:
- small
- potentially narrow in linguistic diversity
- insufficient to capture real-world variation (typos, multilingual input, informal phrasing)

---

### 4.2 Heuristic Data Augmentation

To mitigate this, additional synthetic examples were generated using:
- paraphrasing templates
- controlled LLM generation (with strict label conditioning)

Example transformations:
- "I was charged twice" → "double charged", "duplicate billing issue"
- "app crashes" → "application closes unexpectedly"

#### Why this approach:
- Improves robustness to lexical variation
- Reduces overfitting to specific phrasing
- Expands decision boundaries

#### Trade-offs:
- Synthetic data may introduce **distribution mismatch**
- LLM-generated text can be **too clean / less noisy than real data**
- Risk of reinforcing biases in prompt templates

Mitigation:
- Keep synthetic data ratio < 2x original dataset
- Manually inspect a subset
- Keep evaluation set strictly human-written

---

## 5. Model Architecture

### 5.1 Base Model
- `DistilBERT` (66M parameters)

### 5.2 Classification Head
- Linear layer over `[CLS]` embedding
- Softmax over 5 classes

### 5.3 Training Details
- Loss: Cross-entropy
- Optimizer: AdamW
- Epochs: 3–5
- Batch size: 16–32
- Max sequence length: 128 tokens

---

## 6. Inference Pipeline

1. Input ticket text
2. Tokenization (HuggingFace tokenizer)
3. Forward pass through model
4. Softmax → class probabilities
5. Argmax → predicted label

### Optimization for CPU
- Torch `no_grad()` inference
- Model in evaluation mode
- Optional:
  - ONNX / TorchScript for further speedup
  - Quantization (INT8) if needed

---

## 7. Evaluation Strategy

### 7.1 Metrics
- Accuracy
- Per-class Precision, Recall, F1
- Confusion Matrix

### 7.2 Test Set
- ≥100 samples
- Manually written or verified (NOT LLM-generated)

---

## 8. Observed Failure Modes

### Most Confused Classes
Typically:
- `complaint` vs `technical_issue`
- `feature_request` vs `other`

### Why this happens

#### Complaint vs Technical Issue
- Overlapping language:
  - "this app is terrible" (complaint)
  - "this app crashes" (technical_issue)
- Both express dissatisfaction

#### Feature Request vs Other
- Ambiguity:
  - "can you add dark mode?" (feature_request)
  - "is dark mode available?" (other)

---

## 9. Improvements for Class Separation

### Additional Signals
- Metadata (if available):
  - user intent markers
  - ticket category history
- Keyword features:
  - "error", "bug" → technical
  - "please add", "would like" → feature_request

### Better Data
- More real-world tickets (not synthetic)
- Hard negative examples (borderline cases)
- Multi-label annotation (some tickets are inherently ambiguous)

---

## 10. Latency Validation

A test enforces:
- Batch of 20 predictions
- Total runtime < 500ms

This ensures:
- production readiness
- compliance with system constraint

---

## 11. Trade-offs Summary

| Dimension | Fine-tuned Model | LLM Prompting |
|----------|----------------|--------------|
| Latency | ✅ Fast (<150ms) | ❌ Slow (>800ms) |
| Cost | ✅ Free (local) | ❌ Recurring |
| Consistency | ✅ Deterministic | ❌ Variable |
| Setup | ❌ Training required | ✅ Minimal |
| Data efficiency | ⚠️ Moderate | ✅ Strong |

---

## 12. Future Improvements

### 12.1 Better Fine-Tuning
- Larger pretrained models (e.g., `MiniLM`, `DeBERTa-v3-small`)
- Hyperparameter tuning (learning rate, scheduler)
- Class weighting to handle imbalance

---

### 12.2 Data Improvements
- Collect real production tickets
- Active learning loop:
  - flag low-confidence predictions
  - manually relabel
- Continuous retraining

---

### 12.3 Hybrid System (Best of Both Worlds)
- Use DistilBERT for fast classification
- Route low-confidence cases to LLM for fallback

---

### 12.4 Model Compression
- Quantization (INT8)
- Knowledge distillation
- ONNX runtime deployment

---

### 12.5 Multilingual Support
- Current system assumes English
- Upgrade path:
  - `mBERT` or `XLM-R`
  - or language detection + routing

---

## 13. Limitations

- Synthetic data may not reflect real-world noise
- Single-label classification oversimplifies tickets
- No context (conversation history ignored)
- Edge cases remain ambiguous even for humans

---

## 14. Conclusion

Given strict latency (500ms), CPU-only deployment, and moderate data size, a fine-tuned small transformer provides the best balance of:

- speed  
- cost  
- reliability  

While LLM-based prompting is more flexible and data-efficient, it fails the **hard real-time constraint**, making it unsuitable for this system.

The current design is intentionally simple, but extensible toward hybrid and production-scale improvements.