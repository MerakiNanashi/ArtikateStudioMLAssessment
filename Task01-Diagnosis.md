## Diagnosis Log 1 — Hallucinated Pricing
---

### Initial hypothesis

Incorrect pricing suggests either:

- stale knowledge (knowledge cutoff),
- missing/incorrect retrieval,
- prompt not enforcing grounding (no RAG/Validation),
- or randomness (temperature).

---

### What I would investigate first

1. Check if response cite a source via logs:
    If no source/retreival from pricing db is detected in the log, the most likely culprit is missing/broken retreival.

2. Compare answers vs. known pricing API / DB:
    If answers vary across runs → temperature issue.
    If consistently wrong → retrieval or knowledge issue.

3. Inspect prompt for any weakness:
    Check if the prompt is enforcing strict reliance on retreived info, and adding a fallback statement in case of insufficient info. If not → weak prompt. 

4. Check for Post Validation:
    No Post Validation means the system is allowing for hallucinated pricing to pass through.

---

### What I ruled out
- Temperature issue: hallucinated pricing is usually consistent, not random.
- Knowledge cutoff: pricing is dynamic; even a correct cutoff-aware model should defer, not fabricate.
→ If the model answers confidently, it is not a cutoff limitation but a grounding failure.

---

### Root Cause

#### Primary cause (No/Weak RAG Implementation)
- Either the pricing isn't being retreived at all or incorrect pricing is being retreived (poor indexing/embedding)

#### Secondary cause (No Validation & Weak Prompt)
- No post validation or enforcement of grounding, letting the model hallucinate in case of insufficient data.

---

#### How to distinguish definitively
- Disable retrieval → if answers remain similar → model hallucination.
- Log retrieved chunks:
    If empty → retrieval failure
    If irrelevant → embedding/search issue
- Force deterministic decoding (temperature = 0):
    If still wrong → not randomness

---

#### Concrete fix

1. Enforce grounded answering in prompt
If pricing information is not present in the provided context, respond:


        "I don’t have access to current pricing information."
        Do not guess.


2. Improve retrieval
    Structured pricing index (not raw text)
    Use hybrid search (keyword + vector)
    Add freshness (cache invalidation / DB lookup)
3. Add validation layer
    Post-check: compare extracted price against DB
    Reject if mismatch

---

## Diagnosis Log 2 — Language Switching

---

### Initial Hypothesis

The model is not consistently adhering to the user's input language due to:
- implicit English bias in the system prompt,
- lack of explicit output-language constraints,
- and potential interference from retrieval/context language.

This suggests a **prompt alignment issue**, not a model capability issue.

---

### System-Level Mechanism

In a standard system prompt + user message architecture:

1. **System prompt dominates behavior**
   - Typically written in English (e.g., "You are a helpful assistant").
   - Acts as a persistent prior across all turns.

2. **User input is treated as content, not instruction**
   - Unless explicitly elevated, the model does not treat "language choice" as a constraint.
   - The model optimizes for the most statistically likely continuation → defaults to English.

3. **Retrieval-Augmented Generation (RAG) bias**
   - Retrieved documents are often in English.
   - The model aligns generation with the language of the context window, not the user.

4. **Token-level probability dynamics**
   - English tokens dominate training distribution.
   - When constraints are weak, decoding drifts toward English.

5. **Conversation memory contamination**
   - If earlier turns were in English, the model maintains language consistency across turns—even if the user switches language later.

---

### What I Investigated

1. **System prompt**
   - Checked for explicit language control instructions (none found or too weak).

2. **Prompt hierarchy**
   - Verified whether user language is reinforced or overridden.

3. **Retrieved context**
   - Confirmed that supporting documents are primarily English.

4. **Decoding parameters**
   - Checked temperature/top-p (not causal here, but can amplify drift).

5. **Conversation logs**
   - Observed switching patterns (e.g., Hindi → English after retrieval-heavy queries).

---

### What I Ruled Out

- **Model limitation**  
  GPT-4o is strongly multilingual; failure is not due to capability.

- **Randomness / temperature**  
  Behavior is systematic and reproducible.

- **Encoding or input pipeline issues**  
  User language is correctly passed and tokenized.

---

### Root Cause

**Prompt design flaw:**
- Language alignment is treated as an *implicit preference* instead of an *explicit constraint*.
- No instruction enforces:
  - language detection,
  - strict output matching,
  - or resistance to retrieval-language bias.

---

### Concrete Fix (Prompt Change)

Replace or augment the system prompt with:

```text
You are a multilingual customer support assistant.

LANGUAGE POLICY (STRICT):
1. Detect the language of the user's latest message.
2. Respond ONLY in that language.
3. This rule overrides all other stylistic or formatting preferences.
4. Do NOT switch to English unless the user explicitly does so.
5. If retrieved context is in a different language, translate internally but respond in the user's language.
6. Maintain the same language consistently across the entire response.

If the user's language cannot be determined with high confidence, ask a clarification question in a neutral format.
```

#### Fix 2:

If Prompt Fix doesn't work:

1. Add explicit language detection step (e.g., fastText or compact classifier)
2. Pass detected language as a structured variable in the LLM API params.
3. Enforce via templating:
```text 
Respond in {{user_language}}
```
4. Add output validator to reject mismatched language responses

---

Why this works
- Makes language a hard constraint, not preference
- Overrides retrieval-language bias
- Explicitly prevents fallback to English

---

## Diagnosis Log 3 — Latency Degradation (1.2s → 8–12s)

---

### Initial hypothesis

Latency increasing over time without any code/model change suggests:

- growing input size (conversation history / retrieved context),
- retrieval system slowdown (vector DB scaling),
- or infrastructure-level contention (rate limits, queuing).

This indicates a **load-dependent or data-dependent failure**, not a logic bug.

---

### What I would investigate first

1. Check token usage per request via logs:
   - Track `prompt_tokens` and `completion_tokens` over time.
   - Inspect:
     - conversation history length
     - number of retrieved chunks (top-k)
   - If tokens increased significantly → strong signal of context bloat.

2. Break down latency by pipeline stage:
   - Retrieval time vs LLM inference vs network/API time.
   - If LLM time dominates → token growth issue.
   - If retrieval time dominates → vector DB issue.

3. Inspect retrieval system performance:
   - Query latency vs corpus size
   - Index type (flat vs ANN)
   - Growth in documents/chunks

4. Check API/provider metrics:
   - Rate limiting / throttling
   - Queue times / retries
   - Throughput vs request volume

---

### What I ruled out

- Prompt/model change: explicitly unchanged.
- Random fluctuation: degradation is gradual and consistent.
- Single-point bug: pattern correlates with increased usage.

---

### Root Cause

#### Primary cause (Unbounded Context Growth)
- conversation history accumulating over time
- increasing retrieved chunks per query
- no token budget enforcement

→ Leads to higher token count and slower LLM inference.

#### Secondary cause (Retrieval Scaling)
- larger corpus → slower similarity search
- inefficient indexing or high top-k retrieval

---

### How to distinguish definitively

- Track tokens/request over time:
  - If increasing → confirms context growth
- Measure stage-wise latency:
  - If LLM time ↑ → context issue
  - If retrieval time ↑ → DB/index issue
- Fix token cap temporarily:
  - If latency drops → confirms root cause

---

### Concrete fix

1. Context management (primary fix)
   - Cap total tokens (e.g., 2k–4k)
   - Keep only last N turns (3–5)
   - Add rolling summarization

2. Retrieval optimization
   - Reduce top_k (e.g., 10 → 3–5)
   - Add re-ranking instead of expanding results
   - Apply metadata filtering before vector search

3. Token guardrails
   - Enforce max input/output tokens
   - Trim or reject oversized requests

4. Caching
   - Cache frequent queries and retrieval results

---

### Alt fixes (if primary fix insufficient)

#### 1. Vector DB bottleneck
- Switch to ANN index (HNSW/IVF)
- Add sharding or partitioning
- Periodically rebuild index

#### 2. API rate limiting / queuing
- Add client-side rate limiting
- Batch or queue requests internally
- Upgrade API tier or add fallback model

#### 3. Infrastructure contention
- Scale workers horizontally
- Separate retrieval and inference services
- Use async request handling

---

### Why this is the most likely issue

Latency degradation correlates with **user growth over time**, which directly increases:
- context size,
- retrieval load,
- and system concurrency.

This pattern is characteristic of **missing constraints in system design**, not model failure.

---

## Post-Mortem Summary

The chatbot issues were caused by a combination of data handling gaps and scaling effects, rather than a failure of the underlying AI model.

First, incorrect pricing responses occurred because the system was not reliably pulling real pricing data at runtime. Instead, the model attempted to “fill in the blanks,” leading to confident but incorrect answers. This has been addressed by enforcing strict use of verified data sources and preventing the model from guessing.

Second, the chatbot occasionally replied in the wrong language because it was not explicitly instructed to match the user’s language. We updated the system instructions so responses always follow the user’s input language, regardless of internal data sources.

Third, response times slowed significantly as usage increased. This was due to the system processing more historical conversation and data per request over time, increasing computational load. We fixed this by limiting how much past information is used and optimizing data retrieval.

Overall, these fixes improve accuracy, consistency, and performance without changing the core AI model.

-----