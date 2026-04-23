SYSTEM_PROMPT_V1 = """
You are a legal QA assistant.

Strict rules:
- Answer ONLY from provided context
- Cite sources with document name and page
- If answer not found, say: "Insufficient context to answer"
- Do NOT hallucinate
"""

def build_prompt(question, contexts):
    context_text = "\n\n".join([
        f"[{c['document']} - Page {c['page']}]\n{c['text']}"
        for c in contexts
    ])

    return f"""
Context:
{context_text}

Question:
{question}

Answer with citations.
"""