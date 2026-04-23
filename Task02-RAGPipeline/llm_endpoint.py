from sentence_transformers import SentenceTransformer
from openai import OpenAI
from google import genai
from log import logger

import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# -------------------------
# Embedding model (local)
# -------------------------
embed_model = SentenceTransformer("all-mpnet-base-v2")

# -------------------------
# LLM clients
# -------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# -------------------------
# Embeddings
# -------------------------
def get_embedding(text):
    emb = embed_model.encode(text, normalize_embeddings=True)
    return np.array(emb)


# -------------------------
# Gemini (primary)
# -------------------------
def generate_with_gemini(prompt):
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

        text = response.text.strip()

        if not text:
            raise ValueError("Empty response from Gemini")

        return text

    except Exception as e:
        logger.warning(f"Gemini failed: {e}")
        raise


# -------------------------
# GPT-4o (fallback)
# -------------------------
def generate_with_openai(prompt, model):
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Follow instructions strictly"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# -------------------------
# Unified interface
# -------------------------
def generate_answer(prompt, model="gpt-4o"):
    """
    Primary: Gemini 2.5 Flash (free tier)
    Fallback: GPT-4o (paid, higher quality)
    """

    try:
        logger.info("Using Gemini 2.5 Flash")
        return generate_with_gemini(prompt)

    except Exception:
        logger.warning("Falling back to GPT-4o")

        try:
            return generate_with_openai(prompt, model)

        except Exception as e:
            logger.error(f"Both models failed: {e}")
            return "Model failed to generate response."