import math
import os
from typing import Any

from google import genai


EMBEDDING_MODEL = "text-embedding-004"

_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_embedding(text: str) -> list[float]:
    """Generate a Gemini embedding for a piece of transcript text."""
    if not text.strip():
        return []

    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def search_chunks(
    query_embedding: list[float],
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the most relevant transcript chunks."""
    results = []

    for chunk in chunks:
        embedding = chunk.get("embedding", [])

        score = cosine_similarity(query_embedding, embedding)

        results.append({
            **chunk,
            "score": score,
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]
def chunk_transcript(
    text: str,
    words_per_minute: int = 150,
    chunk_minutes: int = 1,
) -> list[dict[str, Any]]:
    """Split transcript into approximately 1-minute searchable chunks."""
    words = text.split()

    if not words:
        return []

    words_per_chunk = words_per_minute * chunk_minutes
    chunks = []

    for start in range(0, len(words), words_per_chunk):
        chunk_words = words[start:start + words_per_chunk]

        start_time = start / words_per_minute
        end_time = (start + len(chunk_words)) / words_per_minute

        chunks.append({
            "text": " ".join(chunk_words),
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
        })

    return chunks
