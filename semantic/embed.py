import logging
import os
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "google/embeddinggemma-2"
EMBED_DIM = 768
PROMPT_VERSION = "gemma2-asymmetric-v1"

# Lazy-loaded local embedding model
_model: Optional[SentenceTransformer] = None


def _get_device() -> str:
    """Select the best available device: Apple Silicon (MPS) -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model() -> SentenceTransformer:
    """Initialize and return the local EmbeddingGemma 2 model."""
    global _model
    if _model is None:
        device = _get_device()
        logger.info(f"Loading embedding model '{MODEL_NAME}' on {device}...")
        try:
            _model = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(MODEL_NAME, device=device)
        max_len = int(os.environ.get("EMBED_MAX_SEQ_LENGTH", "2048"))
        _model.max_seq_length = max_len
        logger.info(f"EmbeddingGemma 2 max_seq_length set to {max_len}")
    return _model


def format_query(query: str) -> str:
    """Format a query with the documented EmbeddingGemma 2 instruction prefix."""
    cleaned = query.strip().replace("\n", " ")
    if cleaned.startswith("task: search result | query: "):
        return cleaned
    return f"task: search result | query: {cleaned}"


def format_document(text: str) -> str:
    """Format document text for EmbeddingGemma 2: title: {title} | text: {body}"""
    cleaned = text.strip()
    if cleaned.startswith("title: ") and " | text: " in cleaned:
        return cleaned.replace("\n", " ")

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if not lines:
        return "title: none | text: "
    if len(lines) == 1:
        # Single line: use as both or title: none | text: line
        return f"title: {lines[0]} | text: {lines[0]}"
    
    # First line is typically the title
    title = lines[0]
    body = " ".join(lines[1:])
    return f"title: {title} | text: {body}"


def embed_texts(texts: List[str], is_query: bool = False, batch_size: int = 8) -> List[List[float]]:
    """
    Generate 768-dimensional embeddings using google/embeddinggemma-2.

    Args:
        texts: The list of string texts to embed.
        is_query: True for search queries (applies query task prefix).
                  False for documents (applies title/text prefix).
        batch_size: Batch size for encoding (default 8 for 8K context memory safety).

    Returns:
        A list of 768-dimensional float list embeddings normalized to unit length.
    """
    if not texts:
        return []

    model = get_model()

    with torch.inference_mode():
        if is_query:
            formatted = [format_query(t) for t in texts]
            vecs = model.encode(
                formatted,
                batch_size=min(len(formatted), 16),
                truncate_dim=EMBED_DIM,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [v.tolist() for v in vecs]

        # Documents
        formatted = [format_document(t) for t in texts]
        show_progress = len(formatted) > 50
        vecs = model.encode(
            formatted,
            batch_size=batch_size,
            truncate_dim=EMBED_DIM,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [v.tolist() for v in vecs]

