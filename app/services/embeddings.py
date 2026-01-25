"""
Embeddings service using sentence-transformers.
Supports multilingual models for French banking documents.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger("embeddings")

# Global model instance (loaded once)
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the embedding model (singleton pattern).
    
    Supported models:
    - paraphrase-multilingual-MiniLM-L12-v2 (default, good FR support)
    - multilingual-e5-small (alternative, excellent quality)
    """
    global _model
    
    if _model is None:
        model_name = settings.embedding_model
        logger.info("loading_embedding_model", model=model_name)
        
        try:
            _model = SentenceTransformer(model_name)
            logger.info("embedding_model_loaded", 
                        model=model_name, 
                        embedding_dim=_model.get_sentence_embedding_dimension())
        except Exception as e:
            logger.error("embedding_model_load_failed", model=model_name, error=str(e))
            raise
    
    return _model


def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding
    
    Returns:
        numpy array of embeddings (n_texts x embedding_dim)
    """
    if not texts:
        return np.array([])
    
    model = get_embedding_model()
    
    logger.debug("generating_embeddings", count=len(texts), batch_size=batch_size)
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    
    return embeddings


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text string to embed
    
    Returns:
        numpy array of embedding (embedding_dim,)
    """
    if not text:
        raise ValueError("Cannot embed empty text")
    
    model = get_embedding_model()
    embedding = model.encode(
        text,
        normalize_embeddings=True
    )
    
    return embedding


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings from the current model."""
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    Since embeddings are L2-normalized, this is just dot product.
    """
    return float(np.dot(embedding1, embedding2))
