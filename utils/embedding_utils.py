"""
Utility for generating text embeddings using SentenceTransformers.
Provides a reusable function to encode text into dense vector representations.
"""

from sentence_transformers import SentenceTransformer

# Load a lightweight transformer model for text embedding
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    """
    Convert a list of text strings into numerical embeddings.
    Returns a NumPy array of vector representations.
    """
    return embed_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )
