"""
Utility functions for managing the Chroma vector database and embeddings.
Handles setup of persistent storage and embedding configuration.
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Initialize a persistent ChromaDB client (stored under /models/chroma_db)
client = chromadb.PersistentClient(path="models/chroma_db")

# Use a lightweight, sentence-transformer embedding model for text encoding
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection(name="knowledge_base"):
    """
    Create or load a Chroma collection with the specified name.
    Returns a collection object ready for document insertion and querying.
    """
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function
    )
