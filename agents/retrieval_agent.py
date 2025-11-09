"""
Handles storage and retrieval of text and image data from the Chroma vector database.
"""

from utils.chroma_utils import get_collection


class RetrievalAgent:
    def __init__(self, collection_name="knowledge_base"):
        # Connect to or create a ChromaDB collection
        self.collection = get_collection(collection_name)

    def add_documents(self, ids, texts, metadatas):
        """
        Add text or image entries with metadata to the vector store.
        """
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def query(self, query_text, n_results=5):
        """
        Retrieve the most relevant documents for a given query.
        """
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        hits = []

        # Format the query results for downstream agents
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            hits.append({"text": doc, "metadata": meta})

        return hits
