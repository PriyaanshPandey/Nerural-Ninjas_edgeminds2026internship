import os
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingsManager:
    """Manages embedding generation for text chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the embedding model."""
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy loading of the model to optimize start times."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def get_query_embedding(self, query: str) -> List[float]:
        """Generates embedding for a single search query."""
        return self.get_embeddings([query])[0]
