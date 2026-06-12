from typing import List

import ollama


class EmbeddingsManager:
    """Manages embedding generation for text chunks via Ollama."""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            resp = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(resp["embedding"])
        return embeddings

    def get_query_embedding(self, query: str) -> List[float]:
        return self.get_embeddings([query])[0]
