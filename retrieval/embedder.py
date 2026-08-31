from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.enabled = True
        except Exception as e:
            print(f"Warning: SentenceTransformer not available. {e}")
            self.enabled = False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self.enabled or not texts:
            return np.array([])
        return self.model.encode(texts)
