import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
from typing import List, Tuple
from extraction.schema import Claim

class VectorIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.claims: List[Claim] = []
        self.index = None
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)

    def add_claims(self, claims: List[Claim], embeddings: np.ndarray):
        if not FAISS_AVAILABLE or not len(claims) or embeddings.shape[0] == 0:
            self.claims.extend(claims)
            return
            
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.claims.extend(claims)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Claim, float]]:
        if not FAISS_AVAILABLE or not self.claims or self.index is None or self.index.ntotal == 0:
            return []
            
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.claims):
                results.append((self.claims[idx], float(distances[0][i])))
                
        return results
