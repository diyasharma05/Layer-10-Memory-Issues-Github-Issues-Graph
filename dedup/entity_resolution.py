from typing import List, Dict
from extraction.schema import Entity
import numpy as np

def cosine_similarity_np(embeddings):
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1e-10, norm) # Handle zero norm
    normed = embeddings / norm
    return np.dot(normed, normed.T)

def get_entity_mapping(entities: List[Entity], threshold: float = 0.9) -> Dict[str, str]:
    """Returns a dict mapping old entity IDs to canonical entity IDs."""
    mapping = {}
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        # Fallback
        for e in entities:
            mapping[e.id] = e.id
        return mapping
        
    type_groups: Dict[str, List[Entity]] = {}
    for e in entities:
        type_groups.setdefault(e.type, []).append(e)
        
    for t_type, t_entities in type_groups.items():
        if t_type == "Issue":
            # Don't merge distinct issues
            for e in t_entities:
                mapping[e.id] = e.id
            continue
            
        names = [e.name for e in t_entities]
        try:
            embeddings = model.encode(names)
            sim_matrix = cosine_similarity_np(embeddings)
        except Exception:
             for e in t_entities:
                mapping[e.id] = e.id
             continue
             
        visited = set()
        for i in range(len(t_entities)):
            if i in visited:
                continue
            canonical_id = t_entities[i].id
            mapping[canonical_id] = canonical_id
            visited.add(i)
            
            similar_indices = np.where(sim_matrix[i] >= threshold)[0]
            for j in similar_indices:
                if j not in visited:
                    mapping[t_entities[j].id] = canonical_id
                    visited.add(j)
                    
    return mapping

def resolve_entities(entities: List[Entity], mapping: Dict[str, str]) -> List[Entity]:
    """Filters duplicate entities given mappings."""
    canonical_ids = set(mapping.values())
    seen = set()
    resolved = []
    
    for e in entities:
        canonical = mapping.get(e.id, e.id)
        if canonical not in seen:
            e.id = canonical
            resolved.append(e)
            seen.add(canonical)
            
    return resolved
