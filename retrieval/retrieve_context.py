from typing import List, Dict
import networkx as nx

from extraction.schema import Claim, Evidence, ContextPack
from retrieval.embedder import Embedder
from retrieval.vector_index import VectorIndex
from graph.graph_queries import get_subgraph

class HybridRetriever:
    def __init__(self, embedder: Embedder, vector_index: VectorIndex, graph: nx.DiGraph, evidence_map: Dict[str, List[Evidence]], all_claims_map: Dict[str, Claim]):
        self.embedder = embedder
        self.vector_index = vector_index
        self.graph = graph
        self.evidence_map = evidence_map
        self.all_claims_map = all_claims_map

    def _rank_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        # Rank by confidence descending
        return sorted(evidence_list, key=lambda e: e.confidence, reverse=True)

    def retrieve(self, query: str, top_k_claims: int = 5, graph_radius: int = 1) -> ContextPack:
        q_emb = self.embedder.embed_texts([query])
        
        claim_results = self.vector_index.search(q_emb, top_k=top_k_claims)
        
        retrieved_claims_dict = {}
        for claim, sim in claim_results:
            retrieved_claims_dict[claim.claim_id] = claim
            
        # Optional Expansion
        expanded_claim_ids = set()
        for claim_id in retrieved_claims_dict.keys():
            subg = get_subgraph(self.graph, claim_id, radius=graph_radius)
            for node, attrs in subg.nodes(data=True):
                if attrs.get("type", "") == "Claim" and node not in retrieved_claims_dict:
                    expanded_claim_ids.add(node)
                    
        for c_id in expanded_claim_ids:
            if c_id in self.all_claims_map:
                retrieved_claims_dict[c_id] = self.all_claims_map[c_id]
        
        final_claims = list(retrieved_claims_dict.values())
        final_evidence = []
        
        for c in final_claims:
            evs = self.evidence_map.get(c.claim_id, [])
            final_evidence.extend(evs)
            
        ranked_evidence = self._rank_evidence(final_evidence)
        
        return ContextPack(claims=final_claims, evidence_snippets=ranked_evidence)
