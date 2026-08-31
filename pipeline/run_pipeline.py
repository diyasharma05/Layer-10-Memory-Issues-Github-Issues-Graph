import json
import os
import networkx as nx

from extraction.schema import ContextPack
from extraction.artifact_loader import load_artifacts
from extraction.deterministic_extract import extract_deterministic
from extraction.llm_extract import llm_extract_references
from dedup.artifact_dedup import deduplicate_artifacts
from dedup.entity_resolution import get_entity_mapping, resolve_entities
from dedup.claim_dedup import deduplicate_claims
from graph.build_graph import KnowledgeGraph, build_networkx_graph
from retrieval.embedder import Embedder
from retrieval.vector_index import VectorIndex
from retrieval.retrieve_context import HybridRetriever

def save_json(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([item.model_dump(mode='json') for item in data], f, indent=2)

def main():
    print("=== Layer10 Memory System Pipeline ===")
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    RAW_DATA = os.path.join(BASE_DIR, "data", "raw", "github_issues.csv")
    OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Ingestion
    print("1. Ingesting artifacts...")
    artifacts, raw_dicts = [], []
    for art, raw in load_artifacts(RAW_DATA):
        artifacts.append(art)
        raw_dicts.append(raw)
        
    artifacts = deduplicate_artifacts(artifacts)
    print(f"Loaded {len(artifacts)} unique artifacts.")
    
    all_entities, all_claims, all_evidence = [], [], []
    
    # 3. Extraction
    print("3. Running extraction...")
    for art, raw in zip(artifacts, raw_dicts):
        ent_d, claim_d, ev_d = extract_deterministic(art, raw)
        all_entities.extend(ent_d)
        all_claims.extend(claim_d)
        all_evidence.extend(ev_d)
        
        ent_l, claim_l, ev_l = llm_extract_references(art)
        all_entities.extend(ent_l)
        all_claims.extend(claim_l)
        all_evidence.extend(ev_l)
        
    print(f"Extracted {len(all_entities)} entities, {len(all_claims)} claims, {len(all_evidence)} pieces of evidence.")
    
    # 4. Deduplication
    print("4. Canonicalizing entities and deduplicating claims...")
    entity_mapping = get_entity_mapping(all_entities)
    all_entities = resolve_entities(all_entities, entity_mapping)
    
    for c in all_claims:
        c.subject = entity_mapping.get(c.subject, c.subject)
        c.object = entity_mapping.get(c.object, c.object)
        
    all_claims, all_evidence = deduplicate_claims(all_claims, all_evidence)
    print(f"Post-dedup: {len(all_entities)} entities, {len(all_claims)} claims.")
    
    # 5. Graph
    print("5. Building Graph...")
    kg = KnowledgeGraph()
    kg.init_schema()
    kg.insert_data(all_entities, all_claims, all_evidence, artifacts)
    kg.close()
    
    G = build_networkx_graph(all_entities, all_claims, all_evidence, artifacts)
    nx.write_graphml(G, os.path.join(OUT_DIR, "memory_graph.graphml"))
    
    print("Saving outputs...")
    save_json(all_entities, os.path.join(OUT_DIR, "entities.json"))
    save_json(all_claims, os.path.join(OUT_DIR, "claims.json"))
    save_json(all_evidence, os.path.join(OUT_DIR, "evidence.json"))
    save_json(artifacts, os.path.join(OUT_DIR, "artifacts.json"))
    
    # 6. Retrieval
    print("6. Setting up Retrieval...")
    embedder = Embedder()
    vector_index = VectorIndex()
    claim_texts = [f"{c.subject} {c.predicate} {c.object}" for c in all_claims]
    embeddings = embedder.embed_texts(claim_texts)
    vector_index.add_claims(all_claims, embeddings)
    
    evidence_map = {}
    for ev in all_evidence:
        evidence_map.setdefault(ev.claim_id, []).append(ev)
    all_claims_map = {c.claim_id: c for c in all_claims}
        
    retriever = HybridRetriever(embedder, vector_index, G, evidence_map, all_claims_map)
    
    sample_queries_file = os.path.join(BASE_DIR, "examples", "sample_queries.json")
    if os.path.exists(sample_queries_file):
        with open(sample_queries_file, "r") as f:
            queries = json.load(f)
        print("\n--- Example Retrieval Results ---")
        for q in queries:
            print(f"\nQ: {q}")
            pack = retriever.retrieve(q, top_k_claims=2, graph_radius=1)
            for c in pack.claims:
                print(f"  Claim: {c.subject} {c.predicate} {c.object} (Conf: {c.confidence})")
            for ev in pack.evidence_snippets[:2]:
                print(f"    Evidence [{ev.artifact_id[:4]}]: {ev.text_span}")

if __name__ == "__main__":
    main()
