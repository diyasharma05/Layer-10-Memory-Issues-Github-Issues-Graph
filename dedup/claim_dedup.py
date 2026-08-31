from typing import List, Tuple, Dict
from extraction.schema import Claim, Evidence

def deduplicate_claims(claims: List[Claim], evidence: List[Evidence]) -> Tuple[List[Claim], List[Evidence]]:
    claim_map: Dict[str, Claim] = {}
    id_mapping: Dict[str, str] = {}
    
    for c in claims:
        key = f"{c.subject}::{c.predicate}::{c.object}"
        if key not in claim_map:
            claim_map[key] = c
            id_mapping[c.claim_id] = c.claim_id
        else:
            id_mapping[c.claim_id] = claim_map[key].claim_id
            
    deduped_evidence_map = {}
    for e in evidence:
        canonical_claim_id = id_mapping.get(e.claim_id)
        if not canonical_claim_id:
            continue
            
        e.claim_id = canonical_claim_id
        e_key = f"{e.claim_id}::{e.artifact_id}::{e.text_span}"
        if e_key not in deduped_evidence_map:
            deduped_evidence_map[e_key] = e
            
    return list(claim_map.values()), list(deduped_evidence_map.values())
