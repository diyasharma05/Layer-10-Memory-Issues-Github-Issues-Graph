from typing import List, Tuple
from extraction.schema import Artifact, Entity, Claim, Evidence
import hashlib

def generate_id(*args) -> str:
    return hashlib.sha256("_".join(str(a) for a in args).encode("utf-8")).hexdigest()[:16]

def extract_deterministic(artifact: Artifact, raw_data: dict) -> Tuple[List[Entity], List[Claim], List[Evidence]]:
    entities = []
    claims = []
    evidence = []
    
    # Entities
    author_id = f"user_{artifact.author}"
    entities.append(Entity(id=author_id, type="Person", name=artifact.author))
    
    if artifact.type == "issue":
        issue_num = raw_data.get('number', artifact.artifact_id)
        issue_id = f"issue_{issue_num}"
        entities.append(Entity(id=issue_id, type="Issue", name=raw_data.get("title", "Unknown Title")))
        
        # Author claim
        c_author = Claim(
            claim_id=generate_id('created_by', issue_id, author_id),
            subject=issue_id,
            predicate="created_by",
            object=author_id,
            confidence=1.0,
            extraction_version="det_1.0"
        )
        claims.append(c_author)
        evidence.append(Evidence(evidence_id=generate_id(c_author.claim_id, artifact.artifact_id), claim_id=c_author.claim_id, artifact_id=artifact.artifact_id, text_span="Author Metadata", confidence=1.0))
        
        # Status
        status = raw_data.get("state", "open")
        c_status = Claim(
            claim_id=generate_id('has_status', issue_id, status),
            subject=issue_id,
            predicate="has_status",
            object=status,
            confidence=1.0,
            extraction_version="det_1.0"
        )
        claims.append(c_status)
        evidence.append(Evidence(evidence_id=generate_id(c_status.claim_id, artifact.artifact_id), claim_id=c_status.claim_id, artifact_id=artifact.artifact_id, text_span="State Metadata", confidence=1.0))
        
        # Labels
        for label in raw_data.get("labels", []):
            label_name = label.get("name", str(label))
            label_id = f"label_{label_name}"
            entities.append(Entity(id=label_id, type="Label", name=label_name))
            c_label = Claim(
                claim_id=generate_id('labeled_as', issue_id, label_id),
                subject=issue_id,
                predicate="labeled_as",
                object=label_id,
                confidence=1.0,
                extraction_version="det_1.0"
            )
            claims.append(c_label)
            evidence.append(Evidence(evidence_id=generate_id(c_label.claim_id, artifact.artifact_id), claim_id=c_label.claim_id, artifact_id=artifact.artifact_id, text_span=label_name, confidence=1.0))
            
    elif artifact.type == "comment":
        parent_issue_url = raw_data.get("issue_url", "")
        if parent_issue_url:
            parent_issue_num = parent_issue_url.split("/")[-1]
            parent_issue_id = f"issue_{parent_issue_num}"
        else:
            parent_issue_id = "unknown_issue"
            
        c_comment = Claim(
            claim_id=generate_id('commented_on', author_id, parent_issue_id),
            subject=author_id,
            predicate="commented_on",
            object=parent_issue_id,
            confidence=1.0,
            extraction_version="det_1.0"
        )
        claims.append(c_comment)
        evidence.append(Evidence(evidence_id=generate_id(c_comment.claim_id, artifact.artifact_id), claim_id=c_comment.claim_id, artifact_id=artifact.artifact_id, text_span="Comment Metadata", confidence=1.0))
        
    return entities, claims, evidence
