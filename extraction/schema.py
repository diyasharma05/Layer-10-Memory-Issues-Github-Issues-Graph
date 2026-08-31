from pydantic import BaseModel
from typing import Literal, Optional, List
from datetime import datetime

class Artifact(BaseModel):
    artifact_id: str
    type: Literal["issue", "comment"]
    author: str
    timestamp: datetime
    raw_text: str

class Entity(BaseModel):
    id: str
    type: Literal["Person", "Issue", "Label", "Repository"]
    name: str

class Claim(BaseModel):
    claim_id: str
    subject: str
    predicate: Literal["created_by", "assigned_to", "has_status", "labeled_as", "references", "commented_on"]
    object: str
    event_time: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    confidence: float
    extraction_version: str

class Evidence(BaseModel):
    evidence_id: str
    claim_id: str
    artifact_id: str
    text_span: str
    confidence: float

class ContextPack(BaseModel):
    claims: List[Claim]
    evidence_snippets: List[Evidence]
