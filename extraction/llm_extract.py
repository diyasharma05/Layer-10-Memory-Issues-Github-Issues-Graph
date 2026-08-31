import os
import re
import json
from typing import List, Tuple
from pydantic import BaseModel, Field
from extraction.schema import Artifact, Entity, Claim, Evidence
from extraction.deterministic_extract import generate_id

class ExpectedReferences(BaseModel):
    referenced_issue_numbers: List[str] = Field(description="List of issue numbers referenced in the text.")

def heuristic_extract(artifact: Artifact) -> List[str]:
    """Fallback extraction using Regex if no API key is present."""
    matches = re.findall(r'#(\d+)', artifact.raw_text)
    return list(set(matches))

def llm_extract_references(artifact: Artifact) -> Tuple[List[Entity], List[Claim], List[Evidence]]:
    """Extract semantic relationships (like references) using LLM."""
    referenced_issues = []
    
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        referenced_issues = heuristic_extract(artifact)
    else:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL")
            )
            prompt = f"Extract all GitHub issue numbers referenced in this text. Text:\n{artifact.raw_text}"
            
            response = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Output JSON with 'referenced_issue_numbers' array containing strings denoting issue numbers."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result_dict = json.loads(response.choices[0].message.content)
            parsed = ExpectedReferences(**result_dict)
            referenced_issues = parsed.referenced_issue_numbers
        except Exception as e:
            print(f"LLM extraction failed, falling back: {e}")
            referenced_issues = heuristic_extract(artifact)
            
    entities = []
    claims = []
    evidence = []
    
    subject_id = f"user_{artifact.author}"
    
    for ref_num in referenced_issues:
        target_issue_id = f"issue_{ref_num}"
        entities.append(Entity(id=target_issue_id, type="Issue", name=f"Issue #{ref_num}"))
        
        c = Claim(
            claim_id=generate_id('references', subject_id, target_issue_id),
            subject=subject_id,
            predicate="references",
            object=target_issue_id,
            confidence=0.8,
            extraction_version="llm_1.0" if api_key else "regex_1.0"
        )
        claims.append(c)
        evidence.append(Evidence(
            evidence_id=generate_id(c.claim_id, artifact.artifact_id),
            claim_id=c.claim_id,
            artifact_id=artifact.artifact_id,
            text_span=f"#{ref_num}",
            confidence=0.8
        ))
        
    return entities, claims, evidence
