import hashlib
from typing import List
from extraction.schema import Artifact

def hash_text(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()

def deduplicate_artifacts(artifacts: List[Artifact]) -> List[Artifact]:
    """Artifact deduplication using SHA256 hash of normalized text."""
    seen = set()
    deduped = []
    for art in artifacts:
        h = hash_text(art.raw_text)
        if h not in seen:
            seen.add(h)
            deduped.append(art)
    return deduped
