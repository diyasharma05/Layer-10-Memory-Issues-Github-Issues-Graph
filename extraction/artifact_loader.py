import csv
from typing import Iterator, Dict, Any, Tuple
from datetime import datetime
from extraction.schema import Artifact

def row_to_artifact(row: Dict[str, str]) -> Tuple[Artifact, Dict[str, Any]]:
    """Helper to parse a single CSV row into an Artifact."""
    
    # We don't have created_at in the CSV, so we use the current time as a placeholder
    timestamp = datetime.now()

    # The CSV only has issue_url, issue_title, and body
    issue_url = row.get("issue_url", "")
    # Default ID to the URL, or "unknown" if missing
    artifact_id = issue_url if issue_url else "unknown"

    title = row.get("issue_title", "").strip()
    body = row.get("body", "").strip()
    
    # Combine title and body for the raw_text
    raw_text = f"{title}\n\n{body}".strip()

    art = Artifact(
        artifact_id=artifact_id,
        type="issue", # Defaulting to issue for all rows
        author="unknown", # We don't have author in the CSV
        timestamp=timestamp,
        raw_text=raw_text
    )
    return art, row

def load_artifacts(file_path: str) -> Iterator[Tuple[Artifact, Dict[str, Any]]]:
    """Loads a CSV file with GitHub issues and returns an iterator of (Artifact, RawDict)."""
    with open(file_path, "r", encoding="utf-8") as f:
        # We need to handle large CSV files and possible unexpected formats
        # Using csv.DictReader to automatically use the first row as keys
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty URLs
            if not row.get("issue_url"):
                continue
            yield row_to_artifact(row)
