# Layer10 Memory System - GitHub Issues Graph

This project implements a grounded, long-term memory system designed to process unstructured organizational knowledge (GitHub Issues) and convert it into a structured knowledge graph. The system demonstrates data ingestion, extraction, deduplication, graph construction, and retrieval.

## Architecture Overview

1. **Ingestion**: Reads raw GitHub issue data from a large CSV file (`github_issues.csv`).
2. **Extraction Layer**:
   - **Deterministic**: Uses regex to extract issue references, mentions, and URLs from raw text.
   - **LLM-Based (Simulated/Provider via `pydantic` output schema)**: Uses an LLM to cleanly extract context triples (Subject, Predicate, Object) along with confidence scores and evidence pointers.
3. **Deduplication & Canonicalization**:
   - Resolves multiple aliases into canonical entities (e.g., standardizing repository references).
   - Deduplicates identical claims to prevent graph bloat and tracks provenance across multiple artifact sources.
4. **Graph Construction**: Stores the extracted data natively using Neo4j and also constructs an in-memory `NetworkX` graph for visualization.
5. **Retrieval Layer**: Uses a Hybrid Retriever approach. It matches incoming queries against embeddings of the claims (via `SentenceTransformers` + `FAISS`), and optionally expands search context through graph traversals (finding neighboring claims).
6. **Visualization**: A Streamlit application using `pyvis` designed to visualize the NetworkX relationship graph interactively.

---

## 1. Ontology Definition

The ontology for this system structures unstructured text into an interconnected knowledge base suitable for RAG (Retrieval-Augmented Generation) applications.

- **Artifacts**: The source of truth. Each row in the CSV represents an artifact, acting as the foundational document containing the raw text (e.g., `Issue #1234`).
- **Entities**: The nodes in our knowledge graph. These can represent people, repositories, error codes (e.g., `glibc_2.14`), UI components, or concepts.
- **Claims**: The edges (relationships) connecting entities. Each claim is represented as a triple: `(Subject, Predicate, Object)`. Example: `(Issue #1234, depends_on, Issue #1265)`.
- **Evidence Snippets**: The bridge between Claims and Artifacts. Every claim points back to the exact substring in an Artifact that generated it, allowing for complete provenance tracking.

---

## 2. Extraction Contract

The system enforces a strict schema for LLM outputs utilizing `pydantic`. This represents our **Extraction Contract**: The LLM must conform to the defined JSON schema, ensuring that downstream layers (like the graph database) receive highly predictable data.

### Fields required by the contract:
- `entities`: A list of explicitly named entities found in the text.
- `claims`: A list of relation triples, providing `subject`, `predicate`, `object`, and a `confidence` floating-point metric (0.0 to 1.0) of how certain the LLM is.
- `evidence_snippets`: For each claim, a text span directly quoted from the source text explicitly validating the claim.

If the LLM output violates this JSON structure, the system rejects it, preventing malformed data from polluting the memory store.

---

## 3. Deduplication Strategy

Redundant information is a major challenge in memory systems over time. Our deduplication logic occurs in two steps:

1. **Entity Canonicalization**: We use string similarity and exact matching to merge various entity permutations into a single canonical ID. E.g., `node images`, `node-images`, and `Node Images` resolve to one node.
2. **Claim Deduplication**: If an identical claim (e.g., `(User A, reported, Bug B)`) is extracted from multiple artifacts, we **do not create duplicate edges**. Instead, we maintain a single claim and append the new exact `evidence_snippet` and `artifact_id` to that claim's provenance list. This increases our confidence in the claim without cluttering the graph.

---

## 4. Update Semantics (Handling New Code/Data over Time)

When the memory system ingests new data (e.g., a new comment on an older issue):

- **Upserts**: Artifacts are hashed based on their raw content. If the artifact has not changed, we skip extraction. If it is new or updated, we parse it.
- **Additive Claims**: New claims are simply added to the graph.
- **Conflicting Claims**: The retrieval system relies on confidence scores. If an issue is marked "Open" but later marked "Closed", the system leverages the timestamps on the artifacts from which the evidence snippets were sourced to surface the most recent state accurately.

---

## 5. Adaptation to Layer10

This architecture closely mirrors the needs of Layer10's organizational memory. By representing complex internal workflows (like PR reviews, JIRA tickets, Slack conversations, and GitHub issues) as Nodes (Entities) and relationships (Claims), Layer10 can build context-aware AI agents. 

The strict extraction contract ensures high-fidelity data, while the evidence tracking provides transparent provenance—essential features when dealing with crucial enterprise knowledge.

---

## Reproducibility & Running the Project

### Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`

### Step 1: Place Data
Ensure you have downloaded the CSV dataset and placed it precisely at:
`data/raw/github_issues.csv`

The CSV must contain the columns: `issue_url`, `issue_title`, and `body`.

### Step 2: Run the Pipeline (Extraction & Graph Building)
Run the automated pipeline to parse the CSV, extract entities, perform canonicalization, and build the memory store.

```bash
# From the root directory:
python pipeline/run_pipeline.py
```

*Note: The pipeline might take significant time depending on the size of the CSV and the chosen extraction backend (Deterministic vs LLM).*

### Step 3: Outputs
Execution will populate the `data/processed/` directory with:
- `artifacts.json`
- `claims.json`
- `entities.json`
- `evidence.json`
- `memory_graph.graphml` (A serialized version of the NetworkX graph)

### Step 4: Run the Visualization
We provide a Streamlit app to explore the generated network graph interactively.

```bash
streamlit run visualization/app.py
```
This will open a browser window displaying the entities and relationships extracted from your dataset.
