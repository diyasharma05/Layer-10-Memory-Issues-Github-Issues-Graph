import os
import networkx as nx
from neo4j import GraphDatabase
from typing import List, Tuple
from extraction.schema import Entity, Claim, Evidence, Artifact
from graph.graph_schema import create_constraints

class KnowledgeGraph:
    def __init__(self, uri=None, user=None, password=None):
        uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.environ.get("NEO4J_USER", "neo4j")
        password = password or os.environ.get("NEO4J_PASSWORD", "password")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.connected = True
            print("Connected to Neo4j.")
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j. Graph features will use local NetworkX fallback. Error: {e}")
            self.connected = False
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def init_schema(self):
        if not self.connected: return
        with self.driver.session() as session:
            session.execute_write(create_constraints)

    def insert_data(self, entities: List[Entity], claims: List[Claim], evidence: List[Evidence], artifacts: List[Artifact]):
        if not self.connected:
            return
            
        with self.driver.session() as session:
            for e in entities:
                session.run("MERGE (n:Entity {id: $id}) SET n.type = $type, n.name = $name", id=e.id, type=e.type, name=e.name)
            for a in artifacts:
                session.run("MERGE (n:Artifact {artifact_id: $aid}) SET n.type = $type, n.author = $author, n.timestamp = $ts", aid=a.artifact_id, type=a.type, author=a.author, ts=a.timestamp.isoformat())
            for c in claims:
                session.run("MERGE (n:Claim {claim_id: $cid}) SET n.predicate = $pred, n.confidence = $conf", cid=c.claim_id, pred=c.predicate, conf=c.confidence)
                session.run("MATCH (e:Entity {id: $subj}), (c:Claim {claim_id: $cid}) MERGE (e)-[:SUBJECT_OF]->(c)", subj=c.subject, cid=c.claim_id)
                session.run("MATCH (c:Claim {claim_id: $cid}), (e:Entity {id: $obj}) MERGE (c)-[:OBJECT_OF]->(e)", cid=c.claim_id, obj=c.object)
            for ev in evidence:
                session.run("MERGE (n:Evidence {evidence_id: $vid}) SET n.text_span = $span, n.confidence = $conf", vid=ev.evidence_id, span=ev.text_span[:100], conf=ev.confidence)
                session.run("MATCH (c:Claim {claim_id: $cid}), (v:Evidence {evidence_id: $vid}) MERGE (c)-[:SUPPORTED_BY]->(v)", cid=ev.claim_id, vid=ev.evidence_id)
                session.run("MATCH (v:Evidence {evidence_id: $vid}), (a:Artifact {artifact_id: $aid}) MERGE (v)-[:SOURCE]->(a)", vid=ev.evidence_id, aid=ev.artifact_id)

def build_networkx_graph(entities: List[Entity], claims: List[Claim], evidence: List[Evidence], artifacts: List[Artifact]) -> nx.DiGraph:
    """Builds a NetworkX graph for local visualization and retrieval fallback."""
    G = nx.DiGraph()
    
    for e in entities:
        G.add_node(e.id, label=e.name, type=e.type, color="lightblue")
        
    for a in artifacts:
        G.add_node(a.artifact_id, label=f"Artifact {a.artifact_id[:6]}", type=a.type, color="lightgreen")
        
    for c in claims:
        G.add_node(c.claim_id, label=c.predicate, type="Claim", color="orange")
        G.add_edge(c.subject, c.claim_id, label="SUBJECT_OF")
        # If object is an entity that exists in graph
        if c.object in G.nodes:
            G.add_edge(c.claim_id, c.object, label="OBJECT_OF")
        else:
            # Maybe object is a literal (e.g. status)
            G.add_node(c.object, label=c.object, type="Literal", color="lightgray")
            G.add_edge(c.claim_id, c.object, label="OBJECT_OF")
            
    for ev in evidence:
        G.add_node(ev.evidence_id, label="Evidence", type="Evidence", color="yellow")
        if ev.claim_id in G.nodes:
            G.add_edge(ev.claim_id, ev.evidence_id, label="SUPPORTED_BY")
        if ev.artifact_id in G.nodes:
            G.add_edge(ev.evidence_id, ev.artifact_id, label="SOURCE")
            
    return G
