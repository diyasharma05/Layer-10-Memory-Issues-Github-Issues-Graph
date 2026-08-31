import streamlit as st
import json
import os
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network

st.set_page_config(page_title="Layer10 Memory Graph", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")

@st.cache_data
def load_data():
    def load_json(filename):
        path = os.path.join(OUT_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    entities = load_json("entities.json")
    claims = load_json("claims.json")
    evidence = load_json("evidence.json")
    artifacts = load_json("artifacts.json")
    
    graphml_path = os.path.join(OUT_DIR, "memory_graph.graphml")
    G = None
    if os.path.exists(graphml_path):
        G = nx.read_graphml(graphml_path)
        
    return entities, claims, evidence, artifacts, G

entities, claims, evidence, artifacts, G = load_data()

st.title("Layer10 Knowledge Memory Engine")

if not entities:
    st.warning("No data found. Please run the pipeline first: `python pipeline/run_pipeline.py`")
    st.stop()

# Build evidence map
evidence_map = {}
for ev in evidence:
    evidence_map.setdefault(ev["claim_id"], []).append(ev)
artifact_map = {a["artifact_id"]: a for a in artifacts}

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Search & Filters")
    entity_names = ["All"] + [e["name"] for e in entities]
    selected_entity = st.selectbox("Select Entity", entity_names)
    
    predicates = ["All"] + list(set(c["predicate"] for c in claims))
    selected_predicate = st.selectbox("Filter by Predicate", predicates)
    
    filtered_claims = []
    for c in claims:
        if selected_predicate != "All" and c["predicate"] != selected_predicate:
            continue
        if selected_entity != "All":
            ent_id = next((e["id"] for e in entities if e["name"] == selected_entity), None)
            if ent_id and c["subject"] != ent_id and c["object"] != ent_id:
                continue
        filtered_claims.append(c)
        
    st.write(f"Found **{len(filtered_claims)}** claims")
    
    for c in filtered_claims[:50]:
        with st.expander(f"{c['subject']} -> {c['predicate']} -> {c['object']}"):
            st.write(f"**Confidence**: {c['confidence']}")
            evs = evidence_map.get(c["claim_id"], [])
            if evs:
                st.write("**Supporting Evidence:**")
                for ev in evs:
                    st.write(f"- {ev['text_span']}")
                    art = artifact_map.get(ev["artifact_id"])
                    if art:
                        st.caption(f"Source: {art['type']} by {art['author']}")
                        st.text(art['raw_text'])

with col2:
    st.header("Graph Visualization")
    if G is not None:
        sub_nodes = set()
        for c in filtered_claims[:20]:
            sub_nodes.add(c["subject"])
            sub_nodes.add(c["object"])
            sub_nodes.add(c["claim_id"])
            
        sub_nodes = {n for n in sub_nodes if n in G.nodes}
        
        for c in filtered_claims[:5]:
            for ev in evidence_map.get(c["claim_id"], []):
                sub_nodes.add(ev["evidence_id"])
                
        sub_nodes = {n for n in sub_nodes if n in G.nodes}
        sub_G = G.subgraph(sub_nodes)
        
        net = Network(height="600px", width="100%", directed=True, notebook=True)
        # Avoid PyVis warning
        net.toggle_physics(False)
        net.from_nx(sub_G)
        
        path = os.path.join(BASE_DIR, "html_graph.html")
        net.save_graph(path)
        
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=620)
    else:
        st.write("Graph not built.")
