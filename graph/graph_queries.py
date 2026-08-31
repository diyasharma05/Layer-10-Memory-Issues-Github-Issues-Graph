from typing import List, Dict
import networkx as nx

def get_subgraph(G: nx.DiGraph, center_node: str, radius: int = 2) -> nx.DiGraph:
    if center_node not in G:
        return nx.DiGraph()
    
    # Get all nodes within radius
    nodes = set(nx.single_source_shortest_path_length(G.to_undirected(), center_node, cutoff=radius).keys())
    return G.subgraph(nodes).copy()
