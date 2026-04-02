import networkx as nx
import itertools
###############################################################################################################
           ######################## ---- Function to find the paths ---- ########################
###############################################################################################################
def find_m_distinct_paths(G: nx.Graph, u, v, m: int, weight="weight"):
    """
    Find up to m distinct simple paths between nodes u and v.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    u : node
        Start node.
    v : node
        End node.
    m : int
        Maximum number of distinct paths to return.

    Returns
    -------
    paths : list[list]
        List of distinct simple paths from u to v.
    """
    if m <= 0:
        raise ValueError("m must be positive.")
    if u not in G:
        raise ValueError(f"Node {u} is not in the graph.")
    if v not in G:
        raise ValueError(f"Node {v} is not in the graph.")

    paths = []

    try:
        paths_gen = nx.shortest_simple_paths(G, u, v, weight=weight)
        paths = list(itertools.islice(paths_gen, m))
    except nx.NetworkXNoPath:
        return []

    return paths


def shortest_path(G: nx.Graph, u, v, weight: str = "weight"):
    """
    Compute the shortest path between nodes u and v.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    u : node
        Start node.
    v : node
        End node.
    weight : str
        Edge attribute to use as weight. Default is "weight".

    Returns
    -------
    path : list
        List of nodes representing the shortest path.
    length : float
        Total weight of the path.
    """
    if u not in G:
        raise ValueError(f"Node {u} is not in the graph.")
    if v not in G:
        raise ValueError(f"Node {v} is not in the graph.")

    try:
        path = nx.shortest_path(G, source=u, target=v, weight=weight)
        length = nx.shortest_path_length(G, source=u, target=v, weight=weight)
        return path, length
    except nx.NetworkXNoPath:
        return None, float("inf")