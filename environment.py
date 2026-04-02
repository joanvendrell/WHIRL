import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


###############################################################################################################
           ######################## ---- Function to create the graph ---- ########################
###############################################################################################################
import random
import math


def generate_connected_weighted_graph(
    N: int,
    connectivity: float,
    weight_range: tuple = (1.0, 10.0),
    seed: int | None = None,
    path_bias: float = 0.0,
    layout: str = "line",
) -> nx.Graph:
    """
    Generate a connected undirected weighted graph with N nodes.

    Parameters
    ----------
    N : int
        Number of nodes.
    connectivity : float
        Density parameter in [0, 1].
        - 0 gives a sparse connected graph (roughly a random spanning tree).
        - 1 gives a complete graph.
    weight_range : tuple
        Range (min_weight, max_weight) for random edge weights.
    seed : int | None
        Random seed for reproducibility.
    path_bias : float
        Controls how "short-path" the graph tends to be.
        - path_bias > 0: favors long-range edges -> smaller diameter / average path length
        - path_bias < 0: favors local edges     -> larger diameter / average path length
        - path_bias = 0: uniform random edge addition
    layout : str
        Node arrangement used to define "distance" between nodes.
        Options:
        - "line": nodes placed on a line [0, 1, ..., N-1]
        - "circle": nodes placed on a ring

    Returns
    -------
    G : nx.Graph
        Connected undirected weighted graph.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if not (0 <= connectivity <= 1):
        raise ValueError("connectivity must be between 0 and 1.")
    if weight_range[0] > weight_range[1]:
        raise ValueError("weight_range must satisfy min <= max.")
    if layout not in {"line", "circle"}:
        raise ValueError("layout must be either 'line' or 'circle'.")

    rng = random.Random(seed)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    if N == 1:
        return G

    # ------------------------------------------------------------------
    # Helper: structural distance between two nodes in the chosen layout
    # ------------------------------------------------------------------
    def node_distance(u: int, v: int) -> float:
        d = abs(u - v)
        if layout == "circle":
            d = min(d, N - d)
        return float(d)

    # ------------------------------------------------------------------
    # Step 1: force connectivity with a random spanning tree
    # ------------------------------------------------------------------
    nodes = list(range(N))
    rng.shuffle(nodes)

    for i in range(1, N):
        u = nodes[i]
        v = nodes[rng.randint(0, i - 1)]
        w = rng.uniform(weight_range[0], weight_range[1])
        G.add_edge(u, v, weight=w)

    # ------------------------------------------------------------------
    # Step 2: add extra random edges according to connectivity
    #         but bias edge selection to influence path length
    # ------------------------------------------------------------------
    all_possible_edges = list(itertools.combinations(range(N), 2))
    current_edges = {tuple(sorted(e)) for e in G.edges()}
    missing_edges = [e for e in all_possible_edges if tuple(sorted(e)) not in current_edges]

    max_edges = N * (N - 1) // 2
    min_edges = N - 1  # spanning tree
    target_edges = int(round(min_edges + connectivity * (max_edges - min_edges)))
    extra_edges_needed = max(0, target_edges - G.number_of_edges())

    if extra_edges_needed == 0:
        return G

    # Build sampling weights from structural distance
    # Large path_bias -> prefer larger node_distance -> more "shortcuts"
    distances = [node_distance(u, v) for (u, v) in missing_edges]

    # Avoid division/degeneracy when N=2
    max_dist = max(distances) if distances else 1.0
    if max_dist == 0:
        max_dist = 1.0

    scaled_distances = [d / max_dist for d in distances]

    # Exponential bias:
    #   weight ~ exp(path_bias * normalized_distance)
    # so:
    #   path_bias > 0 -> long edges more likely
    #   path_bias < 0 -> short edges more likely
    edge_weights = [math.exp(path_bias * d) for d in scaled_distances]

    # Sample without replacement according to these weights
    chosen_edges = _weighted_sample_without_replacement(
        items=missing_edges,
        weights=edge_weights,
        k=min(extra_edges_needed, len(missing_edges)),
        rng=rng,
    )

    for (u, v) in chosen_edges:
        w = rng.uniform(weight_range[0], weight_range[1])
        G.add_edge(u, v, weight=w)

    return G


def _weighted_sample_without_replacement(items, weights, k, rng):
    """
    Weighted sampling without replacement using sequential roulette-wheel sampling.
    """
    if k <= 0:
        return []

    available_items = list(items)
    available_weights = list(weights)
    chosen = []

    for _ in range(min(k, len(available_items))):
        total_w = sum(available_weights)
        if total_w <= 0:
            # fallback to uniform choice if all weights are zero
            idx = rng.randrange(len(available_items))
        else:
            r = rng.uniform(0, total_w)
            cumsum = 0.0
            idx = 0
            for i, w in enumerate(available_weights):
                cumsum += w
                if r <= cumsum:
                    idx = i
                    break

        chosen.append(available_items.pop(idx))
        available_weights.pop(idx)

    return chosen

def assign_providers_and_client(G, k: int, seed: int | None = None):
    """
    Randomly assign k distinct provider nodes and 1 distinct client node.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    k : int
        Number of provider nodes.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    providers : list
        List of k distinct provider nodes.
    client : int
        A single client node, distinct from all providers.
    """
    nodes = list(G.nodes)
    n = len(nodes)

    if k <= 0:
        raise ValueError("k must be positive.")
    if k >= n:
        raise ValueError("k must be smaller than the number of nodes, since client must be distinct.")

    rng = random.Random(seed)

    selected = rng.sample(nodes, k + 1)
    providers = selected[:k]
    client = selected[k]

    return providers, client


def estimate_path_bias(
    target_L,
    N=30,
    connectivity=0.2,
    layout="line",
    tol=0.2,
    max_iter=12,
):
    """
    Approximate path_bias that yields a target average shortest path length.
    """

    low, high = -6.0, 6.0
    best_bias = 0.0
    best_diff = float("inf")

    for _ in range(max_iter):
        mid = (low + high) / 2

        G = generate_connected_weighted_graph(
            N=N,
            connectivity=connectivity,
            path_bias=mid,
            layout=layout,
            seed=None,
        )

        L = nx.average_shortest_path_length(G)
        diff = abs(L - target_L)

        if diff < best_diff:
            best_diff = diff
            best_bias = mid

        if diff < tol:
            return mid

        # Binary search direction
        if L > target_L:
            # graph too "long" → need more shortcuts → increase bias
            low = mid
        else:
            # graph too "short" → need more locality → decrease bias
            high = mid

    return best_bias