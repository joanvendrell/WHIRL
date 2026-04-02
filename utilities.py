import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

###############################################################################################################
               ######################## ---- Utility functions ---- ########################
###############################################################################################################
def generate_benefit_and_diminishing_matrix(L: int,
                                            k: int,
                                            benefit_range: tuple = (1.0, 2.0),
                                            diminishing_range: tuple = (0.3, 1.0),
                                            symmetric: bool = True,
                                            seed: int | None = None):
    """
    Generate:
      - Benefit matrix B (L x k)
      - Diminishing return matrix D (L x L)

    Parameters
    ----------
    L : int
        Number of items.
    k : int
        Number of providers.
    benefit_range : tuple
        Range (min, max) for benefit values.
    diminishing_range : tuple
        Range (min, max) for diminishing factors (off-diagonal).
        Values closer to 0 → strong diminishing
        Values closer to 1 → weak diminishing
    symmetric : bool
        If True, D[i,j] = D[j,i]
    seed : int | None
        Random seed.

    Returns
    -------
    B : np.ndarray (L x k)
        Benefit matrix.
    D : np.ndarray (L x L)
        Diminishing return matrix.
    """
    if L <= 0 or k <= 0:
        raise ValueError("L and k must be positive.")

    rng = np.random.default_rng(seed)

    # ---- Benefit matrix ----
    B = rng.uniform(benefit_range[0], benefit_range[1], size=(L, k))

    # ---- Diminishing matrix ----
    D = rng.uniform(diminishing_range[0], diminishing_range[1], size=(L, L))

    # enforce diagonal = 1
    np.fill_diagonal(D, 1.0)

    # enforce symmetry if desired
    if symmetric:
        D = (D + D.T) / 2

    return B, D

def path_cost_simple(G, S, path, max_len_path, max_S_size) -> float:
    """
    Cost with h(|S|) = |S|^2.
    """
    if len(path) < 2:
        return 0.0
    
    total_edge_weight = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) is not in the graph.")
        total_edge_weight += G[u][v]["weight"]
    return ((len(S) ** 1)/(max_S_size**1)) * total_edge_weight/max_len_path
#change to ** 2 for squared cost

def path_cost_weight_only(G, path):
    if len(path) < 2:
        return 0.0
    total_edge_weight = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) is not in the graph.")
        total_edge_weight += G[u][v]["weight"]
    return total_edge_weight

def path_cost_submod(G: nx.Graph, providers: list, working_S: dict, working_R: dict, max_len_path: float, max_S_size: int, gamma: float):
    #get all edges used from all paths
    #matrix len(all_edges) x num_providers
    #values 0 or 1 if provider uses edge

    truth_sets = []
    all_edges = set()
    for provider in providers:
        # add case where None path
        if working_R[provider] is None:
            continue
        path = working_R[provider].copy()
        edges = list(zip(path[:-1], path[1:]))
        provider_edges = [] # list of tuples/edges
        for edge in edges:
            provider_edges.append(edge)
            all_edges.add(edge)
        truth_sets.append({provider: provider_edges})
    
    list_of_edges = list(all_edges)
    truth_matrix = np.zeros((len(providers), len(all_edges)))
    
    for truths in truth_sets:
        provider_idx = providers.index(list(truths.keys())[0])
        for edge in list(truths.values())[0]:
            edge_idx = list_of_edges.index(edge)
            truth_matrix[provider_idx, edge_idx] = 1
    total_cost = 0
    for edge in list_of_edges:
        weight = G[edge[0]][edge[1]]["weight"]
        cardinality = 0
        for provider in providers:
            cardinality += len(working_S[provider])*truth_matrix[providers.index(provider), list_of_edges.index(edge)]
        total_cost += (cardinality**gamma)*weight
   
    return total_cost #(total_cost/max_len_path)/((max_S_size/2)**gamma)
    



        


        

