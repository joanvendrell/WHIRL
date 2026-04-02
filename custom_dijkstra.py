import networkx as nx
import numpy as np

def build_marginal_graph_for_provider(
    G: nx.Graph,
    providers: list,
    working_S: dict,
    working_R: dict,
    provider,
    gamma: float,
):
    """
    Return a copy of G where each edge weight is the marginal increase in
    path_cost_submod caused by routing `provider` through that edge,
    assuming all other providers are fixed.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    # Current load induced by all providers except `provider`
    current_edge_load = {}

    for other in providers:
        if other == provider or working_R[other] is None:
            continue

        other_path = working_R[other]
        other_load = len(working_S[other])

        for u, v in zip(other_path[:-1], other_path[1:]):
            e = tuple(sorted((u, v)))
            current_edge_load[e] = current_edge_load.get(e, 0) + other_load

    provider_load = len(working_S[provider])

    for u, v, data in G.edges(data=True):
        e = tuple(sorted((u, v)))
        base_weight = data["weight"]
        c = current_edge_load.get(e, 0)

        # marginal increase if provider uses this edge
        marginal = ((c + provider_load) ** gamma - (c ** gamma)) * base_weight

        H.add_edge(u, v, weight=marginal)

    return H

def submodular_dijkstra_path(
    G: nx.Graph,
    providers: list,
    working_S: dict,
    working_R: dict,
    provider,
    source,
    target,
    gamma: float,
):
    H = build_marginal_graph_for_provider(
        G, providers, working_S, working_R, provider, gamma
    )
    return nx.dijkstra_path(H, source=source, target=target, weight="weight")

