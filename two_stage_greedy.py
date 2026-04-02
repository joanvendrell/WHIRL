import itertools
import networkx as nx
import numpy as np
import copy
from utilities import generate_benefit_and_diminishing_matrix, path_cost_simple, path_cost_submod

'''version with old T(S,r), but paths are computed with Dijkstra'''

def two_stage_greedy(
    G: nx.Graph,
    L: int,
    providers: list,
    provider_targets: dict,
    benefits: np.ndarray,
    diminishings: np.ndarray,
    alpha: float,
    max_len_path: float,
    gamma: float
) -> dict:
    # --------------------------------------------------
    # Stage 1: assign items to providers
    # --------------------------------------------------
    working_S = {idx: [] for idx in providers}
    not_assigned = list(range(L))

    while not_assigned:
        best_provider = 0
        best_item = 0
        max_utility = float("-inf")

        for item in not_assigned:
            best_provider_item = 0
            max_local_utility = float("-inf")

            for provider in providers:
                items = working_S[provider].copy()
                items.append(item)

                # Get f(S)
                f_S = 0
                cumulative_diminsih = 1
                for idx in range(len(items)):
                    f_S += benefits[items[idx]][providers.index(provider)] * cumulative_diminsih
                    next_item = items[idx + 1] if idx + 1 < len(items) else items[idx]
                    cumulative_diminsih *= diminishings[next_item][items[idx]]

                # Get f(S) for other providers
                others = providers.copy()
                others.remove(provider)
                for other_idx in others:
                    other_items = working_S[other_idx].copy()
                    cumulative_diminsih = 1
                    for jdx in range(len(other_items)):
                        f_S += benefits[other_items[jdx]][providers.index(other_idx)] * cumulative_diminsih
                        next_item = other_items[jdx + 1] if jdx + 1 < len(other_items) else other_items[jdx]
                        cumulative_diminsih *= diminishings[next_item][other_items[jdx]]

                # Adjust search
                if f_S >= max_local_utility:
                    max_local_utility = f_S
                    best_provider_item = provider

            # Adjust global search
            if max_local_utility >= max_utility:
                max_utility = max_local_utility
                best_item = item
                best_provider = best_provider_item

        # Update assignment
        working_S[best_provider].append(best_item)
        not_assigned.remove(best_item)

    # --------------------------------------------------
    # Stage 2: assign one shortest path per provider
    # --------------------------------------------------
    working_R = {idx: None for idx in providers}
    cost = 0
    for provider in providers:
        items = working_S[provider].copy()

        source = provider
        target = provider_targets[provider]

        try:
            path = nx.dijkstra_path(G, source=source, target=target, weight="weight")
        except nx.NetworkXNoPath:
            path = None
        
        temp_R = copy.deepcopy(working_R)
        temp_R[provider] = path
        """
        # Get f(S) for selected provider
        f_S = 0
        cumulative_diminsih = 1
        for idx in range(len(items)):
            f_S += benefits[items[idx]][providers.index(provider)] * cumulative_diminsih
            next_item = items[idx + 1] if idx + 1 < len(items) else items[idx]
            cumulative_diminsih *= diminishings[next_item][items[idx]]

        # Get f(S) for other providers too
        others = providers.copy()
        others.remove(provider)
        for other_idx in others:
            other_items = working_S[other_idx].copy()
            cumulative_diminsih = 1
            for jdx in range(len(other_items)):
                f_S += benefits[other_items[jdx]][providers.index(other_idx)] * cumulative_diminsih
                next_item = other_items[jdx + 1] if jdx + 1 < len(other_items) else other_items[jdx]
                cumulative_diminsih *= diminishings[next_item][other_items[jdx]]

        # Get cost T(S,R)
        cost = path_cost_submod(
            G, providers, working_S, temp_R, max_len_path, len(benefits), gamma
        )
        utility = f_S - alpha * cost
        """
        cost += path_cost_submod(
            G, providers, working_S, temp_R, max_len_path, len(benefits), gamma
        )
        working_R[provider] = path
    cost = path_cost_submod(
            G, providers, working_S, temp_R, max_len_path, len(benefits), gamma
        )
    utility = max_utility - alpha * cost
    # --------------------------------------------------
    # Format output
    # --------------------------------------------------
    assignment = {idx: None for idx in range(L)}
    for item in range(L):
        provider = [idx for idx, values in working_S.items() if item in values][0]
        assignment[item] = working_R[provider]

    return assignment, utility