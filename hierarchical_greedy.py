import itertools
import networkx as nx
import numpy as np
import random
import copy
from custom_dijkstra import submodular_dijkstra_path
from utilities import (
    generate_benefit_and_diminishing_matrix,
    path_cost_simple,
    path_cost_submod,
    path_cost_weight_only,
)

def generate_random_S(L: int, providers: list):
    assign_init = random.choices(providers, k=L)
    S_init = {jdx: [] for jdx in providers}

    for idx in range(L):
        provider = assign_init[idx]
        S_init[provider].append(idx)

    return S_init


def hierarchical_greedy(
    G: nx.Graph,
    L: int,
    providers: list,
    provider_targets: dict,   # NEW: target node per provider
    benefits: np.ndarray,
    diminishings: np.ndarray,
    alpha: float,
    max_len_path: float,
    gamma: float,
):
    # initialize S
    S_init = generate_random_S(L, providers)
    assignment = {idx: None for idx in range(L)}

    working_S = S_init
    working_R = {idx: None for idx in providers}
    not_converged = True

    counter = 0

    while not_converged:
        # ------------------------------------------------------------
        # Optimize R: compute one shortest path per provider with Dijkstra
        # ------------------------------------------------------------
        for provider in providers:
            items = working_S[provider].copy()

            source = provider
            target = provider_targets[provider]

            try:
                if gamma == 0:
                    shortest_path_provider = nx.dijkstra_path(G, source=source, target=target, weight="weight")
                else:
                    shortest_path_provider = submodular_dijkstra_path(
                                                                    G=G,
                                                                    providers=providers,
                                                                    working_S=working_S,
                                                                    working_R=working_R,
                                                                    provider=provider,
                                                                    source=source,
                                                                    target=target,
                                                                    gamma=gamma,
                                                                )
            except nx.NetworkXNoPath:
                shortest_path_provider = None

            # If provider has no assigned items, just keep shortest path
            working_R[provider] = shortest_path_provider
            """
            if items == []:
                working_R[provider] = shortest_path_provider
            else:
                temp_R = working_R.copy()
                temp_R[provider] = shortest_path_provider

                # Get f(S)
                f_S = 0
                cumulative_diminsih = 1

                for idx in range(len(items)):
                    f_S += benefits[items[idx]][providers.index(provider)] * cumulative_diminsih
                    next_item = items[idx + 1] if idx + 1 < len(items) else items[idx]
                    cumulative_diminsih *= diminishings[next_item][items[idx]]

                # f(S) of other providers
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
                #print(f"----> Hey dude, this is f_S: {f_S} | and this is the cost: {cost} | so with alpha: {alpha} we have utility: {f_S - alpha * cost}")
                utility = f_S - alpha * cost
                working_R[provider] = shortest_path_provider
        """
        # ------------------------------------------------------------
        # Optimize S, fix paths
        # ------------------------------------------------------------
        not_assigned = list(range(L))
        new_S = {idx: [] for idx in providers}

        while not_assigned:
            best_provider = 0
            best_item = 0
            max_utility = float("-inf")

            for item in not_assigned:
                best_provider_item = 0
                max_local_utility = float("-inf")

                for provider in providers:
                    items = new_S[provider].copy()
                    items.append(item)

                    temp_S = copy.deepcopy(new_S)
                    temp_S[provider].append(item)

                    # Get f(S)
                    f_S = 0
                    cumulative_diminsih = 1
                    for idx in range(len(items)):
                        f_S += benefits[items[idx]][providers.index(provider)] * cumulative_diminsih
                        next_item = items[idx + 1] if idx + 1 < len(items) else items[idx]
                        cumulative_diminsih *= diminishings[next_item][items[idx]]

                    # f(S) of other providers
                    others = providers.copy()
                    others.remove(provider)
                    for other_idx in others:
                        other_items = new_S[other_idx].copy()
                        cumulative_diminsih = 1
                        for jdx in range(len(other_items)):
                            f_S += benefits[other_items[jdx]][providers.index(other_idx)] * cumulative_diminsih
                            next_item = other_items[jdx + 1] if jdx + 1 < len(other_items) else other_items[jdx]
                            cumulative_diminsih *= diminishings[next_item][other_items[jdx]]

                    cost = path_cost_submod(
                        G, providers, temp_S, working_R, max_len_path, len(benefits), gamma
                    )
                    #print(f"> hell man this is palm beach f_S {f_S} and cost {cost} hence bro {f_S - alpha * cost}")
                    utility = f_S - alpha * cost

                    if utility >= max_local_utility:
                        max_local_utility = utility
                        best_provider_item = provider

                if max_local_utility >= max_utility:
                    max_utility = max_local_utility
                    best_item = item
                    best_provider = best_provider_item

            new_S[best_provider].append(best_item)
            not_assigned.remove(best_item)

        if new_S == working_S:
            not_converged = False
        else:
            working_S = new_S
        counter += 1
    # reformat
    assignment = {idx: None for idx in range(L)}
    for item in range(L):
        provider = [idx for idx, values in new_S.items() if item in values][0]
        assignment[item] = working_R[provider]

    print(f"(runned in {counter} iterations!)")
    return assignment, max_utility