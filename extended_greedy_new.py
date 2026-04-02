import itertools
import networkx as nx
import numpy as np
import random
import copy
from utilities import generate_benefit_and_diminishing_matrix, path_cost_simple, path_cost_submod

def generate_possible_for_item(item: int, providers: list, paths_per_provider: dict):
    combinations = [] #list of dict {provider: provider_id, path: [node_list]}
    for provider in providers:
        paths = paths_per_provider.get(provider, [])
        for path in paths:
            combinations.append({"provider": provider, "path": path})
    return combinations

def filter_possible(working_R, providers, combinations):
    filtered = combinations.copy()
    for provider in providers:
        if working_R[provider] is not None:
            for elem in combinations:
                if list(elem.values())[1] != working_R[provider] and list(elem.values())[0] == provider:
                    filtered.remove(elem)
    return filtered

def extended_greedy(G: nx.Graph, L: int, providers: list, paths_per_provider: dict, benefits: np.ndarray, diminishings: np.ndarray, alpha:float, max_len_path:float, gamma: float) -> dict:
    assignment = {idx:None for idx in range(L)}
    working_S = {idx:[] for idx in providers}
    working_R = {idx:None for idx in providers}

    not_assigned = list(range(L))
    while not_assigned:
        max_utility = float("-inf")
        best_item = 0; best_comb = 0
        for item in not_assigned:
            all_possible = generate_possible_for_item(item, providers, paths_per_provider)
            # filter out paths that are not possible
            filtered = filter_possible(working_R, providers, all_possible)
            # 1 possible for each provider if route already picked
            max_local = float("-inf"); best_comb_local = None
            for possible in filtered:
                provider = possible["provider"]
                items = working_S[provider].copy()
                items.append(item)
                temp_S = copy.deepcopy(working_S)
                temp_S[provider].append(item)
                temp_R = copy.deepcopy(working_R)
                temp_R[provider] = possible["path"]

                # calculate f_S - alpha*cost and compare all for that item
                # f_S for provider of interest
                f_S = 0; cumulative_diminsih = 1; cost = 0
                for idx in range(len(items)):
                    f_S += benefits[items[idx]][providers.index(provider)] * cumulative_diminsih
                    next = items[idx+1] if idx + 1 < len(items) else items[idx]
                    cumulative_diminsih *= diminishings[next][items[idx]]
                #cost for this provider
                # cost += path_cost_simple(G, items, paths[provider], max_len_path, len(benefits))

                #calculate f_S for other providers
                others = providers.copy()
                others.remove(provider)
                for other_idx in others:
                    other_items = working_S[other_idx].copy()
                    cumulative_diminsih = 1
                    for jdx in range(len(other_items)):
                        f_S += benefits[other_items[jdx]][providers.index(other_idx)] * cumulative_diminsih
                        next = other_items[jdx+1] if jdx + 1 < len(other_items) else other_items[jdx]
                        cumulative_diminsih *= diminishings[next][other_items[jdx]]
                    #cost for this provider
                    # if paths[other_idx] is not None:
                    #     cost += path_cost_simple(G, other_items, paths[other_idx], max_len_path, len(benefits))
                cost = path_cost_submod(G, providers, temp_S, temp_R, max_len_path, len(benefits), gamma) # computes cost for all paths at once
                
                #total utility
                utility = f_S - alpha*cost
                if utility >= max_local:
                    max_local = utility
                    best_comb_local = possible #dict {"provider": provider_id, "path": [node list]}
            if max_local >= max_utility:
                max_utility = max_local
                best_comb = best_comb_local
                best_item = item
        #assignment
        working_S[best_comb["provider"]].append(best_item)
        working_R[best_comb["provider"]] = best_comb["path"]
        not_assigned.remove(best_item)
        

    #reformat
    for item in range(L):
        provider = [idx for idx, values in working_S.items() if item in values][0]
        assignment[item] = working_R[provider]

    return assignment, max_utility