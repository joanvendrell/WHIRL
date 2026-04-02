import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from environment import generate_connected_weighted_graph, assign_providers_and_client, estimate_path_bias
from utilities import generate_benefit_and_diminishing_matrix, path_cost_simple, path_cost_weight_only
from plot import plot_weighted_graph
from paths import find_m_distinct_paths,shortest_path
from extended_domain_greedy import greedy_algorithm
from two_stage_greedy import two_stage_greedy
from partition_greedy import partition_greedy
from hierarchical_greedy import hierarchical_greedy
from extended_greedy_new import extended_greedy

###############################################
# Model characteristics:
num_nodes = 500 #100
connectivity = 0.5 #0.2
avg_dist = 0.0
num_providers = 15 #3
num_products = 100 #20

path_bias = estimate_path_bias(avg_dist, N=num_nodes,connectivity=connectivity, layout="line", tol=0.2, max_iter=12,)

###############################################################################################################
                      ######################## ---- MAIN ---- ########################
###############################################################################################################
G = generate_connected_weighted_graph(N=num_nodes,connectivity=connectivity,seed=1,path_bias=path_bias,layout="line",)
with open(f"results/graph_{num_nodes}_{connectivity}_{avg_dist}_{num_providers}_{num_products}.pkl", "wb") as f:
    pickle.dump(G, f)
providers, client = assign_providers_and_client(G, k=num_providers, seed=10)

print("Providers:", providers)
print("Client:", client)

B, D = generate_benefit_and_diminishing_matrix(L=num_products,k=num_providers,seed=42)
B = 10*B

# Hierarchical
print("---> Going for Hierarchical ::")
provider_targets = {p:client for p in providers}
start = time.perf_counter()
assignment, max_utility = hierarchical_greedy(G=G, L=num_products, providers=providers, provider_targets=provider_targets, benefits = B, diminishings = D, alpha = 0.1, max_len_path = nx.diameter(G), gamma = 2)
end = time.perf_counter()
print(f"Hierarchical Greedy --> {assignment} and {max_utility} and {end - start:.6f} seconds")
# Two stage greedy
print("---> Going for Two-Stage ::")
start = time.perf_counter()
assignment, max_utility = two_stage_greedy(G=G, L=num_products, providers=providers, provider_targets=provider_targets, benefits = B, diminishings = D, alpha = 0.1, max_len_path = nx.diameter(G), gamma = 2)
end = time.perf_counter()
print(f"Two Stage Greedy --> {assignment} and {max_utility} and {end - start:.6f} seconds")
# Extended Greedy
print("---> Going for Extended ::")
paths_per_provider = {provider:None for provider in providers}
max_path = 0
for provider in providers:
    paths_per_provider[provider] = find_m_distinct_paths(G, provider, client, num_products)
    #for dist_path in paths_per_provider[provider]:
    #    if path_cost_weight_only(G,dist_path) >= max_path:
    #        max_path = path_cost_weight_only(G,dist_path)
start = time.perf_counter()
assignment, max_utility = extended_greedy(G=G, L=num_products, providers=providers, paths_per_provider=paths_per_provider, benefits = B, diminishings = D, alpha = 0.1, max_len_path = nx.diameter(G), gamma = 2)
end = time.perf_counter()
print(f"Extended Greedy --> {assignment} and {max_utility} and {end - start:.6f} seconds")

# plot_weighted_graph(G, providers=providers, client=client,paths=[x for x in list(assignment.values()) if x is not None],layout="spring", show_weights = False,)