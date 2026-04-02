import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches

###############################################################################################################
           ######################## ---- Function to plot the graph ---- ########################
###############################################################################################################
def plot_weighted_graph(
    G: nx.Graph,
    providers: list | None = None,
    client: int | None = None,
    paths: list[list] | None = None,
    layout: str = "spring",
    node_size: int = 700,
    figsize: tuple = (8, 6),
    show_weights: bool = True,
    seed: int | None = 42,
    base_edge_width: float = 1.5,
    path_edge_width: float = 2.5,
    overlap_width_gain: float = 2.0,
):
    """
    Plot a weighted graph with special coloring for providers/client and
    optional highlighted paths.

    Parameters
    ----------
    G : nx.Graph
        Graph to visualize.
    providers : list | None
        List of provider nodes.
    client : int | None
        Client node.
    paths : list[list] | None
        List of paths, where each path is a list of nodes.
        Example: [[0,1,4], [2,1,4]]
    layout : str
        One of: 'spring', 'circular', 'kamada_kawai', 'shell', 'spectral'.
    node_size : int
        Size of nodes in the plot.
    figsize : tuple
        Figure size.
    show_weights : bool
        Whether to display edge weights.
    seed : int | None
        Seed used by layouts that support it.
    base_edge_width : float
        Width of non-highlighted edges.
    path_edge_width : float
        Base width of highlighted path edges.
    overlap_width_gain : float
        Extra width added for each additional overlap on an edge.

    Notes
    -----
    If an edge appears in multiple paths, its width is increased to show overlap.
    """
    plt.figure(figsize=figsize)

    # ---- Layout ----
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError("Unknown layout.")

    providers = set(providers or [])
    paths = paths or []

    # ---- Validate paths and count edge overlaps ----
    edge_counter = Counter()
    path_edges_list = []

    for path in paths:
        if len(path) < 2:
            path_edges_list.append([])
            continue

        path_edges = []
        for u, v in zip(path[:-1], path[1:]):
            if not G.has_edge(u, v):
                raise ValueError(f"Edge ({u}, {v}) from path {path} is not in the graph.")
            e = tuple(sorted((u, v)))  # undirected canonical form
            path_edges.append(e)
            edge_counter[e] += 1
        path_edges_list.append(path_edges)

    highlighted_edges = set(edge_counter.keys())
    normal_edges = [e for e in G.edges() if tuple(sorted(e)) not in highlighted_edges]

    # ---- Assign node colors ----
    node_colors = []
    for node in G.nodes():
        if node == client:
            node_colors.append("red")
        elif node in providers:
            node_colors.append("green")
        else:
            node_colors.append("lightblue")

    # ---- Draw non-highlighted edges first ----
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        width=base_edge_width,
        edge_color="lightgray",
        alpha=0.8
    )

    # ---- Draw highlighted edges with overlap-dependent thickness ----
    if highlighted_edges:
        highlighted_edgelist = list(highlighted_edges)
        highlighted_widths = [
            path_edge_width + overlap_width_gain * (edge_counter[e] - 1)
            for e in highlighted_edgelist
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=highlighted_edgelist,
            width=highlighted_widths,
            edge_color="tab:orange",
            alpha=0.9
        )

    # ---- Draw nodes and labels ----
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size
    )
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10
    )

    # ---- Edge weights ----
    if show_weights:
        edge_labels = {
            (u, v): f"{d['weight']:.2f}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # ---- Legend ----
    legend_elements = [
        mpatches.Patch(color="green", label="Providers"),
        mpatches.Patch(color="red", label="Client"),
        mpatches.Patch(color="lightblue", label="Other nodes"),
    ]
    if paths:
        legend_elements.append(mpatches.Patch(color="tab:orange", label="Selected paths"))

    plt.legend(handles=legend_elements, loc="best")
    plt.title("Connected Weighted Graph with Roles and Paths")
    plt.axis("off")
    plt.show()