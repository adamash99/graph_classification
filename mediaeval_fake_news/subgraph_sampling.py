import pandas as pd
import random
import numpy as np
import networkx as nx

def subgraph_random_sampling(graphs, label, num_iterations=25, num_nodes_sampled=10):
    """
    samples num_nodes_sampled nodes from each graph in graphs. Does this num_iterations times
    returns a dataframe with a label column. each row is a sampled adjacency matrix
    """
    to_concat = []
    for _ in range(num_iterations):
        subgraph_df = pd.DataFrame([get_adj_matrix_helper(g, num_nodes_sampled) for g in graphs])
        subgraph_df['label'] = label
        to_concat.append(subgraph_df)
    all_subgraphs = pd.concat(to_concat)
    return all_subgraphs

def get_adj_matrix_helper(graph, num_nodes_sampled=10):
    """
    randomly samples num_nodes_sampled nodes from graph and returns the adjacency matrix
    """
    all_nodes = list(graph.nodes)
    if len(all_nodes) >= num_nodes_sampled:
        ten_nodes = random.sample(all_nodes, num_nodes_sampled)
        subgraph = graph.subgraph(ten_nodes).copy()
        adj_matrix = nx.convert_matrix.to_numpy_array(subgraph)
    else:
        adj_matrix = nx.convert_matrix.to_numpy_array(graph)
        diff = 10-adj_matrix.shape[0]
        adj_matrix = np.pad(adj_matrix, pad_width=(0, diff), constant_values=0)
    return np.reshape(adj_matrix, 100)


