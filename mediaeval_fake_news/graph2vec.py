from karateclub import Graph2Vec
import numpy as np
import pandas as pd
import networkx as nx

def get_weighted_average_embeddings(graphs, num_dimensions=64):
    """ uses the graph2vec package to get embeddings for the input graph
    returns a dataframe with the average embeddings for each graph's subgraph weighted by the nodes in that subgraph
    """
    # fits the graph2vec model
    all_subgraphs = []
    for G in graphs:
      subgraphs = [G.to_undirected().subgraph(cc).copy() for cc in nx.connected_components(G.to_undirected())]
      all_subgraphs.extend( subgraphs )
    all_subgraphs = [nx.convert_node_labels_to_integers(g) for g in all_subgraphs]
    g2v_model = Graph2Vec(dimensions=num_dimensions)
    g2v_model.fit(all_subgraphs)
    embeddings = g2v_model.get_embedding()

    weighted_averages = np.array([])
    start = 0
    for G in graphs:
      subgraph_count = nx.number_connected_components(G.to_undirected())
      graph_embeddings = embeddings[start : start + subgraph_count]
      if subgraph_count > 1:
        weights = [G.to_undirected().subgraph(cc).copy().number_of_nodes() for cc in nx.connected_components(G.to_undirected())]
        # print(weights)
        # print(graph_embeddings.shape)
        weighted_average = np.average(graph_embeddings, axis=0, weights = weights)
        weighted_average = np.reshape(weighted_average, newshape=(1,num_dimensions))
      else:
        weighted_average = graph_embeddings
      if start == 0:
        weighted_averages = np.array(weighted_average)
      else:
        weighted_averages = np.append(weighted_averages, weighted_average, axis=0)
      start += subgraph_count
    weighted_g2v = pd.DataFrame(data=weighted_averages, columns=["weighted_g2v" + str(i) for i in range(0,num_dimensions)])
    return weighted_g2v

def get_largest_subgraph_embedding(graphs, num_dimensions=64):
    """
    does essentially the same thing as the above function but only for the largest subgraph of each graph
    """
    largest = [G.to_undirected().subgraph(max(nx.connected_components(G.to_undirected()), key=len)).copy() for G in graphs]
    largest = [nx.convert_node_labels_to_integers(g) for g in largest]
    model = Graph2Vec(dimensions=num_dimensions)
    model.fit(largest)
    embeddings = model.get_embedding()
    g2v = pd.DataFrame(data=embeddings, columns=["g2v_largest" + str(i) for i in range(0,num_dimensions)])
    return g2v
