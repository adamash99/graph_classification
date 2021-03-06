import networkx as nx
import numpy as np
import pandas as pd

def prepare_graph(label, nodes, edges):
    """
    takes in graph information and returns a dictionary of basic features for that graph
    as well as the graph as a networkx graph
    
    Arguements:
    label - class label of the graph
    nodes - csv of graph nodes 
    edges = txt file of graph edges
    """
    features = {'label': label}

    G = nx.DiGraph()
    nodes["id"] = nodes["id"].apply(lambda x : str(x))
    features['num_nodes'] = nodes.shape[0]
    op_node = None
    times = []
    friends = []
    followers = []
    for index, row in nodes.iterrows():
      G.add_node(row['id'], time=row['time'], friends=row['friends'], followers = row['followers'])
      times.append(row['time'])
      friends.append(2**row['friends'])
      followers.append(2**row['followers'])
      if row['time'] == 0:
        features['poster_friend_cnt'] = 2**row['friends']
        features['poster_follower_cnt'] = 2**row['followers']
        tweeter_id = row['id']
        op_node = row['id']
    features['avg_time'] = np.mean(times)
    features['avg_friends'] = np.mean(friends)
    features['avg_followers'] = np.mean(followers)
    features['max_followers'] = max(followers)
    features['max_friends'] = max(friends)
    features['friends_25th_percentile'] = np.percentile(friends, 25)
    features['friends_75th_percentile'] = np.percentile(friends, 75)
    features['followers_25th_percentile'] = np.percentile(followers, 25)
    features['followers_75th_percentile'] = np.percentile(followers, 75)
    node_list = []
    edge_count = 0
    for pair in edges:
      node1, node2 = pair.split()[0], pair.split()[1]
      node_list.append(node1)
      node_list.append(node2)
      G.add_edge(node1, node2)
      edge_count += 1
    features['num_edges'] = edge_count
    sum_users_without_followers = sum([1 for (node, val) in G.in_degree() if val==0])
    features['ratio_users_w/out_followers'] = sum_users_without_followers / len(G.nodes)
    features['num_connected_components'] = nx.number_strongly_connected_components(G)
    features['number_of_OPs_followers_who_retweeted'] = G.in_degree(tweeter_id)
    features['percentage_of_OPs_followers_who_retweeted'] = G.in_degree(tweeter_id) / features['poster_follower_cnt']
    features['avg_clustering'] = nx.average_clustering(G)
    features['op_clustering'] = nx.clustering(G,op_node)
    features['transitivity'] = nx.transitivity(G)
    node_list = list(set(node_list))
    features['nodeID_list'] = np.array(node_list)
    return features, G

def get_training_dataframe(dataset_path, two_class=True):
    """
    goes through the provided dataset and creates a dataframe with a row for each graph.
    columns of the dataframe are the basic features defined above
    returns the dataframe and a dictionary mapping labels to lists of graphs for that label

    args:
    two_class - if true, return a dataframe for 2 class classification (conspiracy class is positive)
    if false, return a dataframe for 3 class classification (other conspiracy category is its own class)
    this arguement changes the label in the dataframe and nothing else
    """
    training_df = pd.DataFrame()
    graph_dict = {}

    to_add = []
    
    # add conspiracy graphs to dataframe
    conspiracy_graphs = []
    for i in range(1,271): # 270 total
        conspiracy_path = dataset_path + "5g_corona_conspiracy/"
        nodes = pd.read_csv(conspiracy_path + str(i)+ "/nodes.csv")
        edges = open(conspiracy_path + str(i)+ "/edges.txt")

        features, G = prepare_graph(1, nodes, edges)
        to_add.append(features)
        edges.close()
        conspiracy_graphs.append(G)
    graph_dict['conspiracy_graphs'] = conspiracy_graphs

    non_conspiracy_graphs = []
    for i in range(1,1661): # 1660 total
        path = dataset_path + "non_conspiracy/"
        nodes = pd.read_csv(path + str(i)+ "/nodes.csv")
        edges = open(path + str(i)+ "/edges.txt")

        label = 0 if two_class else 3
        features, G = prepare_graph(label, nodes, edges)
        to_add.append(features)
        edges.close()
        non_conspiracy_graphs.append(G)
    graph_dict['non_conspiracy_graphs'] = non_conspiracy_graphs

    other_conspiracy_graphs = []
    for i in range(1,398): # 397 total
        path = dataset_path + "other_conspiracy/"
        nodes = pd.read_csv(path + str(i)+ "/nodes.csv")
        edges = open(path + str(i)+ "/edges.txt")

        label = 0 if two_class else 2
        features, G = prepare_graph(label, nodes, edges)
        to_add.append(features)
        edges.close()
        other_conspiracy_graphs.append(G)
    graph_dict['other_conspiracy_graphs'] = other_conspiracy_graphs

    training_df = training_df.append(to_add)
    return training_df, graph_dict