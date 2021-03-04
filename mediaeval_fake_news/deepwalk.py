import networkx as nx

#NOTE: the deepwalk script needs to be run from the command line
# see https://github.com/phanein/deepwalk

 

def clean_for_deepwalk(dataset_path, outfile='for_deepwalk'):
    """
    cleans graphs for deepwalk
    make a single (disconnected) graph that is the compilation of every graph in order to run deepwalk on it.
    """

    G = nx.Graph()

    nodes = []
    for i in range(1,271): # 270 total
        conspiracy_path = dataset_path + "5g_corona_conspiracy/"
        edges = open(conspiracy_path + str(i)+ "/edges.txt")
        for pair in edges:
            node1, node2 = pair.split()[0], pair.split()[1]
            G.add_edge(node1, node2)
            if node1 not in nodes:
                nodes.append(node1)
            if node2 not in nodes:
                nodes.append(node2)
        edges.close()
    print(G.number_of_nodes())
    print(len(nodes))

    nodes = []
    for i in range(1,398): # 397 total
        path = dataset_path + "other_conspiracy/"
        edges = open(path + str(i)+ "/edges.txt")
        for pair in edges:
            node1, node2 = pair.split()[0], pair.split()[1]
            G.add_edge(node1, node2)
            if node1 not in nodes:
                nodes.append(node1)
            if node2 not in nodes:
                nodes.append(node2)
        if i % 100 == 0:
            print("added graph " + str(i))
        edges.close()
    print(G.number_of_nodes())
    print(len(nodes))

    nodes = []
    for i in range(1,1661): # 1660 total
        path = dataset_path + "non_conspiracy/"
        edges = open(path + str(i)+ "/edges.txt")
        for pair in edges:
            node1, node2 = pair.split()[0], pair.split()[1]
            G.add_edge(node1, node2)
            if node1 not in nodes:
                nodes.append(node1)
            if node2 not in nodes:
                nodes.append(node2)
        if i % 100 == 0:
            print("added graph " + str(i))
        edges.close()
    print(G.number_of_nodes())
    print(len(nodes))


    print(G.number_of_edges())
    nx.write_edgelist(G, outfile + ".edgelist")
