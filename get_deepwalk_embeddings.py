import csv
import pandas as pd
from karateclub import DeepWalk
import networkx as nx
import gzip
import stringdb

"""
THIS FILE IS UNTESTED
I believe the heterogenous PPI network used to create the network embeddings is the data/graph.mapping.out.gz file.
It is used to populate the deep_map dictionary. We use the deep_map dictionary to instead create a NetworkX graph
and run deepwalk on it. The dimensions can be changed if needed.
"""


# load the model
def deepwalk_embedding(Graph):
    dw = DeepWalk(dimensions=128)
    G = nx.convert_node_labels_to_integers(Graph)
    dw.fit(G)
    embeddings = dw.get_embedding()
    print(embeddings)
    return embeddings

def create_graph():
    deep_map = dict()

    # with gzip.open('data/graph.mapping.out.gz') as f:
    df = pd.read_csv('data/train-mf.csv')
    proteins = list(df['proteins'].unique())
    string_ids = stringdb.get_string_ids(proteins)
    print(string_ids)
    network_df = stringdb.get_network(string_ids.queryItem)
    print(network_df.columns)
    print(network_df.head(1))
    for idx in range(len(network_df)):
        A = network_df['stringId_A'].iloc[idx]
        B = network_df['stringId_B'].iloc[idx]
        deep_map[B] = A

    # tsv_file = open("data/string_interactions.tsv")
    # read_tsv = csv.reader(tsv_file, delimiter="\t")
    # print("read tsv")
    # first = True
    # for line in read_tsv:
    #     if first:
    #         first = False
    #         continue
    #     # it = line.strip().split('\t')
    #     # deep_map[it[1]] = it[0]
    #     deep_map[line[1]] = line[0]
    print(deep_map)

    G = nx.Graph()
    G.add_nodes_from(deep_map.keys())
    for k, v in deep_map.items():
        G.add_edges_from(([(k, t) for t in v]))
    return G


if __name__ == '__main__':
    ppi_graph = create_graph()
    embeddings = deepwalk_embedding(ppi_graph)

