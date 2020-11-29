import csv
import pandas as pd
from karateclub import DeepWalk
import networkx as nx
import numpy as np
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
    return embeddings

def generate_embeddings():
    access_map = dict()  # accession -> {}
    networks_map = dict()
    df = pd.read_csv('data/df_head_1000.csv')
    # df = df.head(5)
    indexes = list(df['index'])
    for index in indexes:
        access_map[index] = {}
    accessions = list(df['accessions'].unique())
    print("accession list", len(accessions))
    for idx, accession in enumerate(accessions):
        # print("idx:", idx, "accession:", accession)
        index_val = df['index'].iloc[idx]
        net_df = stringdb.get_network(accession)

        access_map[index_val]['accession'] = accession
        networks_map[index_val] = net_df
        if len(net_df) == 0:
            # print("zero vector")
            access_map[index_val]['deepwalk_embedding'] = np.zeros(128)
        else:
            # print("net_len:", len(net_df))
            deep_map = dict()
            for ix in range(len(net_df)):
                A = net_df['stringId_A'].iloc[ix]
                B = net_df['stringId_B'].iloc[ix]
                deep_map[B] = A
            G = nx.Graph()
            G.add_nodes_from(deep_map.keys())
            for k, v in deep_map.items():
                G.add_edges_from(([(k, t) for t in v]))
            embedding = deepwalk_embedding(G)
            # print(embedding[0].shape)
            access_map[index_val]['deepwalk_embedding'] = embedding[0]
    print(len(access_map))
    full_df = pd.DataFrame.from_dict(access_map, orient='columns').T
    return full_df

    # string_ids = stringdb.get_string_ids(proteins)
    # print("string ids", len(string_ids))

    # network_df = stringdb.get_network(accessions)
    # print(network_df.columns)
    # print(len(network_df))
    # for idx in range(len(network_df)):
    #     A = network_df['stringId_A'].iloc[idx]
    #     B = network_df['stringId_B'].iloc[idx]
    #     deep_map[B] = A
    #
    # print(len(deep_map))
    # G = nx.Graph()
    # G.add_nodes_from(deep_map.keys())
    # for k, v in deep_map.items():
    #     G.add_edges_from(([(k, t) for t in v]))
    # return G


if __name__ == '__main__':
    embeddings_df = generate_embeddings()
    embeddings_df.to_csv('deepwalk_embeddings.csv')
    # embeddings = deepwalk_embedding(ppi_graph)
    # with open("deepwalk_embeddings.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(embeddings)

