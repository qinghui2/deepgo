from karateclub import DeepWalk
import networkx as nx
import gzip

"""
THIS FILE IS UNTESTED
I believe the heterogenous PPI network used to create the network embeddings is the data/graph.mapping.out.gz file.
It is used to populate the deep_map dictionary. We use the deep_map dictionary to instead create a NetworkX graph
and run deepwalk on it. The dimensions can be changed if needed.
"""


# load the model
def deepwalk_embedding(Graph):
    dw = DeepWalk(dimensions=128)
    dw.fit(Graph)
    embeddings = dw.get_embedding()
    return embeddings

def create_graph():
    # mapping = dict()
    # with open('data/string2uni.tab') as f:
    #     for line in f:
    #         it = line.strip().split('\t')
    #         mapping[it[0].upper()] = it[1]
    #
    # with open('data/string_idmapping.dat') as f:
    #     for line in f:
    #         it = line.strip().split('\t')
    #         mapping[it[2].upper()] = it[0]
    #
    # with open('data/uniprot-string.tab') as f:
    #     for line in f:
    #         it = line.strip().split('\t')
    #         mapping[it[1].upper()[:-1]] = it[0]

    deep_map = dict()

    with gzip.open('data/graph.mapping.out.gz') as f:
        for line in f:
            it = line.strip().split('\t')
            deep_map[it[1]] = it[0]
    G = nx.DiGraph()
    G.add_nodes_from(deep_map.keys())
    for k, v in deep_map.items():
        G.add_edges_from(([(k, t) for t in v]))
    return G


if __name__ == '__main__':
    ppi_graph = create_graph()
    embeddings = deepwalk_embedding(ppi_graph)

