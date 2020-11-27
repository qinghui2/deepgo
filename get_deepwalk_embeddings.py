from karateclub import DeepWalk
import networkx as nx
import dendropy

# load the model
def deepwalk_embedding(Graph):
    dw = DeepWalk(dimensions=128)
    dw.fit(Graph)
    embeddings = dw.get_embedding()
    return embeddings

def create_graph(input):
    map = {} # node object -> id
    parent_map = {} # map of node to its parent
    len_map = {} #map of edge length between a node and its parent
    taxa = dendropy.TaxonNamespace() #taxon namespace
    cur_id = 0
    tree = dendropy.Tree.get(path=input, schema="newick",taxon_namespace=taxa)
    for n in tree.preorder_node_iter():
        cur_id = cur_id + 1
        if n.parent_node == None:
            map[n] = cur_id
        else:
            map[n] = cur_id
            parent_id = map[n.parent_node]
            parent_map[n] = n.parent_node
            len_map[n] = n.edge_length
    # write the edge list
    with open("data/deepwalk_embds/edge_list.txt","w") as f:
        for n in parent_map:
            par_id = map[parent_map[n]]
            cur_id = map[n]
            len_to_par = len_map[n]*100
            if(len_to_par <= 0):
                len_to_par = 1
            f.write(str(par_id)+" "+str(cur_id)+" "+str(len_to_par)+"\n")
        f.close()
    G = nx.read_weighted_edgelist("data/deepwalk_embds/edge_list.txt")
    return G

tree_graph = create_graph("data/test_tree.nwk")
embeddings = deepwalk_embedding(tree_graph)

