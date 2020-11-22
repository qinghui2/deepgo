from Bio import Phylo
import networkx as nx
import dendropy

## get the node2vec tree embedding from a phylogenetic tree
##input: the tree file in nwk format
# path_node2vec the node2vec folder path
#outname output file path name
## output: the node2vec embedding for each node in the tree
def get_tree_embedding_node_2_vec(input, path_node2vec,outname):
    command = "python"+path_node2vec+"src/main.py"+" --input " + input +" --output "+outname


## output edge list of a newick tree for embedding
#input is the tree file in nwk format
# output map: map node_id(int) to taxon name for only leaf nodes
# output edge list: id0 id1 weight

def convert(input):
    map = {} # node object -> id
    parent_map = {} # map of node to its parent
    len_map = {} #map of edge length between a node and its parent
    taxon_label_map = {} # if a node is a taxon, map node -> node_label
    taxon_label_map_inv = {} # if a node is a taxon, map node_label -> node
    taxa = dendropy.TaxonNamespace() #taxon namespace
    cur_id = 0
    out_map_file = open("data/tree_embds/tree_node_mapping.txt","w")
    # tree = Phylo.read(input, "newick")
    tree = dendropy.Tree.get(path=input, schema="newick",taxon_namespace=taxa)
    # a = 2
    for n in tree.preorder_node_iter():
        cur_id = cur_id + 1
        if n.parent_node == None:
            map[n] = cur_id
        else:
            map[n] = cur_id
            parent_id = map[n.parent_node]
            parent_map[n] = n.parent_node
            len_map[n] = n.edge_length
            if n.is_leaf():
                # print(n.taxon)
                taxon_label_map[n] = n.taxon
                taxon_label_map_inv[str(n.taxon).strip('\'')] = n
    # write taxon name, taxon id
    with open("data/tree_embds/taxon_map.txt", "w") as f:
        for n in taxon_label_map_inv:
            f.write(n+" "+str(map[taxon_label_map_inv[n]])+"\n")
        f.close()
    # write the edge list
    with open("data/tree_embds/edge_list.txt","w") as f:
        for n in parent_map:
            par_id = map[parent_map[n]]
            cur_id = map[n]
            len_to_par = len_map[n]*100
            if(len_to_par <= 0):
                len_to_par = 1
            f.write(str(par_id)+" "+str(cur_id)+" "+str(len_to_par)+"\n")
        f.close()
    G = nx.read_weighted_edgelist("data/tree_embds/edge_list.txt")
    # c = nx.is_connected(G)
    # b = 2

# #input: MSA from DNA data
# def generate_test_data(input):
#     pass

convert("data/test_tree.nwk")
# # # print(tree)
# tree.ladderize()
# Phylo.draw(tree)
