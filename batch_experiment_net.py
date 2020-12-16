import os

functions = ['bp', 'mf','cc']
method = 'tree'
seeds = ['0', '1310']

for f in functions:
    for s in seeds:
        exp_id = "results/"+str(f)+'-'+str("tree")+'-'+str(s)
        if os.path.isdir(exp_id+"/evaluation"):
            continue
        if os.path.isfile(exp_id+"/tree_embeddings/tree_node_embedding/emb.emb"):
            cmd = cmd = 'python3 nn_hierarchical_network.py --cached 1'+' --function '+f+' --shuffleseed '+s+' --embeddingmethod net' + ' --evomodel '+e+' --buildmethod '+b
        else:
            cmd = 'python3 nn_hierarchical_network.py '+'--function '+f+' --shuffleseed '+s+' --embeddingmethod net'
        print(cmd)
        os.system(cmd)