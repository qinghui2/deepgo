
import os

functions = ['bp', 'mf','cc']
method = 'deep'
seeds = ['0', '1310']


for f in functions:
    for s in seeds:
                exp_id = "results/"+str(f)+'-'+str("tree")+'-'+str(s)
                if os.path.isdir(exp_id+"/evaluation"):
                    continue
                cmd = 'python3 nn_hierarchical_network.py '+'--function '+f+' --shuffleseed '+s+' --embeddingmethod deep'
                print(cmd)
                os.system(cmd)