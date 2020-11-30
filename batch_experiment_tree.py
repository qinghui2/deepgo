import os

functions = ['bp', 'mf','cc']
method = 'tree'
seeds = ['0', '1310']
evo_model = ['P','F',"D"]
build_methods = ['B','O','D']

for f in functions:
    for s in seeds:
        for e in evo_model:
            for b in build_methods:
                cmd = 'python3 nn_hierarchical_network.py '+'--function '+f+' --shuffleseed '+s+' --embeddingmethod tree' + ' --evomodel '+e+' --buildmethod '+b
                print(cmd)
                os.system(cmd)