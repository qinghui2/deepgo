#!/usr/bin/env python

"""
python nn_hierarchical_network.py
"""

import numpy as np
import pandas as pd
import click as ck
import os
from Bio import SeqIO
import subprocess
import pexpect
import networkx as nx
import dendropy
from sklearn.utils import shuffle
from shutil import copyfile
# from node2vec import Node2Vec

from keras.models import Sequential, Model, load_model
from keras.layers import (
    Dense, Dropout, Activation, Input,
    Flatten, Highway, merge, BatchNormalization)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (
    Convolution1D, MaxPooling1D)
from keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.metrics import classification_report
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    get_parents,
    DataGenerator,
    FUNC_DICT,
    MyCheckpoint,
    save_model_weights,
    load_model_weights,
    get_ipro)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras import backend as K
import sys
from collections import deque
import time
import logging
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from multiprocessing import Pool
import get_deepwalk_embeddings

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/swiss/'
MAXLEN = 1000
REPLEN = 256
ind = 0


@ck.command()
@ck.option(
    '--function',
    default='mf',
    help='Ontology id (mf, bp, cc)')
@ck.option(
    '--device',
    # default='gpu:0',
    default='cpu:0', # use cpu instead

    help='GPU or CPU device id')
@ck.option(
    '--org',
    default=None,
    help='Organism id for filtering test set')
@ck.option('--train', is_flag=True)
@ck.option('--param', default=0, help='Param index 0-7')
@ck.option(
    '--evomodel',
    help='protein evolution model, P-dist(P), F81-like(F), LG(L), Default(D) ',
    default = 'default'
)
@ck.option(
    '--buildmethod',
    help = 'TaxAdd_BalME(B), TaxAdd_OLSME(O), BIONJ(I), NJ(N), Default(D)',
    default = 'default'
)
@ck.option(
    '--shuffleseed',
    default = 0,
    help = 'shuffle seed of permuting the dataframe, a shuffle seed 0 means no shuffle'
)
@ck.option(
    '--embeddingmethod',
    help = 'original network embedding(net), tree based embedding(tree), deepwalk embedding(deep), fixed embedding all initialized to 0.1(fix)',
    default = 'net'
)
def main(function, device, org, train, param,embeddingmethod, shuffleseed,buildmethod, evomodel):
    global BUILDMETHOD
    BUILDMETHOD = buildmethod
    global EVOMODEL
    EVOMODEL = evomodel
    global EMBEDDINGMETHOD
    EMBEDDINGMETHOD = embeddingmethod
    global SEED
    SEED = shuffleseed
    global FUNCTION
    FUNCTION = function
    global GO_ID
    GO_ID = FUNC_DICT[FUNCTION]
    global go
    go = get_gene_ontology('go.obo')
    global ORG
    ORG = org
    func_df = pd.read_pickle(DATA_ROOT + FUNCTION + '.pkl')
    global functions
    functions = func_df['functions'].values
    global func_set
    func_set = set(functions)
    global all_functions
    all_functions = get_go_set(go, GO_ID)
    global experiment_id
    experiment_id = str(function)+'-'+str(embeddingmethod)+'-'+str(shuffleseed)+'-'+str(buildmethod)+'-'+str(evomodel)
    logging.info('Functions: %s %d' % (FUNCTION, len(functions)))
    a = experiment_id
    global resdir
    resdir = "results/"+experiment_id
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    if ORG is not None:
        logging.info('Organism %s' % ORG)
    global go_indexes
    go_indexes = dict()
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind
    global node_names
    node_names = set()
    with tf.device('/' + device):
        params = {
            'fc_output': 1024,
            'learning_rate': 0.001,
            'embedding_dims': 128,
            'embedding_dropout': 0.2,
            'nb_conv': 3,
            'nb_dense': 2,
            'filter_length': 128,
            'nb_filter': 32,
            'pool_length': 64,
            'stride': 32
        }
        # model(params, is_train=train)
        model(params, is_train=True)
        # dims = [64, 128, 256, 512]
        # nb_filters = [16, 32, 64, 128]
        # nb_convs = [1, 2, 3, 4]
        # nb_dense = [1, 2, 3, 4]
        # for i in range(param * 32, param * 32 + 32):
        #     dim = i % 4
        #     i = i / 4
        #     nb_fil = i % 4
        #     i /= 4
        #     conv = i % 4
        #     i /= 4
        #     den = i 
        #     params['embedding_dims'] = dims[dim]
        #     params['nb_filter'] = nb_filters[nb_fil]
        #     params['nb_conv'] = nb_convs[conv]
        #     params['nb_dense'] = nb_dense[den]
            # f = model(params, is_train=train)
            # print(dims[dim], nb_filters[nb_fil], nb_convs[conv], nb_dense[den], f)
    # performanc_by_interpro()
#Author: Qinghui Zhou
#extract taxa from dataframe(each taxa is a protein sequence)
#if the df is train df, return 20179 sequences, for tree reconstruct
# return a dictionary where key is the id of the protein sequence and value is the protein sequence
def extract_taxa(df):
    # df_cp = df
    # t =2
    # pass
    taxa_dict = {}
    for index, row in df.iterrows():
        taxa_dict[index] = df.loc[index, 'sequences']
   # r = 2
    return taxa_dict
#author:Qinghui Zhou
## output edge list of a newick tree for embedding
#input is the tree file in nwk format
# output map: map node_id(int) to taxon name for only leaf nodes
# output edge list: id0 id1 weight
def convert(input):
    tree_res_dir = resdir+"/tree_embeddings"
    if(not os.path.isdir(tree_res_dir)):
        os.mkdir(tree_res_dir)
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
    copyfile("data/tree_embds/taxon_map.txt",tree_res_dir+"/taxon_mapping.txt")
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
    copyfile("data/tree_embds/edge_list.txt",tree_res_dir+"/edge_list.txt")
    G = nx.read_weighted_edgelist("data/tree_embds/edge_list.txt")
    c = nx.is_connected(G)
    b = 2
    return  G
#Author: Qinghui Zhou
# given a set of sequences defined by df, generate the multiple sequence alignment
# df: merged train and test df
# path_to_alignment: the path to alignment executable
def get_tree_emb(df,path_to_alignment):
    tree_res_dir = resdir + "/tree_embeddings"
    if(not os.path.isdir(tree_res_dir)):
        os.mkdir(tree_res_dir)
    msa_path = tree_res_dir+"/msa"
    if(not os.path.isdir(msa_path)):
        os.mkdir(msa_path)
    # write a fasta file for the input sequences in df
    print("writting sequence file:")
    out_f = open("temp_output/sequence_file_concatednated.fasta","w")
    out_f_name = "temp_output/sequence_file_concatednated.fasta"
    for k in df:
        out_f.write(">"+str(k)+"\n")
        out_f.write(df[k]+"\n")
    out_f.close()
    #get msa with clustalomega/muscle
    print("getting multiple sequence alignment")
    aln_out_name = "temp_output/alignment_result.fasta"
    # if(not os.path.isfile(aln_out_name)):
    # reference: muscle downloaded from: https://www.drive5.com/muscle/
    # muscle reference: Edgar RC. MUSCLE: multiple sequence alignment with high accuracy and high throughput. Nucleic Acids Res. 2004 Mar 19;32(5):1792-7. doi: 10.1093/nar/gkh340. PMID: 15034147; PMCID: PMC390337.
    cmd = "./"+path_to_alignment+" -in "+out_f_name+" -out "+aln_out_name+" -maxiters 3"#set iter to 1 for testing
    print(cmd)
    os.system(cmd)
    copyfile(aln_out_name, msa_path+"/msa_result.fasta")
    copyfile(out_f_name,msa_path+"/msa_in.fasta")
    # else:
    #     print("there is existing alignment, using cached")
    # convert fasta to phylip
    records = SeqIO.parse(aln_out_name, "fasta")
    aln_out_name_phy = "temp_output/alignment_result.phylip"
    count = SeqIO.write(records, aln_out_name_phy, "phylip")
    print("Converted %i records" % count)
    #get phylogenetic tree from input alignment
    print("getting phylogenetic tree")
    #get node embeddings for leaves of the tree

    tree_name = "temp_output/alignment_result.phylip_fastme_tree.txt"
    stat_name = "temp_output/alignment_result.phylip_fastme_stat.txt"
    # if(True or not os.path.isfile(tree_name)):
    if(os.path.isfile(tree_name)):
        os.remove(tree_name)
    if(os.path.isfile(stat_name)):
        os.remove(stat_name)
    # reference of fastme: PhyML : "A simple, fast, and accurate algorithm to estimate large phylogenies by maximum likelihood."
    # code of fastme downloaded from: http://www.atgc-montpellier.fr/fastme/binaries.php
    # using pexpect to spawn fastme was referenced from https://github.com/c5shen/CS581HW3/blob/master/FastME-TaxAdd_BalME/pipeline_fastme.py
    # and https://github.com/c5shen/CS581HW3/blob/master/NJ/pipeline_nj.py
    c = pexpect.spawn('fastme')
    c.sendline(aln_out_name_phy)
    c.sendline('I')
    c.sendline('P')
    # BUILDMETHOD = buildmethod
    # global EVOMODEL
    # EVOMODEL = evomodel
    if(EVOMODEL!="D"):
        c.sendline("E")
        c.sendline(EVOMODEL)
        print("evolution model", EVOMODEL)
    if(BUILDMETHOD!="D"):
        c.sendline("M")
        c.sendline(BUILDMETHOD)
    c.sendline('Y')
    c.interact()
    G = convert(tree_name)# get edge list
    tree_dir = tree_res_dir+"/tree"
    if(not os.path.isdir(tree_dir)):
        os.mkdir(tree_dir)
    copyfile(tree_name, tree_dir+"/tree.txt")
    copyfile(stat_name,tree_dir+"/stat.txt")
    # else:
    #     G = convert(tree_name)  # get edge list
    #     print("there is already tree file, using cached")
    #get node embedding
    emb_file = "data/tree_embds/edge_list.txt"
    from node2vec import Node2Vec #import here to test previous
    node2vec = Node2Vec(G, dimensions=256, walk_length=30, num_walks=200, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.most_similar('2')  # Output node names are always strings
    model.wv.save_word2vec_format("temp_output/emb.emb")
    model.save("temp_output/emb.model")
    tree_node_emb_dir = tree_res_dir+"/tree_node_embedding"
    if(not os.path.isdir(tree_node_emb_dir)):
        os.mkdir(tree_node_emb_dir)
    copyfile("temp_output/emb.emb",tree_node_emb_dir+"/emb.emb")
    copyfile("temp_output/emb.model",tree_node_emb_dir+"/emb.model")

#Author: Qinghui Zhou
#replace the embedding(network) with fixed vector, n>0 and n<1
def replace_with_fixed_embedding(df,n):
    # emb_frame = df['embeddings']#.to_frame()
    # d = 2
   # pass
    for index, row in df.iterrows():
        embd = row['embeddings']
        fixed_vec = n*np.ones_like(embd)
        ind = index
        #embd[str(index)]['embeddings']=fixed_vec
        # r = df[14149]['embeddings']
        df.loc[index, 'embeddings'] = fixed_vec
        #t = 1
    return df
    #
    # for index, row in df.iterrows():
    #     embd = row['embeddings']


#Author: Qinghui Zhou
# replace the embedding with tree based embeddings
# the input embedding file should be in format id, emb, ...
def replace_with_tree_based_embedding(df):
    # pass
    emb_f_name = "temp_output/emb.emb"
    emb_file = open(emb_f_name,"r")
    next(emb_file)
    # read mapping
    emb_mapping = dict()

    for line in emb_file.readlines():
        # c = line
        l_arr = line.split(" ")
        node_id, emb = l_arr[0], l_arr[1:]
        d = 2
        emb = list(map(float, emb))
        emb_mapping[node_id] = emb
    # z = emb_mapping['8']
    # z1 = emb_mapping['12']
    # z2 = emb_mapping['14']
    # z3 = emb_mapping['15']
    # map the embedding mapping back to original id
    taxon_mapping = open("data/tree_embds/taxon_map.txt","r")
    true_emb_mapping = dict()
    s = taxon_mapping.readlines()
    for k in s:
        orig, mapped = k.split(" ")[0], k.split(" ")[1].rstrip()
        true_emb_mapping[orig] = emb_mapping[mapped]
    # z1 = true_emb_mapping['14149']
    # z2 = true_emb_mapping['3127']
    # z3 = true_emb_mapping['26305']
    # z4 = true_emb_mapping['21094']
    # replace the embedding with tree embedding in df
    c =1
    for index, row in df.iterrows():
        embd = row['embeddings']

        ind = index
        fixed_vec = np.ones_like(embd)
        # r = df[14149]['embeddings']
        tree_emb_ind = true_emb_mapping[str(ind)]
        df.loc[index, 'embeddings'] = tree_emb_ind
    t = 2
    return df

"""
@Author: Sejal K. Parmar
replace embedding with deepwalk embeddings
"""
def replace_with_deepwalk_embeddings(df):
    embedding_df = get_deepwalk_embeddings.generate_embeddings(df)
    df.embeddings = embedding_df.deepwalk_embedding
    return df




def load_data(org=None):
    # FUNCTION ="bp"
    #size 25224
    df = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    test_df = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    # df = pd.concat([df, test_df], ignore_index=True)
    # df = shuffle(df,random_state=20)
    if(int(SEED)!=0):
        logging.info("shuffling with seed")
        print(int(SEED))
        df = shuffle(df,random_state=int(SEED))
    else:
        logging.info("not shuffinling data")
    df = df.head(1000)
    if(EMBEDDINGMETHOD == 'tree'):
        all_taxa = extract_taxa(df)
        get_tree_emb(all_taxa,"../muscle")
        df = replace_with_tree_based_embedding(df)
    if(EMBEDDINGMETHOD == 'fixed'):
        df = replace_with_fixed_embedding(df, 0.1)
    n = len(df)
    print(n)
    index = df.index.values
    valid_n = int(n * 0.8)
    valid_n_end = int(n*0.9)
    #train df, 8 columbns, accessions, gos, labels, ngrams, proteins, sequences, orgs, embeddings
    train_df = df.loc[index[:valid_n]] # size 20179
    # net_embeddings = train_df['embeddings']#20179*256 network embeddings training set
    valid_df = df.loc[index[valid_n:valid_n_end]]#size 2522
    test_df = df.loc[index[valid_n_end:]]#size 2523
    file_archive = resdir+"/train_and_test_data"
    if not os.path.isdir(file_archive):
        os.mkdir(file_archive)
    train_df.to_csv(file_archive+"/train_df.csv")
    valid_df.to_csv(file_archive+"/valid_df.csv")
    test_df.to_csv(file_archive+"/test_df.csv")
    train_taxa = extract_taxa(train_df)
    # get the sequences from the dataframe
    sq = extract_taxa(df)
    tr = train_df
    net_embeddings = train_df['embeddings'].to_frame()
    print(net_embeddings)
    vl = valid_df
    ts = test_df
    if org is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == org]
        logging.info('Filtered test size: %d' % len(test_df))

    # Filter by type
    # org_df = pd.read_pickle('data/eukaryotes.pkl')
    # orgs = org_df['orgs']
    # test_df = test_df[test_df['orgs'].isin(orgs)]

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    def normalize_minmax(values):
        mn = np.min(values)
        mx = np.max(values)
        if mx - mn != 0.0:
            return (values - mn) / (mx - mn)
        return values - mn

    def get_values(data_frame):
        labels = reshape(data_frame['labels'].values)
        ngrams = sequence.pad_sequences(
            data_frame['ngrams'].values, maxlen=MAXLEN)
        ngrams = reshape(ngrams)
        rep = reshape(data_frame['embeddings'].values)
        data = (ngrams, rep)
        return data, labels

    train = get_values(train_df)
    valid = get_values(valid_df)
    test = get_values(test_df)
    # test = get_values(train_df[:10000])

    return train, valid, test, train_df, valid_df, test_df


def get_feature_model(params):
    embedding_dims = params['embedding_dims']
    max_features = 8001
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        dropout=params['embedding_dropout']))
    for i in range(params['nb_conv']):
        model.add(Convolution1D(
            nb_filter=params['nb_filter'],
            filter_length=params['filter_length'],
            border_mode='valid',
            activation='relu',
            subsample_length=1))
    model.add(MaxPooling1D(
        pool_length=params['pool_length'], stride=params['stride']))
    model.add(Flatten())
    summmary = model.summary()
    return model


def merge_outputs(outputs, name):
    if len(outputs) == 1:
        return outputs[0]
    return merge(outputs, mode='concat', name=name, concat_axis=1)


def merge_nets(nets, name):
    if len(nets) == 1:
        return nets[0]
    return merge(nets, mode='sum', name=name)


def get_node_name(go_id, unique=False):
    name = go_id.split(':')[1]
    if not unique:
        return name
    if name not in node_names:
        node_names.add(name)
        return name
    i = 1
    while (name + '_' + str(i)) in node_names:
        i += 1
    name = name + '_' + str(i)
    node_names.add(name)
    return name


def get_function_node(name, inputs):
    output_name = name + '_out'
    # net = Dense(256, name=name, activation='relu')(inputs)
    output = Dense(1, name=output_name, activation='sigmoid')(inputs)
    return output, output



def get_layers(inputs):
    q = deque()
    layers = {}
    name = get_node_name(GO_ID)
    layers[GO_ID] = {'net': inputs}
    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))
    while len(q) > 0:
        node_id, net = q.popleft()
        parent_nets = [inputs]
        # for p_id in get_parents(go, node_id):
        #     if p_id in func_set:
        #         parent_nets.append(layers[p_id]['net'])
        # if len(parent_nets) > 1:
        #     name = get_node_name(node_id) + '_parents'
        #     net = merge(
        #         parent_nets, mode='concat', concat_axis=1, name=name)
        name = get_node_name(node_id)
        net, output = get_function_node(name, inputs)
        if node_id not in layers:
            layers[node_id] = {'net': net, 'output': output}
            for n_id in go[node_id]['children']:
                if n_id in func_set and n_id not in layers:
                    ok = True
                    for p_id in get_parents(go, n_id):
                        if p_id in func_set and p_id not in layers:
                            ok = False
                    if ok:
                        q.append((n_id, net))

    for node_id in functions:
        childs = set(go[node_id]['children']).intersection(func_set)
        if len(childs) > 0:
            outputs = [layers[node_id]['output']]
            for ch_id in childs:
                outputs.append(layers[ch_id]['output'])
            name = get_node_name(node_id) + '_max'
            layers[node_id]['output'] = merge(
                outputs, mode='max', name=name)
    return layers


def get_model(params):
    logging.info("Building the model")
    inputs = Input(shape=(MAXLEN,), dtype='int32', name='input1')
    inputs2 = Input(shape=(REPLEN,), dtype='float32', name='input2')#network embedding
    feature_model = get_feature_model(params)(inputs)#embedding layer without network embedding
    net = merge(
        [feature_model, inputs2], mode='concat',
        concat_axis=1, name='merged')
    for i in range(params['nb_dense']):
        net = Dense(params['fc_output'], activation='relu')(net)
    layers = get_layers(net)
    output_models = []
    for i in range(len(functions)):
        output_models.append(layers[functions[i]]['output'])
    net = merge(output_models, mode='concat', concat_axis=1)#concatenation of inputs1(the mebdding of 3grams) and inputs2(the network embedding)
    # net = Dense(1024, activation='relu')(merged)
    # net = Dense(len(functions), activation='sigmoid')(net)
    model = Model(input=[inputs, inputs2], output=net)
    logging.info('Compiling the model')
    optimizer = RMSprop(lr=params['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy')
    logging.info(
        'Compilation finished')
    return model


def model(params, batch_size=128, nb_epoch=6, is_train=True):
    # set parameters:
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Loading Data")
    train, val, test, train_df, valid_df, test_df = load_data()
    train_df = pd.concat([train_df, valid_df])
    test_gos = test_df['gos'].values
    train_data, train_labels = train
    val_data, val_labels = val
    test_data, test_labels = test
    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Training data size: %d" % len(train_data[0]))
    logging.info("Validation data size: %d" % len(val_data[0]))
    logging.info("Test data size: %d" % len(test_data[0]))

    model_path = (DATA_ROOT + 'models/model_' + FUNCTION + '.h5') 
                  # '-' + str(params['embedding_dims']) +
                  # '-' + str(params['nb_filter']) +
                  # '-' + str(params['nb_conv']) +
                  # '-' + str(params['nb_dense']) + '.h5')
    checkpointer = ModelCheckpoint(
        filepath=model_path,
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    logging.info('Starting training the model')

    train_generator = DataGenerator(batch_size, nb_classes)
    train_generator.fit(train_data, train_labels)
    valid_generator = DataGenerator(batch_size, nb_classes)
    valid_generator.fit(val_data, val_labels)
    test_generator = DataGenerator(batch_size, nb_classes)
    test_generator.fit(test_data, test_labels)

    if is_train:
        model = get_model(params)
        model.fit_generator(
            train_generator,
            samples_per_epoch=len(train_data[0]),
            nb_epoch=nb_epoch,
            validation_data=valid_generator,
            nb_val_samples=len(val_data[0]),
            max_q_size=batch_size,
            callbacks=[checkpointer, earlystopper])
    logging.info('Loading best model')
    start_time = time.time()
    model_dir = resdir+"/model"
    if(not os.path.isdir(resdir+"/model")):
        os.mkdir(model_dir)
        copyfile(DATA_ROOT + 'models/model_' + FUNCTION + '.h5',model_dir+"/model.h5")

    #model = load_model(model_path)
    logging.info('Loading time: %d' % (time.time() - start_time))
    # orgs = ['9606', '10090', '10116', '7227', '7955',
    #         '559292', '3702', '284812', '6239',
    #         '83333', '83332', '224308', '208964']
    # for org in orgs:
    #     logging.info('Predicting for %s' % (org,))
    #     train, val, test, train_df, valid_df, test_df = load_data(org=org)
    #     test_data, test_labels = test
    #     test_gos = test_df['gos'].values
    #     test_generator = DataGenerator(batch_size, nb_classes)
    #     test_generator.fit(test_data, test_labels)
    start_time = time.time()
    preds = model.predict_generator(
        test_generator, val_samples=len(test_data[0]))
    running_time = time.time() - start_time
    logging.info('Running time: %d %d' % (running_time, len(test_data[0])))
    logging.info('Computing performance')
    f, p, r, t, preds_max = compute_performance(preds, test_labels, test_gos)
    roc_auc = compute_roc(preds, test_labels)
    mcc = compute_mcc(preds_max, test_labels)
    logging.info('Fmax measure: \t %f %f %f %f' % (f, p, r, t))
    logging.info('ROC AUC: \t %f ' % (roc_auc, ))
    logging.info('MCC: \t %f ' % (mcc, ))
    print(('%.3f & %.3f & %.3f & %.3f & %.3f' % (
        f, p, r, roc_auc, mcc)))
    eval_dir = resdir + '/evaluation'
    if (not os.path.isdir(eval_dir)):
        os.mkdir(eval_dir)
    res_f = open(eval_dir + "/result.txt", "w")
    res_f.write('Fmax measure: \t %f %f %f %f\n' % (f, p, r, t))
    res_f.write('ROC AUC: \t %f \n' % (roc_auc, ))
    res_f.write('MCC: \t %f \n' % (mcc, ))
    res_f.close()
    # return f
    # logging.info('Inconsistent predictions: %d' % incon)
    # logging.info('Saving the predictions')
    proteins = test_df['proteins']
    predictions = list()
    for i in range(preds_max.shape[0]):
        predictions.append(preds_max[i])
    df = pd.DataFrame(
        {
            'proteins': proteins, 'predictions': predictions,
            'gos': test_df['gos'], 'labels': test_df['labels']})
    df.to_pickle(DATA_ROOT + 'test-' + FUNCTION + '-preds.pkl')
    # logging.info('Done in %d sec' % (time.time() - start_time))

    # function_centric_performance(functions, preds.T, test_labels.T)


def load_prot_ipro():
    proteins = list()
    ipros = list()
    with open(DATA_ROOT + 'swissprot_ipro.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            if len(it) != 3:
                continue
            prot = it[1]
            iprs = set(it[2].split(';'))
            proteins.append(prot)
            ipros.append(iprs)
    return pd.DataFrame({'proteins': proteins, 'ipros': ipros})


def performanc_by_interpro():
    pred_df = pd.read_pickle(DATA_ROOT + 'test-' + FUNCTION + '-preds.pkl')
    ipro_df = load_prot_ipro()
    df = pred_df.merge(ipro_df, on='proteins', how='left')
    ipro = get_ipro()

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    for ipro_id in ipro:
        if len(ipro[ipro_id]['parents']) > 0:
            continue
        labels = list()
        predictions = list()
        gos = list()
        for i, row in df.iterrows():
            if not isinstance(row['ipros'], set):
                continue
            if ipro_id in row['ipros']:
                labels.append(row['labels'])
                predictions.append(row['predictions'])
                gos.append(row['gos'])
        pr = 0
        rc = 0
        total = 0
        p_total = 0
        for i in range(len(labels)):
            tp = np.sum(labels[i] * predictions[i])
            fp = np.sum(predictions[i]) - tp
            fn = np.sum(labels[i]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                pr += precision
                rc += recall
        if total > 0 and p_total > 0:
            rc /= total
            pr /= p_total
            if pr + rc > 0:
                f = 2 * pr * rc / (pr + rc)
                print(('%s\t%d\t%f\t%f\t%f' % (
                    ipro_id, len(labels), f, pr, rc)))


def function_centric_performance(functions, preds, labels):
    preds = np.round(preds, 2)
    for i in range(len(functions)):
        f_max = 0
        p_max = 0
        r_max = 0
        x = list()
        y = list()
        for t in range(1, 100):
            threshold = t / 100.0
            predictions = (preds[i, :] > threshold).astype(np.int32)
            tp = np.sum(predictions * labels[i, :])
            fp = np.sum(predictions) - tp
            fn = np.sum(labels[i, :]) - tp
            sn = tp / (1.0 * np.sum(labels[i, :]))
            sp = np.sum((predictions ^ 1) * (labels[i, :] ^ 1))
            sp /= 1.0 * np.sum(labels[i, :] ^ 1)
            fpr = 1 - sp
            x.append(fpr)
            y.append(sn)
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            f = 2 * precision * recall / (precision + recall)
            if f_max < f:
                f_max = f
                p_max = precision
                r_max = recall
        num_prots = np.sum(labels[i, :])
        roc_auc = auc(x, y)
        eval_dir = resdir +'evaluation'
        if(not os.path.isdir(eval_dir)):
            os.mkdir(eval_dir)
        f = open(eval_dir+"/result.txt","w")
        #write evaluation result
        f.write("functions[i], f_max, p_max, r_max, num_prots, roc_auc\n")
        f.write('%s %f %f %f %d %f' % (
            functions[i], f_max, p_max, r_max, num_prots, roc_auc))
        f.close()
        print(('%s %f %f %f %d %f' % (
            functions[i], f_max, p_max, r_max, num_prots, roc_auc)))


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

#
def compute_performance(preds, labels, gos):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max


def get_gos(pred):
    mdist = 1.0
    mgos = None
    for i in range(len(labels_gos)):
        labels, gos = labels_gos[i]
        dist = distance.cosine(pred, labels)
        if mdist > dist:
            mdist = dist
            mgos = gos
    return mgos


def compute_similarity_performance(train_df, test_df, preds):
    logging.info("Computing similarity performance")
    logging.info("Training data size %d" % len(train_df))
    train_labels = train_df['labels'].values
    train_gos = train_df['gos'].values
    global labels_gos
    labels_gos = list(zip(train_labels, train_gos))
    p = Pool(64)
    pred_gos = p.map(get_gos, preds)
    total = 0
    p = 0.0
    r = 0.0
    f = 0.0
    test_gos = test_df['gos'].values
    for gos, tgos in zip(pred_gos, test_gos):
        preds = set()
        test = set()
        for go_id in gos:
            if go_id in all_functions:
                preds |= get_anchestors(go, go_id)
        for go_id in tgos:
            if go_id in all_functions:
                test |= get_anchestors(go, go_id)
        tp = len(preds.intersection(test))
        fp = len(preds - test)
        fn = len(test - preds)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall
            f += 2 * precision * recall / (precision + recall)
    return f / total, p / total, r / total


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')


if __name__ == '__main__':
    main()
