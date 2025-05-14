from torch_geometric.data import Data
from torch_geometric.utils import convert
import torch
import pandas as pd
import os
import itertools
import numpy as np

import networkx as nx
import random
from collections import Counter
from torch_geometric.utils import to_networkx

#Edge creation fucnction for the graph - reformatting the gene expression input
def convert_to_networkx(graph, n_sample=None):
    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()
    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]
    return g, y


#Data preprocessing to graph
def data_preprocessing(folder_name,file_expression,file_ordering,path_gold):

    hetero = pd.read_csv(os.path.join(folder_name, file_expression), index_col=0)
    null = pd.read_csv(os.path.join(folder_name, file_ordering), index_col=0)
    #traject = pd.read_csv(default_path+file_traject,sep='\t')
    gold = pd.read_csv(path_gold, index_col=False)

    node_features = hetero.join(null)
    gene_to_index = {row['index']: i for i, row in node_features.reset_index().iterrows()}
    print("node_features")
    print(node_features.shape)
    print(gene_to_index)
    gold['Gene1'] = gold['Gene1'].apply(lambda x: gene_to_index[x] if x in gene_to_index.keys() else None)
    gold['Gene2'] = gold['Gene2'].apply(lambda x: gene_to_index[x] if x in gene_to_index.keys() else None)
    basic_data = Data()
    basic_aug_data = Data()
    gold = gold.dropna(axis=0)
    basic_data.x = torch.tensor(node_features.values)
    #basic_data.edge_attr=edscore
    basic_data.edge_index = torch.tensor(gold.values).T #edge_lab_index
    torch. save(basic_data, path+"basic_data_hESC.pt")
    #basic_TS_data.x =torch.tensor(listfinal).T
    #basic_TS_data.edge_attr=edscore
    #basic_TS_data.edge_index = gold.values #edge_lab_index

    G = convert.to_networkx(basic_data, to_undirected=True)
    # Are the labels messed up?
    pagerank = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
    clustering_coef = nx.algorithms.cluster.clustering(G)
    betweeness_centrality = nx.betweenness_centrality(G, k=50)
    degree = G.degree()

    pg_list =list(pagerank.values())
    clu_coef_list =list(clustering_coef.values())
    bet_cen_list =list(betweeness_centrality.values())

    deg_list =[x[1] for x in degree]

    basic_aug_data.x = torch.cat([torch.tensor(node_features.values), torch.tensor([pg_list, clu_coef_list,bet_cen_list, deg_list]).T], axis=1)
    basic_aug_data.edge_index = torch.tensor(gold.values).T
    torch. save(basic_aug_data, path+"basic_data_aug_hESC.pt")
    return basic_data,basic_aug_data

if __name__=="main":
    #Path for the files
    path = "./BEELINE/"
    gold_std = "./BEELINE/hESC-ChIP-seq-network.csv"
    basic_data,basic_aug_data = data_preprocessing(path, "ExpressionData.csv", "GeneOrdering.csv", gold_std)
