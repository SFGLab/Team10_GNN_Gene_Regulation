from torch_geometric.data import Data
from torch_geometric.utils import convert
import torch
import pandas as pd
import os
import math
import numpy as np
import time

import networkx as nx
import random
from collections import Counter
from torch_geometric.utils import to_networkx
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#Edge creation fucnction for the graph - reformatting the gene expression input
def convert_to_networkx(graph, n_sample=None):
    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()
    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]
    return g, y


def _limited_betweenness_centrality(G, k=50):
    # Calculate betweenness centrality using k samples
    betweenness = nx.betweenness_centrality(G, k=k)
    return betweenness


def _log10_degree(G):
    return {u: math.log10(d + 1.0) for u, d in G.degree()}


AUG_GRN = {
    "degree_log10": _log10_degree,
    "degree_centrality": nx.algorithms.centrality.degree_centrality,
    "clustering": nx.algorithms.cluster.clustering,
    "betweenness_centrality":_limited_betweenness_centrality,
    "pagerank": nx.algorithms.link_analysis.pagerank_alg.pagerank
}

AUG_G3D = {
    "degree_centrality": nx.algorithms.centrality.degree_centrality
}

MULTIPROCESSING_GLOBAL_DATA = {}  # READ ONLY!!!

def _calculate_graph_feature(name, fun, data_key):
    logging.info(f"Calculating {name} from {data_key}")
    t0 = time.time()
    G = MULTIPROCESSING_GLOBAL_DATA[data_key]    
    return name, data_key, fun(G), t0


def load_graph_from_3d_data(path_3d):
    return None


#Data preprocessing to graph
def data_preprocessing(
        folder_name, file_expression, file_ordering, path_gold, path_3d=None,
        aug_grn=AUG_GRN, aug_g3d=AUG_G3D,
        executor_factory=ProcessPoolExecutor, max_workers=None
    ):
    hetero = pd.read_csv(os.path.join(folder_name, file_expression), index_col=0)
    null = pd.read_csv(os.path.join(folder_name, file_ordering), index_col=0)
    #traject = pd.read_csv(default_path+file_traject,sep='\t')
    gold = pd.read_csv(path_gold, index_col=False)
    if path_3d is not None:
        G3D = load_graph_from_3d_data(path_3d)    

    logging.info(f"Prepraing basic data")
    node_features = hetero.join(null)
    gene_to_index = {row['index']: i for i, row in node_features.reset_index().iterrows()}
    logging.info(f"Node features shape: {node_features.shape}")
    # print(gene_to_index)
    gold['Gene1'] = gold['Gene1'].apply(lambda x: gene_to_index[x] if x in gene_to_index.keys() else None)
    gold['Gene2'] = gold['Gene2'].apply(lambda x: gene_to_index[x] if x in gene_to_index.keys() else None)
    gold = gold.dropna(axis=0)

    basic_data = Data()        
    basic_data.x = torch.tensor(node_features.values)
    #basic_data.edge_attr=edscore
    basic_data.edge_index = torch.tensor(gold.values).T #edge_lab_index    
    #basic_TS_data.x =torch.tensor(listfinal).T
    #basic_TS_data.edge_attr=edscore
    #basic_TS_data.edge_index = gold.values #edge_lab_index

    G = convert.to_networkx(basic_data, to_undirected=True)    
    MULTIPROCESSING_GLOBAL_DATA["GRN"] = G  # Multiprocessing trick (UNIX only)
    # Are the labels messed up?    
    
    G3D = G  # TODO: REMOVE THIS
    MULTIPROCESSING_GLOBAL_DATA["G3D"] = G3D

    aug_results = {
        'G3D': {}, 
        'GRN': {}
    }
    logger = logging.getLogger()
    with executor_factory(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_calculate_graph_feature, name, fun, _data_key)            
            for _data_key, _funs in zip(["GRN", "G3D"], [aug_grn, aug_g3d])
            for name, fun in _funs.items()
        ]
        for future in as_completed(futures):
            name, data_key, result, t0 = future.result()
            aug_results[data_key][name] = result
            logger.info(f"Done calculating {name} from {data_key} in {time.time() - t0:.2f} seconds")
            # TODO: why does it not flush?
    aug_featues_grn = [v[1] for v in sorted(aug_results["GRN"].items(), key=lambda x: x[0])]  # keep same order always
    aug_featues_g3d = [v[1] for v in sorted(aug_results["G3D"].items(), key=lambda x: x[0])]

    logging.info(f"Augmenting data")
    basic_aug_data = Data()    
    basic_aug_data.x = torch.cat(
        [
            torch.tensor(node_features.values),
            torch.tensor([list(x.values()) for x in aug_featues_grn]).T
        ], axis=1)
    basic_aug_data.edge_index = torch.tensor(gold.values).T

    logging.info(f"Augmenting data + 3D")
    basic_aug3d_data = Data()
    basic_aug3d_data.x = torch.cat(
        [
            torch.tensor(node_features.values),
            torch.tensor([list(x.values()) for x in aug_featues_grn + aug_featues_g3d]).T
        ], axis=1)
    basic_aug_data.edge_index = torch.tensor(gold.values).T
    return basic_data, basic_aug_data, basic_aug3d_data


def save_torch(basic_data, basic_aug_data, basic_aug3d_data, preprocessed_dir):
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir, exist_ok=True)
    torch.save(basic_data, os.path.join(preprocessed_dir, "basic_data.pt"))
    torch.save(basic_aug_data, os.path.join(preprocessed_dir, "basic_data_aug.pt"))
    torch.save(basic_aug3d_data, os.path.join(preprocessed_dir, "basic_data_aug3d.pt"))


def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def main():    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess the BEELINE data.")
    parser.add_argument(
        "--config", type=str, default="config_hESC.json", help="Path to the configuration file"
    )
    args = parser.parse_args()
    logging.info(f"In: {os.getcwd()}")

    # Load paths from the specified configuration file
    config = load_config(args.config)    
    logging.info(f"Loaded configuration from {args.config}")
    logging.info(f"Configuration: {config}")

    data_path = config["data_path"]
    gold_std = config["gold_std"]
    preprocessed_dir = config["preprocessed_dir"]    
    data3d_path = config["data3d_path"]
    
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    assert os.path.exists(gold_std), f"Gold standard path {gold_std} does not exist"    
    assert os.path.exists(os.path.join(data_path, "ExpressionData.csv")), f"Expression data file does not exist in {data_path}"
    assert os.path.exists(os.path.join(data_path, "GeneOrdering.csv")), f"Gene ordering file does not exist in {data_path}"
    # assert os.path.exists(data3d_path), f"3D data path {data3d_path} does not exist"

    basic_data, basic_aug_data, basic_aug3d_data = data_preprocessing(data_path, "ExpressionData.csv", "GeneOrdering.csv", gold_std, path_3d=data3d_path)
    logging.info(f"Basic data shape: {basic_data.x.shape}")
    logging.info(f"Augmented data shape: {basic_aug_data.x.shape}")
    logging.info(f"Augmented 3D data shape: {basic_aug3d_data.x.shape}")
    save_torch(basic_data, basic_aug_data, basic_aug3d_data, preprocessed_dir)
    logging.info(f"Preprocessed data saved to {preprocessed_dir}")


if __name__== "__main__":
    main()
