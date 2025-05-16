#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install hic-straw


# In[ ]:


import networkx as nx
import numpy as np
import hicstraw
import pandas as pd
import pickle
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


# Different approaches that could be considered to improve the 3D graph:
# - find a way to automatically handle outdated gene symbols corresponding to multiple modern gene symbols, so that they could be included in this dataset
# - increase flanking region to get more relevant contacts between genes
# - consider a different way to define contact strength between two genes: currently this method tries to avoid bin length bias in genes - it only picks one contact bin with highest count to define contact signal between genes
# - ignore contact frequency bin below a certain threshold and treat it as zero instead
# - consider a different way to define genomic coordinates for genes - currently the coordinates take into account one transcript, not its alternative spliced versions
# - speed up the mapping of genes to contacts

# In[ ]:


# parameters that I could extend to go into a config file
res = 5000 # in bp, resolution of the contact map
flank_length = 4 #to include neighbouring regions around the gene to collect more contact frequency information (in bins)
threshold = 0.0# values under this threshold are considered to be 0 in a contact map


# In[ ]:


# load the csv with gene names corresponding to genomic coordinates
gene_locations = pd.read_csv("./symbol_mapping/00_BEELINE_hESC_run/slim_hESC_symbols_mapped.csv")
# remove gene names that correspond to more than one gene in symbol mapping
unique_map_gene_locations = gene_locations.loc[gene_locations["Symbol.Alt"].isnull(), :].copy()

# convert to 0-based indexing
unique_map_gene_locations["Start.BP"] = unique_map_gene_locations["Start.BP"] - 1
unique_map_gene_locations["End.BP"] = unique_map_gene_locations["End.BP"] - 1
# assign a bin ID to each BP start and end 
unique_map_gene_locations["Start.BP.res5kbp"] = unique_map_gene_locations["Start.BP"].values // res
unique_map_gene_locations["End.BP.res5kbp"] = unique_map_gene_locations["End.BP"].values // res


# In[ ]:


# unique_map_gene_locations = unique_map_gene_locations.head(10)
# unique_map_gene_locations


# In[ ]:


# download relevant Hi-C data from 4DN Data Portal
# downloaded from https://data.4dnucleome.org/files-processed/4DNFI2WSZPG9/#file-overview

# You can use the command below:
# !wget -P ./data/ https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/9e26fac7-1e30-4abb-8946-8a23b3487365/4DNFI2WSZPG9.hic


# In[ ]:


hic = hicstraw.HiCFile("./data/4DNFI2WSZPG9.hic")
# highest res possible in this dataset: 1kbp, but we use 5kbp now

chrom_sizes = {ch.name: ch.length for ch in hic.getChromosomes()}
# print(chrom_sizes)

# hic.getResolutions() # check available resolutions


# In[ ]:


n_genes = len(unique_map_gene_locations)

graph_data = []

def _compute_gene(i):
    t0 = time.time()
    results = np.zeros(n_genes, dtype=float)

    for j in range(n_genes):        
        gene1 = unique_map_gene_locations.iloc[i]["Symbol"]
        chrname1 = unique_map_gene_locations.iloc[i]["Chromosome"].removeprefix("chr")
        start1 = unique_map_gene_locations.iloc[i]["Start.BP.res5kbp"]
        end1 = unique_map_gene_locations.iloc[i]["End.BP.res5kbp"]

        gene2 = unique_map_gene_locations.iloc[j]["Symbol"]
        chrname2 = unique_map_gene_locations.iloc[j]["Chromosome"].removeprefix("chr")
        start2 = unique_map_gene_locations.iloc[j]["Start.BP.res5kbp"]
        end2 = unique_map_gene_locations.iloc[j]["End.BP.res5kbp"]

        if i == j:
            results[j] = 0.0
            continue

        start1 = (start1 - flank_length) * res
        start2 = (start2 - flank_length) * res
        end1 = (end1 + 1 + flank_length) * res - 1
        end2 = (end2 + 1 + flank_length) * res - 1

        # clip to data size
        start1 = max(0, min(chrom_sizes[chrname1] - 1, start1))
        start2 = max(0, min(chrom_sizes[chrname2] - 1, start2))
        end1 = max(0, min(chrom_sizes[chrname1] - 1, end1))
        end2 = max(0, min(chrom_sizes[chrname2] - 1, end2))

        map = hic.getMatrixZoomData(chrname1, chrname2, "oe", "VC_SQRT", "BP", res) 
        contacts = map.getRecordsAsMatrix(start1, end1, start2, end2)
        # print(contacts.shape)
        # contacts = np.where(contacts < threshold, 0, contacts)
        edge_value = np.max(contacts)
        results[j] = edge_value

    return i, results, t0

# print(_compute_gene(3))


# In[ ]:


graph_data = np.zeros((n_genes, n_genes), dtype=float)

max_workers = os.cpu_count() - 1 
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    print(f"Computing for {n_genes} genes with {max_workers} workers")
    futures = [executor.submit(_compute_gene, i) for i in range(n_genes)]
    for future in as_completed(futures):
        i, result, t0 = future.result()
        gene1 = unique_map_gene_locations.iloc[i]["Symbol"]
        print(f"Computed for gene {i}({gene1}) in {time.time() - t0:.2f}s")
        graph_data[i, :] = result

np.savetxt("./data/3Ddata_graph_hESC_endoderm.csv", graph_data, delimiter=",", fmt="%s")

