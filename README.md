# Team10: GNNs for GRNs

## Workflow

#### Data preparation and preprocessing

1. Get expression data from BEELINE
2. Make preprocessing code that transforms the data to what the network input requires
3. Get 3D conformation data (ChIA-PET from 4DN)
4. Map Genes, TFs to their location in 3D network
5. Create 3D netowrks, integrate with BEELINE data.
6. Upload data to AWS
7. Make holdout set
   
#### Create GNN model

1. Upload the reference notebook to AWS, check all libraries, etc.
2. Check out Wang et al. as a possible alternative reference implementation (may work better with BEELINE)
3. Adjust the notebook to use our datasets
4. Do a technical run (small data set)
5. Train on a single cell line (hESC)
6. Check results and time, decide viability of further steps.

#### Model adjustments

1. Incorporate enhancer data from EnhancerDb
2. Prepare the No3D network, compare thre results.
3. Compare between cell lines.

#### Conclusion

1. Summary metric (F1, AUC)
2. Visualization of the netowrks
3. Conclude

## Main papers/resources:

### Basic GNNs:
1. Quick intro: https://blogs.nvidia.com/blog/what-are-graph-neural-networks/
2. A larger review, but I'm not sure if it is best. Nice picture for PowerPoint though:): https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00876-4

### Ground truth data:
1. BEELINE: https://pmc.ncbi.nlm.nih.gov/articles/PMC7098173/ and https://zenodo.org/records/3701939
2. https://bio.liclab.net/scGRN/index.php - not sure if it has more data than pt.1
3. DREAM5: (v.old) https://www.synapse.org/Synapse:syn2820440/wiki/71026

### GNN-GRN approaches:
1. https://www.sciencedirect.com/science/article/pii/S1568494624006732#da1 - we can treat this as a baseline, they also provide code using PyTorch Geometric
2. https://www.sciencedirect.com/science/article/pii/S200103702030444X
3. https://genome.cshlp.org/content/32/5/930.full

### Misc:
1. Tunning GNNs (in case we need it): https://machinelearningmastery.com/guide-to-iteratively-tuning-gnns/
2. More on GRNs: https://www.sciencedirect.com/science/article/pii/S2452310021000184
