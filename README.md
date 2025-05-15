# Team10: GNN4GRN-3D: Gene regulation prediciton from scRNA-Seq expression and 3D genomic data.

## Aim

We use Chromatin conformation captrure assay data to enhance for GRN predictions from combined expression (scRNA-Seq) and chromosome conformation capture reflecting 3D genome structure (Hi-C, ChIA-PET) data.

We provide an extensible framework for training GNN models on custom datasets, adding the ability to adjust and provide custom features by the user without the need to rewrite training code. We build on previous approaches (Paul et al. 2024) in particular by incorporating chromosome conformation capture data (Hi-C), to improve the GRN predictions from scRNA-Seq to bridge graph-based machine learning and 3D Genomics.

## Contributions

Michał Denkiewicz, Paulina Kaczyńska, Sabrina Kwak, Gobikrishnan Subramaniam, Jacob Bumgarner, Asha Ajithakumari Sobhanakumar, Palash Sethi, Ammara Saleem

## Method

TODO: fix, so that instad of two sequential 3D and No3D, we have 3 parallel boxes: Base, Aug, Aug3D

<p align="center">
  <img src="./img/workflow.png" width="50%"/>
</p>

* For assessment of GNN4GNR-3D we focus on hESC data as an example
* We prepared the dataset for the hESC cell line from the BEELINE dataset.
For the hESC cell line, The input GRN consits of 18234 nodes (Genes), out of which 14996 have corresponding scRNASeq data.

<p align="center">
  <img src="./img/dataComparison.jpg" width="50%"/>
</p>

* We use BEELINE data (Pratapa et al., 2020): the interactions forming GRNs are the predicted data, while the scRNA-Seq data is used as input
We combine this with the Chromosome Conformation Capture (3D) data for the same cell line.
* Data preprocessing: scRNA-Seq matrix is imputed to an autoencoder to produce embeddings, serving as primary features for the network nodes
* The 3D genomics data is employed as follows: each gene is mapped to a loci using the appropriate assembly, and a signal value is extracted for each gene-gene pair from the Hi-C matric (maximum value of O/E). Pairs that exceed a specified threshold form edges of the new network.
* The GNN architecture is based on (Paul et al. 2024).
* Additional node features are added: augmented features computed from the GRN structure (as in base work: betweenness centrality, PageRank score, node degree, clustering coefficient). Importantly, node features are added that contain data relating to 3D structure: each gene-node is mapped to its genomic location, and assigned to a node in the 3D network. Then, betweenness centrality, page rank, degree, clustering coefficient of the gene-node in the 3D network.
* 3 types of models are produced for each dataset: Base (pure expression data), Aug (with network science features calculated from the GRN network), ad finally Aug3D (with features extracted both from GRN and 3D network)

## Installation & Usage

### Requirements

* `PyTorch Geometric` capable system:
  * OS: Linux, pref. Ubuntu >= 22.04
  * CPU: pref. 12+ cores for data preprocessing
  * GPU: Nvidia CUDA-capable (pref. 12.6). CPU computation is also possible
  * RAM: >= 16 GB
* Docker image with the above stack is availible

### Input data format

TODO

### Direct installation

1. Setup a python virtual environment (pyenv, conda, mamba etc.)
2. Install dependencies from requirements.txt
3. Prepare the data directory and a config `.json` file (see examples) TODO: elaborate
4. Preprocess input data using `preprocessing.py`
5. Train a network using `train.py` TODO: elaborate
6. TODO: a way to use a ready network for prediction
  
### Using docker

TODO

## Results

We performed TODO: describe training
* data splits
* neg. set construction
* used parameters
* architecture overview

Performance on our dataset is lower than the one reported in (Paul et al. 2024), however we identifed a possible issue. Augumentation improves performance slightly.

3D augumentaion results in TODO

Resulting scores:
|    | dec     | af_val   |   num_layers |   epoch | aggr   | var            |      auc |     aupr |       mcc |   jac_score |   cohkap_score |       f1 |   top_k | aug    |
|---:|:--------|:---------|-------------:|--------:|:-------|:---------------|---------:|---------:|----------:|------------:|---------------:|---------:|--------:|:-------|
|  0 | dot_sum | F.silu   |            3 |     200 | sum    | HypergraphConv | 0.992565 | 0.981231 | 0.0889979 |     0.25594 |      0.0157168 | 0.407568 | 0.27321 | no_aug |
|  0 | dot_sum | F.silu   |            3 |     200 | sum    | HypergraphConv | 0.732785 | 0.688903 | 0         |     0.25    |      0         | 0.4      | 0.25    | aug    |
|  0 | dot_sum | F.silu   |            3 |     200 | sum    | HypergraphConv | 0.732784 | 0.688903 | 0         |     0.25    |      0         | 0.4      | 0.25    | aug3d  |

Training trajectories:
<p align="center">
  <img src="./img/train.png" width="80%"/>
</p>

## Conclusions

We provide an extensible framework for training GNN models for GRN edge prediction on custom datasets, adding the ability to adjust and provide custom features by the user without the need to rewrite training code.

## Future directions

1. Add more architecture configuration options, expand the number of networks used
2. More comprehensive review of datasets
3. Add enhancer data to the network
4. Provide an automated 3D network generation method

Technical:
1. Improve negative set construction
2. Evaluate feature importance
3. Improve the usability and configurability

## References

TODO: Write it as proper markdown.

GNN review https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00876-
BEELINE: https://pmc.ncbi.nlm.nih.gov/articles/PMC7098173/ and https://zenodo.org/records/3701939

GNN-GRN approaches:
1. https://www.sciencedirect.com/science/article/pii/S1568494624006732#da1 - we treat this as a baseline, they also provide code using PyTorch Geometric
2. https://www.sciencedirect.com/science/article/pii/S200103702030444X
3. https://genome.cshlp.org/content/32/5/930.full

## Misc. resources
1. Tunning GNNs (in case we need it): https://machinelearningmastery.com/guide-to-iteratively-tuning-gnns/
2. More on GRNs: https://www.sciencedirect.com/science/article/pii/S2452310021000184
3. https://bio.liclab.net/scGRN/index.php - extra data?
4. DREAM5: (v.old) https://www.synapse.org/Synapse:syn2820440/wiki/71026 - extra data?
5. Quick intro: https://blogs.nvidia.com/blog/what-are-graph-neural-networks/

   
