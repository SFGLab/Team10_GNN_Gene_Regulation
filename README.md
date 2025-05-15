# Team10: GNN4GNR-3D: Gene regulation prediciton from scRNA-Seq expression and 3D genomic data.

## Aim

Use Chromatin conformation captrure assay data to enhance for GRN predictions from combined expression (scRNA-Seq) and chromosome conformation capture (3D, ChIA-PET) data.

## Contributions

Michał Denkiewicz, Paulina Kaczyńska, Sabrina Kwak, Gobikrishnan Subramaniam, Jacob Bumgarner, Asha Ajithakumari Sobhanakumar, Palash Sethi, Ammara Saleem

## Method

TODO: fix, so that instad of two sequential 3D and No3D, we have 3 parallel boxes: Base, Aug, Aug3D

<p align="center">
  <img src="./img/workflow.png" width="50%"/>
</p>

We prepared the dataset for the hESC cell line from the BEELINE dataset.
For the hESC cell line, The input GRN consits of 18234 nodes (Genes), out of which 14996 have corresponding scRNASeq data.

<p align="center">
  <img src="./img/dataComparison.jpg" width="50%"/>
</p>

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

We performed TODO: descripbe training
* data splits
* neg. set construction
* used parameters
* architecture overview

Performance is lower than the one reported in literature (however, we fixed a bug?)

Augumentation improves performance slightly

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

## Future directions

1. More comprehensive review of datasets
2. Improve negative set construction
3. Evaluate feature importance
4. Improve the usability and configurability

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
1. Quick intro: https://blogs.nvidia.com/blog/what-are-graph-neural-networks/

### Progress (legacy)

1. Get expression data from BEELINE: [DONE]
   * Evalutaing other cell lines
2. Get 3D conformation data from 4DN: [IN_PROGRESS]
3. Make preprocessing code: [IN_PROGRESS]
4. Map GRN nodes to the 3D data. [TODO]
5. Setup final shared training environment [IN_PROGRESS]
   * Problems installing pytorch_geometric on AWS
   * ...but we have other options
6. Adjust the notebook to use our datasets [IN_PROGRESS]
7. Technical run (No3D net) [DONE]
8. Results presentation and visualization [IN_PROGRESS]
   
