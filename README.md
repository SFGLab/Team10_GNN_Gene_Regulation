# Team10: GNNs for GRNs

## Workflow

#### Data preparation and preprocessing

```mermaid
flowchart TB
  %% Phase 1: Data Preparation & Preprocessing
  subgraph Phase_1["1. Data Preparation & Preprocessing"]
    A1[Gather input data]
    A2[Determine exact format of network inputs/outputs]
    A3[Transform data to required format]
    A4[Obtain ChIA-PET from 4DNucleome]
    A5[Upload data to cloud environment]
    A6[Write preprocessing code for GNN model]
    A7[Preprocess expression data & outputs]
    A8[Map Genes/TFs to 3D network locations]
    A9[Optional Add enhancer data from EnhancerDb]
    A10[Build network; compute features (Paul et al.)]
    A11[Check alternative features (Wang et al.)]
    A12[Assess 3D-feature methods]
    A13[Make holdout set]

    A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7 --> A8 --> A9
    A9 --> A10 --> A11 --> A12 --> A13
  end

  %% Phase 2: GNN Model Preparation & Technical Runs
  subgraph Phase_2["2. GNN Model Prep & Technical Runs"]
    B1[Upload reference notebook; verify imports]
    B2[Review Wang et al. implementation]
    B3[Technical run (small dataset); record time]
    B4[Technical run (subset of actual data)]
    B5[Train on single cell line (hESC)]
    B6[Evaluate results & timing]
    B7[Decide on extra runs; limited parameter tuning]

    A13 --> B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7
  end

  %% Phase 3: Additional Training Scenarios
  subgraph Phase_3["3. Additional Training Runs"]
    C1[Disable 3D features; compare performance]
    C2[Train on another human cell line; cross-test]
    C3[Add interaction type (inhibitory/activatory)]

    B7 --> C1 --> C2 --> C3
  end

  %% Phase 4 & 5: Summarize & Conclude
  C3 --> D1[Summarise and visualize results]
  D1 --> E1[Conclude]
```

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
