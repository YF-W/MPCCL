# Paper: Attributed Graph Clustering with Multi-Scale Weight-Based Pairwise Coarsening and Contrastive Learning

paper address:https://doi.org/10.1016/j.neucom.2025.130796

***This study introduces the Multi-Scale Weight-Based Pairwise Coarsening and Contrastive Learning (MPCCL) model, a novel approach for attributed graph clustering that effectively bridges critical gaps in existing methods, including long-range dependency, feature collapse, and information loss. Traditional methods often struggle to capture high-order graph features due to their reliance on low-order attribute information, while contrastive learning techniques face limitations in feature diversity by overemphasizing local neighborhood structures. Similarly, conventional graph coarsening methods, though reducing graph scale, frequently lose fine-grained structural details. MPCCL addresses these challenges through an innovative multi-scale coarsening strategy, which progressively condenses the graph while prioritizing the merging of key edges based on global node similarity to preserve essential structural information. It further introduces a one-to-many contrastive learning paradigm, integrating node embeddings with augmented graph views and cluster centroids to enhance feature diversity, while mitigating feature masking issues caused by the accumulation of high-frequency node weights during multi-scale coarsening. By incorporating a graph reconstruction loss and KL divergence into its self-supervised learning framework, MPCCL ensures cross-scale consistency of node representations. Experimental evaluations reveal that MPCCL achieves a significant improvement in clustering performance, including a remarkable 15.24% increase in NMI on the ACM dataset and notable robust gains on smaller-scale datasets such as Citeseer, Cora and DBLP. In the large-scale Reuters dataset, it significantly improved by 17.84%, further validating its advantage in enhancing clustering performance and robustness. These results highlight MPCCL’s potential for application in diverse graph clustering tasks, ranging from social network analysis to bioinformatics and knowledge graph-based data mining.***

**Authors:Binxiong Li, Yuefei Wang,*, Binyu Zhao, Heyang Gao, Benhan Yang, Quanzhou Luo, Xue Li, Xu Xiang, Yujie Liu, Huijie Tang**

### 1. Architecture Overview


## Environment Requirement

python==3.9.0

pytorch==2.1.1

numpy==1.26.4

sklearn==1.4.1

munkres==1.1.4

## Run Code

For example: 

```python
python train_conclu.py
```

## Dataset:

Google drive：
[Click Here to Access](https://drive.google.com/drive/folders/1FfXnNGiOTuFUhNoEEkWvQpdZ1sIzBWp6?usp=sharing)
