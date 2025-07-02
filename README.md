# Paper: Attributed Graph Clustering with Multi-Scale Weight-Based Pairwise Coarsening and Contrastive Learning

![QQ_1751446870427](https://github.com/user-attachments/assets/ad6a747d-6902-4d99-9d6a-0c302d8c072e)

paper address:https://doi.org/10.1016/j.neucom.2025.130796

***This study introduces the Multi-Scale Weight-Based Pairwise Coarsening and Contrastive Learning (MPCCL) model, a novel approach for attributed graph clustering that effectively bridges critical gaps in existing methods, including long-range dependency, feature collapse, and information loss. Traditional methods often struggle to capture high-order graph features due to their reliance on low-order attribute information, while contrastive learning techniques face limitations in feature diversity by overemphasizing local neighborhood structures. Similarly, conventional graph coarsening methods, though reducing graph scale, frequently lose fine-grained structural details. MPCCL addresses these challenges through an innovative multi-scale coarsening strategy, which progressively condenses the graph while prioritizing the merging of key edges based on global node similarity to preserve essential structural information. It further introduces a one-to-many contrastive learning paradigm, integrating node embeddings with augmented graph views and cluster centroids to enhance feature diversity, while mitigating feature masking issues caused by the accumulation of high-frequency node weights during multi-scale coarsening. By incorporating a graph reconstruction loss and KL divergence into its self-supervised learning framework, MPCCL ensures cross-scale consistency of node representations. Experimental evaluations reveal that MPCCL achieves a significant improvement in clustering performance, including a remarkable 15.24% increase in NMI on the ACM dataset and notable robust gains on smaller-scale datasets such as Citeseer, Cora and DBLP. In the large-scale Reuters dataset, it significantly improved by 17.84%, further validating its advantage in enhancing clustering performance and robustness. These results highlight MPCCL’s potential for application in diverse graph clustering tasks, ranging from social network analysis to bioinformatics and knowledge graph-based data mining.***

**Authors:Binxiong Li, Yuefei Wang,*, Binyu Zhao, Heyang Gao, Benhan Yang, Quanzhou Luo, Xue Li, Xu Xiang, Yujie Liu, Huijie Tang**

### Architecture Overview

![MPCCL](https://github.com/YF-W/MPCCL/blob/1446e417487806a2357471439b806349a0a23425/MPCCL.png)

***We propose a novel attribute graph clustering method based on weight-paired multi-scale graph coarsening and Laplacian regularization contrastive learning, termed MPCCL. Initially, we enhance the data through feature dropout, forcing the model not to rely on specific features during training and providing a second view for contrastive learning. We devise a multi-scale graph coarsening approach to capture structural information from the coarsened graph at different levels. During the coarsening process, we prioritize merging node pairs with higher similarity across the entire graph, rather than solely relying on the similarity of local neighbors. The goal is to ensure that node pairs with global structural significance are preserved, avoiding the loss of global information caused by excessive reliance on local structures. More importantly, when merging, we only retain the structural information between nodes, while the original node features are preserved, preventing the loss of feature information due to excessive coarsening. As shown in Figure 1, on the ACM dataset, the model using multi-scale graph coarsening performs significantly better than the one without it, indicating that capturing multi-level graph structure information through coarsening enhances clustering performance. In contrastive learning, we employ the KMeans clustering method, clustering the node representations from the second view and obtaining cluster centers for each view. By this approach, low-frequency samples within the same cluster are influenced by the contrastive learning of positive pairs, making the features of low-frequency samples in the same cluster more similar to those of high-frequency samples. This enables the preservation of important learning signals during the coarsening process, achieving a complementary mechanism between contrastive learning and multi-scale graph coarsening. Figure 1 compares the models with and without contrastive learning, demonstrating the effectiveness of our contrastive learning design on the ACM dataset. To further enhance model stability and representational capability, we add a Laplacian regularization term in the loss function to ensure smoother representations among neighboring nodes. Additionally, we design a GCN-based encoder to generate a reconstructed adjacency matrix, constructing a reconstruction loss term in the total loss to preserve the structural information of the graph as much as possible.***

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
