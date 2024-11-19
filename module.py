import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.cluster import KMeans
from torch.nn import Module

def compute_edge_weights(G, node_features):
    for u, v in G.edges():
        # 获取两个节点的特征向量
        feature_u = node_features[u]
        feature_v = node_features[v]

        # 计算余弦相似度
        weight = F.cosine_similarity(feature_u.unsqueeze(0), feature_v.unsqueeze(0)).item()

        # 将余弦相似度作为边的权重
        G[u][v]['weight'] = weight

    return G


def multi_scale_graph_coarsening(G, scales=[0.3, 0.15, 0.06]):
    coarse_graphs = []
    for scale in scales:
        target_size = max(10, int(G.number_of_nodes() * scale))
        G_coarse = fast_graph_coarsening(G, target_size=target_size, min_edges=5)  # 设置最小边数

        print(
            f"Generated coarse graph with scale {scale}: {G_coarse.number_of_nodes()} nodes, {G_coarse.number_of_edges()} edges")

        if G_coarse.number_of_edges() > 0:
            coarse_graphs.append(G_coarse)
        else:
            print(f"Skipping this scale due to insufficient edges.")

    return coarse_graphs

def fast_graph_coarsening(G, target_size=10, min_edges=1):
    # 创建原始图的副本
    G_coarse = G.copy()

    # 当粗化后的图的节点数仍然大于target_size时，继续粗化
    while G_coarse.number_of_nodes() > target_size:
        """匹配节点"""
        # 存储尚未匹配的节点
        unmatched_nodes = set(G_coarse.nodes())
        # 存储每一轮中匹配的节点对
        matchings = []
        # 根据边的权重降序排列图中的边，优先合并较高权重的边
        edges = sorted(G_coarse.edges(data=True), key=lambda x: x[2].get('weight', 1), reverse=True)

        # 遍历图中的边，如果边的两个端点（u和v）都还没有匹配，则将它们配对，并将它们从unmatched_nodes中移除
        for u, v, data in edges:
            # print("data:",data)
            if u in unmatched_nodes and v in unmatched_nodes:
                matchings.append((u, v))
                unmatched_nodes.remove(u)
                unmatched_nodes.remove(v)

        """合并节点"""
        for u, v in matchings:
            # 创建一个新节点new_node，代表u和v的合并节点
            new_node = f"{u}-{v}"
            # 新节点的权重是u和v的权重之和
            G_coarse.add_node(new_node, weight=G_coarse.nodes[u].get('weight', 1) + G_coarse.nodes[v].get('weight', 1))

            # 合并节点u和v的邻居，如果邻居不是u或v，则计算新节点与邻居之间的新权重
            for neighbor in set(G_coarse.neighbors(u)).union(set(G_coarse.neighbors(v))):
                if neighbor in {u, v} or neighbor not in G_coarse:
                    continue
                new_weight = (G_coarse[u][neighbor].get('weight', 1) if neighbor in G_coarse[u] else 0) + \
                             (G_coarse[v][neighbor].get('weight', 1) if neighbor in G_coarse[v] else 0)
                # 如果新节点和邻居之间已经有一条边，则更新这条边的权重；如果没有，则添加一条新的边
                if G_coarse.has_edge(new_node, neighbor):
                    G_coarse[new_node][neighbor]['weight'] += new_weight
                else:
                    G_coarse.add_edge(new_node, neighbor, weight=new_weight)

            # 删除原始节点
            G_coarse.remove_node(u)
            G_coarse.remove_node(v)

        """检查边数"""
        if G_coarse.number_of_edges() < min_edges:
            print(f"Warning: Coarsened graph has fewer than {min_edges} edges. Stopping further coarsening.")
            break

    # 确保所有节点类型一致
    """转换节点标签"""
    # 将粗化后的图的节点标签转换为整数，以确保节点标签一致
    G_coarse = nx.convert_node_labels_to_integers(G_coarse)

    # print("G_coarse:", G_coarse)
    return G_coarse


def compute_normalized_laplacian(similarity_matrix):
    # 计算度矩阵 D
    degree = similarity_matrix.sum(dim=1)
    degree_matrix = torch.diag(degree)

    # 计算未归一化的拉普拉斯矩阵 L = D - S
    laplacian_matrix = degree_matrix - similarity_matrix

    # 计算 D^{-1/2}
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-10))

    # 计算归一化的拉普拉斯矩阵 L_norm = D^{-1/2} L D^{-1/2}
    laplacian_norm = d_inv_sqrt @ laplacian_matrix @ d_inv_sqrt

    return laplacian_norm

def contrastive_loss_batch(z1, z2, temperature=1, lambda_reg=0.01, n_clusters=3):
    batch_size = 2000
    z1 = F.normalize(z1, dim=-1, p=2)
    z2 = F.normalize(z2, dim=-1, p=2)
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes)
    losses = []

    # Step 1: Perform KMeans clustering on z1 and z2
    kmeans_1 = KMeans(n_clusters=n_clusters).fit(z1.cpu().detach().numpy())  # Cluster z1
    kmeans_2 = KMeans(n_clusters=n_clusters).fit(z2.cpu().detach().numpy())  # Cluster z2

    # Get cluster centroids and labels
    centroids_2 = torch.tensor(kmeans_2.cluster_centers_, device=z2.device)  # Centroids for z2
    labels_2 = torch.tensor(kmeans_2.labels_, device=z2.device)  # Cluster labels for z2

    # Step 2: Compute positive pairs (z1[i] and z2[i], and z2's cluster centroid)
    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]

        # Original positive pairs: z1[mask] and z2[mask]
        inter_sim = f(torch.mm(z1[mask], z2.t()))
        pos_sim = inter_sim[:, mask]  # z1[i] and z2[i]
        pos_sim_diag = pos_sim.diag()  # Diagonal elements representing z1[i] and z2[i]

        # Step 3: Add z2's cluster centroid as positive pair
        centroids_for_batch = centroids_2[labels_2[mask]]  # Get centroids for z2[mask]
        inter_sim_with_centroid = f(torch.mm(z1[mask], centroids_for_batch.t()))  # z1[i] and z2's centroid
        pos_sim_with_centroid = inter_sim_with_centroid.diag()  # Diagonal elements for z1[i] and cluster centroids

        # Combine the positive similarities (original + centroid)
        combined_pos_sim = pos_sim_diag + pos_sim_with_centroid

        # Step 4: Compute negative pairs (same as before)
        intra_sim_11 = f(torch.mm(z1[mask], z1.t()))
        intra_sim_22 = f(torch.mm(z2[mask], z2.t()))

        epsilon = 1e-8
        denom_12 = intra_sim_11.sum(1) + inter_sim.sum(1) - intra_sim_11[:, mask].diag() + epsilon
        denom_21 = intra_sim_22.sum(1) + inter_sim.sum(1) - intra_sim_22[:, mask].diag() + epsilon

        # Step 5: Compute contrastive loss
        loss_12 = -torch.log((combined_pos_sim + epsilon) / denom_12)
        loss_21 = -torch.log((combined_pos_sim + epsilon) / denom_21)
        losses.append(loss_12 + loss_21)

    # Compute final contrastive loss
    contrastive_loss = torch.cat(losses).mean()

    # Step 6: Laplacian regularization remains the same
    inter_sim_full_1 = f(torch.mm(z1, z1.t()))
    inter_sim_full_2 = f(torch.mm(z2, z2.t()))
    laplacian_norm_1 = compute_normalized_laplacian(inter_sim_full_1)
    laplacian_norm_2 = compute_normalized_laplacian(inter_sim_full_2)

    reg_loss_z1 = torch.trace(z1.t() @ laplacian_norm_1 @ z1)
    reg_loss_z2 = torch.trace(z2.t() @ laplacian_norm_2 @ z2)
    reg_loss = (reg_loss_z1 + reg_loss_z2) / (num_nodes * 2)

    total_loss = contrastive_loss + lambda_reg * reg_loss
    return total_loss

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden, activation='relu', base_model=GCNConv):
        super(Encoder, self).__init__()
        self.base_model = base_model

        self.gcn1 = base_model(in_channels, hidden)
        self.gcn2 = base_model(hidden, out_channels)
        self.gcn3 = base_model(out_channels, hidden)
        self.gcn4 = base_model(hidden, in_channels)

        # 激活函数
        self.s = nn.Sigmoid()

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        enc_1 = self.activation(self.gcn1(x, edge_index))
        enc_2 = self.gcn2(enc_1, edge_index)

        adj_hat =self.s(torch.mm(enc_2, enc_2.t()))

        dec_1 = self.activation(self.gcn3(enc_2, edge_index))
        z_gcn_hat = self.activation(self.gcn4(dec_1, edge_index))

        return enc_2, adj_hat, z_gcn_hat

class Contra(Module):
    def __init__(self,
                 encoder,
                 hidden_size,
                 projection_size,
                 projection_hidden_size,
                 n_cluster,
                 v=1):
        super().__init__()

        # backbone encoder
        self.encoder = encoder

        # projection layer for representation contrastive
        self.rep_projector = MLP(hidden_size, projection_size, projection_hidden_size)
        # t-student cluster layer for clustering
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster, hidden_size), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def kl_cluster(self, z1: torch.Tensor, z2: torch.Tensor):
        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)  # q1 n*K
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z2.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return q1, q2

    def forward(self, feat, adj):
        h, adj_hat, h_gcn_hat = self.encoder(feat, adj)
        z = self.rep_projector(h)

        return h, z, adj_hat, h_gcn_hat
