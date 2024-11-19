import argparse
from evaluation import eva
from module import *
from utils import *
from augmentation import *
import warnings

warnings.filterwarnings("ignore")


def train(model, adj, x, drop_feature_rate, label, epochs, adj_real):
    adj_real = adj_real.to(device)
    # 调用图粗化方法，将图简化到指定的大小
    G = nx.Graph()
    G.add_edges_from(adj.cpu().numpy().T)  # 将 PyTorch 的 edge_index 转换为 NetworkX 图
    node_features = x.cpu()
    G = compute_edge_weights(G, node_features)
    target_size = max(10, int(adj.size(1) * 0.1))  # 设置目标规模，可以根据需要调整
    G_coarse = fast_graph_coarsening(G, target_size=target_size)


    # 将简化后的图转换回 PyTorch 的 edge_index
    edge_list = list(G_coarse.edges())
    adj_coarse = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(adj.device)

    for epoch in range(epochs):
        # get feature augmentation
        x_aug = drop_feature(x, drop_feature_rate)

        # learning representation
        # 使用简化后的图进行训练
        h1, z1, adj_hat_1, h1_gcn_hat = model(x, adj)
        h2, z2, adj_hat_2, h2_gcn_hat = model(x_aug, adj_coarse)

        loss_w = F.mse_loss(h1_gcn_hat + h2_gcn_hat, torch.spmm(adj_real, x))
        loss_a = F.mse_loss(adj_hat_1 + adj_hat_2, adj_real.to_dense())
        loss_gae = loss_w + 0.1 * loss_a

        # 公式3
        l_h = contrastive_loss_batch(h1, h2)
        en_loss = 0.5 * l_h.mean()

        # 公式4
        l_z = contrastive_loss_batch(z1, z2)
        pro_loss = 0.5 * l_z.mean()

        # 公式5
        loss = args.rep * en_loss + args.pro * pro_loss + loss_gae
        print('Epoch [{:2}/{}]: loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # h是GCN的输出，z是GCN-MLP的输出
    with torch.no_grad():
        # h, z = model(x, adj_coarse)
        h, z, _, _ = model(x, adj)
    # k-means with node representation
    kmeans = KMeans(n_clusters=n_cluster, n_init=20)
    cluster_z = kmeans.fit(z.data.cpu().numpy())
    y_z = cluster_z.labels_
    clu_z = eva(label, y_z, 'z-representation-kmeans')
    cluster_h = kmeans.fit(h.data.cpu().numpy())
    y_h = cluster_h.labels_
    clu_h = eva(label, y_h, 'h-representation-kmeans')

    torch.save(model.encoder.state_dict(), 'pretrain/{}_contra.pkl'.format(args.dataset))

    return clu_z, clu_h

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--pro_hid', type=int, default=1024)

    # 节点属性
    parser.add_argument('--mask', type=float, default=0.2)

    parser.add_argument('--rep', type=float, default=1)
    parser.add_argument('--pro', type=float, default=1)

    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--project', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    if args.dataset == 'acm':
        args.n_clusters = 3
        args.n_input = 1870
    elif args.dataset == 'dblp':
        args.n_clusters = 4
        args.n_input = 334
    elif args.dataset == 'cite':
        args.n_clusters = 6
        args.n_input = 3703
    elif args.dataset == 'cora':
        args.n_clusters = 7
        args.n_input = 1433
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x, labels, adj_real, edge_index = get_dataset(args.dataset)
    x = x.astype(float)
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = edge_index.T
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    features = x

    n_cluster = args.n_clusters

    features = features.to(device)
    edge_index = edge_index.to(device)

    feature_drop = args.mask

    # model
    encoder = Encoder(in_channels=args.n_input,
                      out_channels=args.out_dim,
                      hidden=args.hidden,
                      activation=args.activation)

    model = Contra(encoder=encoder,
                   hidden_size=args.out_dim,
                   projection_hidden_size=args.pro_hid,
                   projection_size=args.pro_hid,
                   n_cluster=n_cluster).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.00E-02)

    clu_z, clu_h = train(model, edge_index, features, feature_drop, labels, args.epochs, adj_real)