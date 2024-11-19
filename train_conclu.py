import argparse
from evaluation import eva
from module import *
from utils import get_dataset, setup_seed
from augmentation import *
from logger import Logger, metrics_info, record_info
import datetime
import warnings

warnings.filterwarnings("ignore")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_multi_scale(model, adj, x, drop_feature_rate, label, epochs, adj_real):
    clu = []
    best_acc = 0
    best_epoch = 0
    metrics = [' acc', ' nmi', ' ari', ' f1']
    logger = Logger(args.dataset + '==' + nowtime)
    logger.info(model)
    logger.info(args)
    logger.info(metrics_info(metrics))

    coarse_path = 'coarse'

    # 生成多尺度图
    G = nx.Graph()
    G.add_edges_from(adj.cpu().numpy().T)
    node_features = x.cpu()
    G = compute_edge_weights(G, node_features)
    scales = [0.3, 0.15, 0.06]  # 适当调整尺度比例
    coarse_graphs = multi_scale_graph_coarsening(G, scales=scales)

    # 将多尺度图转换为 PyTorch 的 edge_index 格式
    adj_coarse_scales = []
    for G_coarse in coarse_graphs:
        if G_coarse.number_of_edges() > 0:
            edge_index = torch.tensor(list(G_coarse.edges()), dtype=torch.long).t().contiguous()
            if edge_index.size(1) > 0:
                adj_coarse_scales.append(edge_index.to(adj.device))

    # 检查是否有有效的图用于训练
    if not adj_coarse_scales:
        print("Error: All coarse graphs are empty. Please check the graph files.")
        return clu


    # 创建 optimizer，加入 weight_decay 作为 L2 正则化项
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(epochs):
        model.train()

        x_aug = drop_feature(x, drop_feature_rate)

        x = x.to(device)
        adj_real = adj_real.to(device)

        # 在多尺度下继续使用特征进行训练
        h1, z1, adj_hat_1, h1_gcn_hat = model(x, adj)
        h2_0, z2_0, adj_hat_2_0, h2_0_gcn_hat = model(x_aug, adj_coarse_scales[0].to(device))
        h2_1, z2_1, adj_hat_2_1, h2_1_gcn_hat = model(x_aug, adj_coarse_scales[1].to(device))
        h2_2, z2_2, adj_hat_2_2, h2_2_gcn_hat = model(x_aug, adj_coarse_scales[2].to(device))

        h2 = (0.3 * h2_0 + 0.15 * h2_1 + 0.06 * h2_2) / 3
        z2 = (0.3 * z2_0 + 0.15 * z2_1 + 0.06 * z2_2) / 3
        adj_hat_2 = (0.3 * adj_hat_2_0 + 0.15 * adj_hat_2_1 + 0.06 * adj_hat_2_2) / 3
        h2_gcn_hat = (0.3 * h2_0_gcn_hat + 0.15 * h2_1_gcn_hat + 0.06 * h2_2_gcn_hat) / 3

        q1, q2 = model.kl_cluster(h1, h2)

        q1_pred = q1.detach().cpu().numpy().argmax(1)
        acc, nmi, ari, f1 = eva(label, q1_pred, 'Q1_self_cluster', True)
        logger.info("epoch%d%s:\t%s" % (epoch, ' Q1', record_info([acc, nmi, ari, f1])))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

        if epoch % args.update_p == 0:
            p1 = target_distribution(q1.data)

        loss_w = F.mse_loss(h1_gcn_hat + h2_gcn_hat, torch.spmm(adj_real, x))
        loss_a = F.mse_loss(adj_hat_1 + adj_hat_2, adj_real.to_dense())
        loss_gae = loss_w + 0.1 * loss_a

        kl1 = F.kl_div(q1.log(), p1, reduction='batchmean')
        kl2 = F.kl_div(q2.log(), p1, reduction='batchmean')
        con = F.kl_div(q2.log(), q1, reduction='batchmean')
        clu_loss = kl1 + kl2 + con

        l_h = contrastive_loss_batch(h1, h2, n_clusters=n_cluster)
        enc_loss = 0.5 * l_h.mean()

        l_z = contrastive_loss_batch(z1, z2, n_clusters=n_cluster)
        pro_loss = 0.5 * l_z.mean()

        # 总损失，包括正则化
        loss = args.rep * enc_loss + args.pro * pro_loss + args.clu * clu_loss + loss_gae

        clu.append((acc, nmi, ari, f1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Best accuracy: {best_acc:.4f} at epoch {best_epoch}')
    return clu, logger, best_epoch


if __name__ == '__main__':
    setup_seed(2018)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--pro_hid', type=int, default=1024)

    parser.add_argument('--mask', type=float, default=0.2)

    parser.add_argument('--rep', type=float, default=1)
    parser.add_argument('--clu', type=float, default=1)
    parser.add_argument('--pro', type=float, default=1)
    parser.add_argument('--update_p', type=int, default=1)

    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1.00E-02)


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
    elif args.dataset == 'hhar':
        args.n_clusters = 6
        args.n_input = 561
    elif args.dataset == 'reut':
        args.n_clusters = 4
        args.n_input = 2000
    elif args.dataset == 'usps':
        args.n_clusters = 10
        args.n_input = 256
    elif args.dataset == 'wisc':
        args.n_clusters = 5
        args.n_input = 1703
    elif args.dataset == 'wiki':
        args.n_clusters = 17
        args.n_input = 4973
    elif args.dataset == 'texas':
        args.n_clusters = 5
        args.n_input = 1703
    elif args.dataset == 'amap':
        args.n_clusters = 8
        args.n_input = 745

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
                      activation='relu',  # 这里使用 ReLU 作为激活函数
                      base_model=GCNConv).to(device)

    model = Contra(encoder=encoder,
                   hidden_size=args.out_dim,
                   projection_hidden_size=args.pro_hid,
                   projection_size=args.pro_hid,
                   n_cluster=n_cluster).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.00E-02)

    # load pre-train for clustering initialization
    save_model = torch.load('pretrain/{}_contra.pkl'.format(args.dataset), map_location='cpu')
    # 使用学习率调度器，逐步减小学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    model.encoder.load_state_dict(save_model)
    with torch.no_grad():
        h_o, z_o, _, _ = model(features, edge_index)
    kmeans = KMeans(n_clusters=n_cluster, n_init=20)
    clu_pre = kmeans.fit_predict(h_o.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(labels, clu_pre, 'Initialization')

    # 根据 --use_saved 参数，控制是否直接读取文件
    clu_acc, logger, best_epoch = train_multi_scale(model, edge_index, features, feature_drop, labels, args.epochs, adj_real)
    clu_q_max = np.max(np.array(clu_acc), 0)
    logger.info("%sepoch%d:\t%s" % ('Best Acc is at ', best_epoch, record_info(clu_q_max)))
    clu_q_final = clu_acc[-1]
