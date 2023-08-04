import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import torch.optim as optim
from sklearn import metrics

from util import load_data, load_graph
from models.udci import U_DCI
from models.clf_model import Classifier

sig = torch.nn.Sigmoid()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def finetune(args, model_pretrain, device, test_graph, feats_num):
    model = Classifier(args.num_layers, args.num_mlp_layers, feats_num, args.hidden_dim, args.final_dropout, args.neighbor_pooling_type, device).to(device)
    
    # replace the encoder in joint model with the pre-trained encoder，把外面训练好的CDI导进来，这个视角encoder吗
    pretrained_dict = model_pretrain.state_dict()   # state_dict()是pytorch里调用所有参数信息的函数，这里是把DCI的参数信息存起来
    model_dict = model.state_dict() #这里就是吧Classifier的参数信息存起来
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}    # 只留下Classifier里有的字段的DCI的参数信息，为了防止下一步错叭
    model_dict.update(pretrained_dict)  # 这两步就是把Classifier的参数信息更新成DCI的参数信息
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 优化器的意思应该是反向传播的那个，不要纠结细节上次学过了虽然我知道你忘了
    
    criterion_tune = nn.BCEWithLogitsLoss() # 二分类的交叉熵损失函数

    res = []
    train_idx = test_graph[2]
    node_train = test_graph[-1][train_idx, 0].astype('int')
    label_train = torch.FloatTensor(test_graph[-1][train_idx, 1]).to(device)
    for _ in range(1, args.finetune_epochs+1):  # 这一段应该就是在训练
        model.train()
        output = model(test_graph[0], test_graph[1])
        loss = criterion_tune(output[node_train], torch.reshape(label_train, (-1, 1)))
        
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # testing
        model.eval()    # 设成eval模式
        auc = evaluate(model, test_graph)
        res.append(auc)

    return np.max(res)  # 返回最大的auc

def evaluate(model, test_graph):
    output = model(test_graph[0], test_graph[1])    #预测结果，其中0和1分别是adj和features，顺序记得确认一下
    pred = sig(output.detach().cpu())   # 用激活函数压缩一下然后结果传到cpu上
    test_idx = test_graph[3]    # 这个idx具体指什么要看model，我猜是abnormal data的idx
    
    labels = test_graph[-1] # 那么labels我猜就是abnormal data的label
    pred = pred[labels[test_idx, 0].astype('int')].numpy()  # 不知道了这里回头再来看吧
    target = labels[test_idx, 1]
    
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(false_positive_rate, true_positive_rate)  # auc越接近1越好

    return auc




def main():
    parser = argparse.ArgumentParser(description='PyTorch deep cluster infomax')
    parser.add_argument('--dataset', type=str, default="wiki",
                        help='name of dataset (default: wiki)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers (default: 2)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='number of finetune epochs (default: 100)')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='number of folds (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num_cluster', type=int, default=2,
                        help='number of clusters (default: 2)')
    parser.add_argument('--recluster_interval', type=int, default=20,   # 指重新聚类的间隔，这里是每train 20次就重新聚类一次
                        help='the interval of reclustering (default: 20)')
    parser.add_argument('--final_dropout', type=float, default=0.5,     # 指定模型在最后一层有几个node会变成0，是用来防止过拟合的，介于0-1，e.g. 0.5就是一半的node会变成0
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"], # 指的是对每个node，它的adj node的feature是做sum还是average
                        help='Pooling for over neighboring nodes: sum or average')
    parser.add_argument('--training_scheme', type=str, default="decoupled", choices=["decoupled", "joint"], # 用decoupled还是joint，因为对比的是DCI和别的所以没有提供joint DGI，我猜的
                        help='Training schemes: decoupled or joint')
    # parser.add_argument('--module', type=str, default='U-GCN', choices=['DCI', 'U-GCN'],
    #                     help='module to generate feature matrix: DCI or U-GCN')
    args = parser.parse_args()

    setup_seed(0)

    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")   # 听说这里可以改简单一点
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_index, feats, split_idx, label, nb_nodes = load_data(args.dataset, args.num_folds) # 这里的feats是自动生成的，不用管
    input_dim = feats.shape[1]

    kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(feats)
    ss_label = kmeans.labels_
    cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]

    idx = np.random.permutation(nb_nodes)
    shuf_feats = feats[idx, :]

    adj_1hop, adj_2hop = load_graph(torch.LongTensor(edge_index), nb_nodes, device)
    feats = torch.FloatTensor(feats).to(device)
    shuf_feats = torch.FloatTensor(shuf_feats).to(device)

    model_pretrain = U_DCI(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, args.neighbor_pooling_type, device).to(device)
    optimizer_train = optim.Adam(model_pretrain.parameters(), lr=args.lr)


        
    for epoch in range(1, args.epochs + 1):
        model_pretrain.train()
        loss_pretrain = model_pretrain(feats, shuf_feats, adj_1hop, adj_2hop, None, None, None, cluster_info, args.num_cluster)
        if optimizer_train is not None:
            optimizer_train.zero_grad()
            loss_pretrain.backward()         
            optimizer_train.step()
        # re-clustering
        if epoch % args.recluster_interval == 0 and epoch < args.epochs:
            model_pretrain.eval()
            emb = model_pretrain.get_emb(feats, adj_1hop, adj_2hop)
            kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
            ss_label = kmeans.labels_
            cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]

    print('Pre-training Down!')

    #fine-tuning process
    fold_idx = 1
    every_fold_auc = []
    for (train_idx, test_idx) in split_idx: # split_idx在load data的时候就生成了
        test_graph = (feats, adj_1hop, train_idx, test_idx, label)
        tmp_auc = finetune(args, model_pretrain, device, test_graph, input_dim)
        every_fold_auc.append(tmp_auc)
        print('AUC on the Fold'+str(fold_idx)+': ', tmp_auc)    # 会返回每种folder的auc，每个auc是模型优化后最好的auc（因为是max）
        fold_idx += 1
    print('The averaged AUC score: ', np.mean(every_fold_auc))

if __name__ == '__main__':
    main()