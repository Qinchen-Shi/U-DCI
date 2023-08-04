import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics

from util import load_data
from models.clf_model import Classifier
from models.dci import DCI
from sklearn.cluster import KMeans

sig = torch.nn.Sigmoid()    # ig是一个sigmoid激活函数，sigmoid(x) = 1 / (1 + exp(-x))

# 设置随机种子使随机结果可以复现
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  

# 生成一个无向adjacency matrix并转到对应的device上
def preprocess_neighbors_sumavepool(edge_index, nb_nodes, device):  # 分别是边的索引，节点数，设备
    adj_idx = edge_index
        
    adj_idx_2 = torch.cat([torch.unsqueeze(adj_idx[1], 0), torch.unsqueeze(adj_idx[0], 0)], 0)  # edge_index上下颠倒来一遍，因为是无向图所以反过来的edge的idx也要加上
    adj_idx = torch.cat([adj_idx, adj_idx_2], 1)    # 拼接原来的+颠倒的

    self_loop_edge = torch.LongTensor([range(nb_nodes), range(nb_nodes)])   # 生成自环边，就是node自己到自己的edge的idx
    adj_idx = torch.cat([adj_idx, self_loop_edge], 1)   # 拼成原来+颠倒+自环
        
    adj_elem = torch.ones(adj_idx.shape[1]) # 生成对应的权重，这里是1

    adj = torch.sparse.FloatTensor(adj_idx, adj_elem, torch.Size([nb_nodes, nb_nodes])) # 使用了 PyTorch 的稀疏张量（torch.sparse.FloatTensor）来表示邻接矩阵。稀疏张量是一种高效存储和处理稀疏数据的数据结构，适用于处理大规模的图结构。

    return adj.to(device)   # 返回adjacency matrix并转到对应的device上

# 评估模型在test set上的性能
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

# 这其实是一个cross validation函数
def finetune(args, model_pretrain, device, test_graph, feats_num):
    # initialize the joint model，所以使用cluster的方法optimize model然后还是用普通的joint model来验证的
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

def main():
    # 这一段是传参数or文件or模型，总之是执行句
    # 默认wiki dataset, use gpu 0, epochs = 50, layers = 2, hidden_dim = 128, finetune_epochs = 100, lr = 0.01
    # cluster = 2, recluster interval = 20, dropout = 0.5, neighbor pooling type = sum, scheme = decoupled
    parser = argparse.ArgumentParser(description='PyTorch deep cluster infomax')
    parser.add_argument('--dataset', type=str, default="wiki",
                        help='name of dataset (default: wiki)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers (default: 2)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 128)')
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
    args = parser.parse_args()  # args是用来存上面的参数或者叫命令的

    setup_seed(0)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu") # 这里是指定用哪个gpu，如果没有gpu就用cpu，可是用cpu会报错啊

    # data loading
    edge_index, feats, split_idx, label, nb_nodes = load_data(args.dataset, args.num_folds)
    input_dim = feats.shape[1]
    # pre-clustering and store userID in each clusters
    kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(feats) # 用KMeans把raw user data分成两类的意思，可以理解成两种业务or行为？
    ss_label = kmeans.labels_   # ss_label是每个node的cluster label
    cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]    # 把每个cluster的node的id存起来
    # the shuffled features are used to contruct the negative sample-pairs
    idx = np.random.permutation(nb_nodes)
    shuf_feats = feats[idx, :]  # 把node的idx打乱生成乱的features序列，为了做negative sample来做对比训练

    adj = preprocess_neighbors_sumavepool(torch.LongTensor(edge_index), nb_nodes, device)   # 用前面的函数生成了一个adj matrix
    feats = torch.FloatTensor(feats).to(device) # 把features转成tensor放到gpu上
    shuf_feats = torch.FloatTensor(shuf_feats).to(device)   # 把shuffled features转成tensor放到gpu上

    # pre-training process
    model_pretrain = DCI(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, args.neighbor_pooling_type, device).to(device)   # 看来这是一个joint的DCI
    # 下面的基本逻辑是：如果用decoupled，就先用DCI学data，通过loss优化node的features，每20次重新算一下cluster用来算之后的loss，直到循环结束
    if args.training_scheme == 'decoupled':
        optimizer_train = optim.Adam(model_pretrain.parameters(), lr=args.lr)   # 用定好的lr设一个优化器
        for epoch in range(1, args.epochs + 1):
            model_pretrain.train()
            loss_pretrain = model_pretrain(feats, shuf_feats, adj, None, None, None, cluster_info, args.num_cluster)
            if optimizer_train is not None:
                optimizer_train.zero_grad()
                loss_pretrain.backward()         
                optimizer_train.step()
            # re-clustering
            if epoch % args.recluster_interval == 0 and epoch < args.epochs:
                model_pretrain.eval()
                emb = model_pretrain.get_emb(feats, adj)
                kmeans = KMeans(n_clusters=args.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
                ss_label = kmeans.labels_
                cluster_info = [list(np.where(ss_label==i)[0]) for i in range(args.num_cluster)]
        
        print('Pre-training Down!')
            
    #fine-tuning process
    fold_idx = 1
    every_fold_auc = []
    for (train_idx, test_idx) in split_idx: # split_idx在load data的时候就生成了
        test_graph = (feats, adj, train_idx, test_idx, label)
        tmp_auc = finetune(args, model_pretrain, device, test_graph, input_dim)
        every_fold_auc.append(tmp_auc)
        print('AUC on the Fold'+str(fold_idx)+': ', tmp_auc)    # 会返回每种folder的auc，每个auc是模型优化后最好的auc（因为是max）
        fold_idx += 1
    print('The averaged AUC score: ', np.mean(every_fold_auc))  #print综合auc


if __name__ == '__main__':
    main()
