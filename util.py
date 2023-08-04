import numpy as np
from sklearn.model_selection import StratifiedKFold

import scipy.sparse as sp
import torch

def load_data(datasets, num_folds):
    # load the adjacency
    adj = np.loadtxt('./data/'+datasets+'.txt')
    num_user = len(set(adj[:, 0]))
    num_object = len(set(adj[:, 1]))
    adj = adj.astype('int')
    nb_nodes = np.max(adj) + 1
    edge_index = adj.T
    print('Load the edge_index done!')
    
    # load the user label
    label = np.loadtxt('./data/'+datasets+'_label.txt')
    y = label[:, 1]
    print('Ratio of fraudsters: ', np.sum(y) / len(y))
    print('Number of edges: ', edge_index.shape[1])
    print('Number of users: ', num_user)
    print('Number of objects: ', num_object)
    print('Number of nodes: ', nb_nodes)

    # split the train_set and validation_set
    split_idx = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    for (train_idx, test_idx) in skf.split(y, y):
        split_idx.append((train_idx, test_idx))
   
    # load initial features
    feats = np.load('./features/'+datasets+'_feature64.npy')

    return edge_index, feats, split_idx, label, nb_nodes

# main_dci里也有，晚点可以把它从main里删掉，注意这里把device删掉了
def preprocess_neighbors_sumavepool(edge_index, nb_nodes):  # 分别是边的索引，节点数，设备
    adj_idx = edge_index
        
    adj_idx_2 = torch.cat([torch.unsqueeze(adj_idx[1], 0), torch.unsqueeze(adj_idx[0], 0)], 0)  # edge_index上下颠倒来一遍，因为是无向图所以反过来的edge的idx也要加上
    adj_idx = torch.cat([adj_idx, adj_idx_2], 1)    # 拼接原来的+颠倒的

    self_loop_edge = torch.LongTensor([range(nb_nodes), range(nb_nodes)])   # 生成自环边，就是node自己到自己的edge的idx
    adj_idx = torch.cat([adj_idx, self_loop_edge], 1)   # 拼成原来+颠倒+自环
        
    adj_elem = torch.ones(adj_idx.shape[1]) # 生成对应的权重，这里是1

    adj = torch.sparse.FloatTensor(adj_idx, adj_elem, torch.Size([nb_nodes, nb_nodes])) # 使用了 PyTorch 的稀疏张量（torch.sparse.FloatTensor）来表示邻接矩阵。稀疏张量是一种高效存储和处理稀疏数据的数据结构，适用于处理大规模的图结构。

    return adj

# U-GCN的生成2个adj
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx












# def load_graph(dataset, num_nodes, device):
#     # edges = np.genfromtxt(dataset, dtype=np.int32).T
    
#     sadj = preprocess_neighbors_sumavepool(dataset, num_nodes)    # 这里的sadj是一个带自环的稀疏矩阵

#     sadj2 = torch.matmul(sadj.to_dense(), sadj.to_dense())  # 把sadj转成dense，然后做矩阵乘法，得到sadj2
#     sadj2 = sadj2.to_sparse()   # 再把sadj2转成稀疏矩阵

#     return sadj.to(device), sadj2.to(device)

def load_graph(dataset, num_nodes, emb_module, device):
    # edges = np.genfromtxt(dataset, dtype=np.int32).T
    adj = preprocess_neighbors_sumavepool(dataset, num_nodes)    # 这里的sadj是一个带自环的稀疏矩阵
    adj2 = None
    
    if emb_module == 'U_DCI':
        adj2 = torch.matmul(adj.to_dense(), adj.to_dense())  # 把sadj转成dense，然后做矩阵乘法，得到sadj2
        adj2 = adj2.to_sparse()   # 再把sadj2转成稀疏矩阵

    return [adj.to(device), adj2.to(device)] if adj2 is not None else adj.to(device)