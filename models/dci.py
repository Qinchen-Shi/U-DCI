import torch
import torch.nn as nn
from layers import GraphCNN, AvgReadout, Discriminator
import sys
sys.path.append("models/")

# 定义一个继承NN模型的DCI类
class DCI(nn.Module):   # 继承了pytorch里的nn模型就是神网模型，有很多功能比如计算和向前向后传播
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        super(DCI, self).__init__()
        self.device = device
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)   # 自定义的GIN model
        self.read = AvgReadout()    # 自定义的readout model
        self.sigm = nn.Sigmoid()    # sigmoid函数
        self.disc = Discriminator(hidden_dim)   # 自定义的discriminator model

    # NN的向前传播，就是prediction，因为继承了NN所以在类似train的时候会用到这些吧，我猜的
    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, cluster_info, cluster_num):
        h_1 = self.gin(seq1, adj)   # 先用GIN layer算出sequence1的embedding，就是features的表示形式，这里会不会是hidden state的意思
        h_2 = self.gin(seq2, adj)

        loss = 0
        batch_size = 1
        criterion = nn.BCEWithLogitsLoss() 
        for i in range(cluster_num):    #相比较DGI来说就是这里套了一个循环按cluster来跑的
            node_idx = cluster_info[i]  # cluster_info存的是每个cluster的信息

            h_1_block = torch.unsqueeze(h_1[node_idx], 0)   # 增加一个维度变成三维（batch_size, node_num, feature_num）
            c_block = self.read(h_1_block, msk)   # 用readout layer算出cluster的embedding，不知道这是不是汇聚的意思，msk是掩码会指定哪些节点是有效的
            c_block = self.sigm(c_block)    # 激活函数压缩一下
            h_2_block = torch.unsqueeze(h_2[node_idx], 0)

            lbl_1 = torch.ones(batch_size, len(node_idx))   # 生成一个全1的tensor，大小是batch_size*len(node_idx)
            lbl_2 = torch.zeros(batch_size, len(node_idx))  # 生成一个全0的tensor，大小是batch_size*len(node_idx)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)  # 把两个tensor拼接起来，变成batch_size*2*len(node_idx)，放到device上

            ret = self.disc(c_block, h_1_block, h_2_block, samp_bias1, samp_bias2)  # 用discriminator layer判断哪个特征属于哪类，应该是在做normal和abnormal的区分，ret可能是一个分数or概率
            loss_tmp = criterion(ret, lbl)  # 用loss function计算loss，这里用的是BCEWithLogitsLoss，ret是分数，lbl是标签
            loss += loss_tmp

        return loss / cluster_num   # 返回loss的平均值
    
    # 只是一个简单的embedding操作
    def get_emb(self, seq1, adj):
        h_1 = self.gin(seq1, adj)
        return h_1