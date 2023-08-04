import torch
import torch.nn as nn
# from layers import U_GCN, AvgReadout, Discriminator
from layers import U_GCN, GraphCNN, AvgReadout, Discriminator
import sys
sys.path.append("models/")

class U_DCI(nn.Module):
    # def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
    #     super(U_DCI, self).__init__()
    #     self.device = device
    #     self.ugcn = U_GCN(64, 64, 64, 16, 0.3, 0.2, 8)
    #     self.read = AvgReadout()
    #     self.sigm = nn.Sigmoid()
    #     self.disc = Discriminator(hidden_dim)

    def __init__ (self, emb_module, hidden_dim, config_emb, device):
        super(U_DCI, self).__init__()
        self.device = device
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)

        if emb_module == 'U_GCN':
            self.module = U_GCN(config_emb['input_dim'], config_emb['out_features'], config_emb['final_features'], config_emb['dropout'], config_emb['alpha'], config_emb['nheads'])
        elif emb_module == 'GIN':
            self.module = GraphCNN(config_emb['num_layers'], config_emb['num_mlp_layers'], config_emb['input_dim'], config_emb['hidden_dim'], config_emb['neighbor_pooling_type'], device)












    # def forward(self, seq1, seq2, sadj, sadj2, msk, samp_bias1, samp_bias2, cluster_info, cluster_num):
    #     h_1 = self.ugcn(seq1, sadj, sadj2)  #原feature
    #     h_2 = self.ugcn(seq2, sadj, sadj2)  # shuffled feature
    def forward(self, feats, shuf_feats, adj, msk, samp_bias1, samp_bias2, cluster_info, cluster_num):
        h_1 = self.module(feats, adj)
        h_2 = self.module(shuf_feats, adj)

        loss = 0
        batch_size = 1 # hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
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
            # print(f'ret_shape:{ret.shape}') ---> ret_shape:torch.Size([1, 10486])
            loss_tmp = criterion(ret, lbl)  # 用loss function计算loss，这里用的是BCEWithLogitsLoss，ret是分数，lbl是标签
            loss += loss_tmp

        return loss / cluster_num   # 返回loss的平均值













    # def get_emb(self, seq1, sadj, sadj2):
    #     h_1 = self.emb(seq1, sadj, sadj2)
    #     return h_1

    def get_emb(self, feats, adj):
        h_1 = self.module(feats, adj)
        return h_1