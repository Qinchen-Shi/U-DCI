import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.graphattention import GraphAttention


class GAT(nn.Module):
    def __init__(self, in_features, out_features, final_features, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(in_features, out_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(out_features * nheads, final_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class Attention(nn.Module): # 它会得到一个注意力系数
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z) # 183*3*1，就是把每一行的feature合成了一个
        # print(f'shape of w: {w.shape}')
        beta = torch.softmax(w, dim=1)  # 183*3*1，每个beta代表一个node在一种module下的注意力权重
        # print(f'beta: {beta.shape}')
        output = (beta * z).sum(1)  #beta是系数乘上了GAT处理过的adj，得到的结果是每个node在一种module下的特征，然后在node维度上求和，就变成了三种module的特征和
        # print(f'output: {output.shape}')
        # print(output)
        return output, beta