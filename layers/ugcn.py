import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np



class GraphAttention(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    


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
    



class U_GCN(nn.Module):
    # def __init__(self, in_features, nclass, out_features, final_features, dropout, alpha, nheads):
    def __init__(self, in_features, out_features, final_features, dropout, alpha, nheads):
        super(U_GCN, self).__init__()

        # use GCN or GAT
        self.SGAT1 = GAT(in_features, out_features, final_features, dropout, alpha, nheads)
        self.SGAT2 = GAT(in_features, out_features, final_features, dropout, alpha, nheads)
        self.attention = Attention(final_features)

    # def forward(self, x, sadj, sadj2):
    #     emb1 = self.SGAT1(x, sadj) 
    #     emb2 = self.SGAT2(x, sadj2)
    #     emb = torch.stack([emb1, emb2], dim=1)
    #     emb, att = self.attention(emb)
    #     return emb # 剩下的等需要了再加

    def forward(self, feature, adj):
        emb1 = self.SGAT1(feature, adj[0]) 
        emb2 = self.SGAT2(feature, adj[1])
        emb = torch.stack([emb1, emb2], dim=1)
        emb, att = self.attention(emb)
        return emb
