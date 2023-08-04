import torch.nn as nn
from layers.gat import GAT, Attention
import torch

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
