import torch.nn as nn
import torch
import torch.nn.functional as F
# from gat import GAT
from .gat import GAT
from .attention import Attention



class U_GCN(nn.Module):
    # def __init__(self, in_features, out_features, final_features, dropout, alpha, nheads):
    #     super(U_GCN, self).__init__()
    #     self.SGAT1 = GAT(in_features, out_features, final_features, dropout, alpha, nheads)
    #     self.SGAT2 = GAT(in_features, out_features, final_features, dropout, alpha, nheads)
    #     self.attention = Attention(final_features)


    def __init__(self, config_emb, attention, device):
        super(U_GCN, self).__init__()
        self.SGAT1 = GAT(config_emb['input_dim'], config_emb['out_features'], config_emb['final_features'], config_emb['dropout'], config_emb['alpha'], config_emb['nheads'])
        self.SGAT2 = GAT(config_emb['input_dim'], config_emb['out_features'], config_emb['final_features'], config_emb['dropout'], config_emb['alpha'], config_emb['nheads'])
        self.attention = Attention(config_emb['final_features'])
        
    
    


    
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
        emb, _ = self.attention(emb)
        return emb
