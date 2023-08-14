import torch.nn as nn
from layers import GraphCNN, MLP, U_GCN, GCN, GAT, GraphSAGE
import torch.nn.functional as F
import sys
sys.path.append("models/")

# 只用在了main.py里的finetune函数里，只是为了算AUC值，可以理解成
# class Classifier(nn.Module):    # 因为继承了NN所以很多东西不用写在这里
#     def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, final_dropout, neighbor_pooling_type, device):
#         super(Classifier, self).__init__()
#         self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
#         self.linear_prediction = nn.Linear(hidden_dim, 1)
#         self.final_dropout = final_dropout
        
#     def forward(self, seq1, adj):
#         h_1 = self.gin(seq1, adj)   # 先是生成features embedding
#         score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training = self.training)    # 分别是张量，dropout概率和是不是training set
#         return score_final_layer    # 返回的是失活张量





# class Classifier(nn.Module):    # 因为继承了NN所以很多东西不用写在这里
#     def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, final_dropout, neighbor_pooling_type, device):
#         super(Classifier, self).__init__()
#         self.ugcn = U_GCN(64, 64, 64, 16, 0.3, 0.2, 8)
#         self.linear_prediction = nn.Linear(hidden_dim, 1)
#         self.final_dropout = final_dropout
        
#     def forward(self, seq1, sadj, sadj2):
#         h_1 = self.ugcn(seq1, sadj, sadj2)   # 先是生成features embedding
#         score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training = self.training)    # 分别是张量，dropout概率和是不是training set
#         return score_final_layer






class Classifier(nn.Module):
    def __init__(self, emb_module, hidden_dim, final_dropout, config_emb, attention, device):
        super(Classifier, self).__init__()
    #     if emb_module == 'U_GCN':
    #         self.emb_module = U_GCN(config_emb['input_dim'], config_emb['out_features'], config_emb['final_features'], config_emb['dropout'], config_emb['alpha'], config_emb['nheads'])
    #     elif emb_module == 'GIN':
    #         self.emb_module = GraphCNN(config_emb['num_layers'], config_emb['num_mlp_layers'], config_emb['input_dim'], config_emb['hidden_dim'], config_emb['neighbor_pooling_type'], device)
    #     elif emb_module == 'GCN':
    #         self.emb_module = GCN(config_emb['input_dim'], config_emb['nhid'], config_emb['out'], config_emb['dropout'])
    #     elif emb_module == 'GAT':
    #         self.emb_module = GAT(config_emb['input_dim'], config_emb['nhid'], config_emb['out'], config_emb['dropout'])
    #     elif emb_module == 'GraphSAGE':
    #         self.emb_module = GraphSAGE(config_emb['input_dim'], config_emb['nhid'], config_emb['out'], config_emb['dropout']

        module = {
            'U_GCN': U_GCN,
            'GIN': GraphCNN,
            'GCN': GCN,
            'GAT': GAT,
            'GraphSAGE': GraphSAGE
        }
        self.module = module[emb_module](config_emb, attention, device)

        self.linear_prediction = nn.Linear(hidden_dim, 1)
        self.final_dropout = final_dropout

    def forward(self, feats, adj):
        h_1 = self.module(feats, adj)
        score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training = self.training)
        return score_final_layer