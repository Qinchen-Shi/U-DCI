import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
from .attention import Attention

# 这是作者修改过的代码，合理怀疑是看了DGI为了改DCI套的GIN
class GraphCNN(nn.Module):
    # def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
    def __init__(self, config_emb, attention, device):
        '''
            num_layers: number of layers in the neural networks
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = config_emb['num_layers']
        self.neighbor_pooling_type = config_emb['neighbor_pooling_type']
        self.attention = attention
        self.att_module = Attention(config_emb['hidden_dim'])

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        # 这里做了一个loop控制当前的model是输入层还是中间层
        for layer in range(self.num_layers):
            if layer == 0:
                # self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.mlps.append(MLP(config_emb['num_mlp_layers'], config_emb['input_dim'], config_emb['hidden_dim'], config_emb['hidden_dim']))
            else:
                # self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.mlps.append(MLP(config_emb['num_mlp_layers'], config_emb['hidden_dim'], config_emb['hidden_dim'], config_emb['hidden_dim']))

            self.batch_norms.append(nn.BatchNorm1d(config_emb['hidden_dim'])) # 这句是输出层

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
        
        #If sum or average pooling
        pooled = torch.spmm(Adj_block, h)
        if self.neighbor_pooling_type == "average":
            #If average pooling
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            
            pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    
    def forward(self, feats, adj):
        h = feats
        
        if self.attention:
            for layer in range(self.num_layers):
                h = self.next_layer(h, layer, Adj_block = adj)
                h = self.att_module(h)
        else:
            for layer in range(self.num_layers):
                h = self.next_layer(h, layer, Adj_block = adj)

        return h
