import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, aggregator_type="mean"):
        super(GraphSAGELayer, self).__init__()
        self.aggregator_type = aggregator_type
        self.W = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        h = self.W(x)  # Apply linear transformation
        
        if self.aggregator_type == "mean":
            agg_neighbor = torch.spmm(adj, h)
            adj = adj.to_dense()
            agg_neighbor = agg_neighbor.div(adj.sum(1).unsqueeze(1) + 1e-6)
        else:
            raise NotImplementedError("Aggregator type '{}' not supported.".format(self.aggregator_type))
        
        output = F.relu(agg_neighbor)
        return output

class GraphSAGE(nn.Module):
    # def __init__(self, nfeat, nhid, nclass, dropout):
    def __init__(self, config_emb, attention, device):
        super(GraphSAGE, self).__init__()
        self.sage1 = GraphSAGELayer(config_emb['input_dim'], config_emb['nhid'])
        self.sage2 = GraphSAGELayer(config_emb['nhid'], config_emb['out'])
        self.dropout = config_emb['dropout']
        self.attention = attention
        self.att_module = Attention(config_emb['out'])

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, adj)
        if self.attention:
            x, _ = self.att_module(x)
        return F.log_softmax(x, dim=1)
