import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSAGE, self).__init__()
        self.sage1 = GraphSAGELayer(nfeat, nhid)
        self.sage2 = GraphSAGELayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, adj)
        return F.log_softmax(x, dim=1)
