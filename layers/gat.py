import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

    def forward(self, x, adj):
        h = self.W(x)  # Apply linear transformation
        N = h.size(0)
        
        # Prepare attention coefficients using broadcasting
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        e = self.a(a_input).view(N, N)
        attention = F.softmax(e, dim=1)
        
        # Apply attention to the adjacency matrix
        adj_att = adj * attention
        
        # Perform graph convolution
        output = torch.spmm(adj_att, h)
        return output

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GAT, self).__init__()
        self.gat1 = GraphAttentionLayer(nfeat, nhid)
        self.gat2 = GraphAttentionLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        return F.log_softmax(x, dim=1)
