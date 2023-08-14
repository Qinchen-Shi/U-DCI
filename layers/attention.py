import torch.nn as nn
import torch



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
        beta = torch.softmax(w, dim=1)  # 183*3*1，每个beta代表一个node在一种module下的注意力权重
        output = (beta * z).sum(1)  #beta是系数乘上了GAT处理过的adj，得到的结果是每个node在一种module下的特征，然后在node维度上求和，就变成了三种module的特征和
        return output