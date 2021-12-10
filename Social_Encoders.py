import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from Attention import Attention

class Social_Encoder(nn.Module):

    def __init__(self, u2e, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        self.u2e = u2e
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(3*self.embed_dim, self.embed_dim) # nn.Linear(2 * self.embed_dim, self.embed_dim)  #
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, nodes):

        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])
        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network

        #self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        #self_feats = self_feats.t()
        self_feats = self.u2e.weight[nodes]
        self_cluster = self.base_model.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats,self_cluster], dim=1)#self_feats#
        combined = F.relu(self.linear1(combined))
        combined = self.dropout(combined)
        return combined
