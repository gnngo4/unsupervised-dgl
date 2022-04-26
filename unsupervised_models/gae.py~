import torch
import torch.nn as nn
import math
from unsupervised_models.GCNEncoder import GCN
from unsupervised_models.utils import *

'''
pytorch implementation of graph autoencoder using deep graph library
'''

class Encoder(nn.Module):
    def __init__(self,g,in_feats,hidden_dims,activation,dropout):
        super(Encoder,self).__init__()
        self.g = g
        self.conv = GCN(g,in_feats,hidden_dims,activation,dropout)

    def forward(self,features):
        features = self.conv(features)

        return features

class GAE(nn.Module):
    def __init__(self,g,in_feats,hidden_dims,activation,dropout):
        super(GAE,self).__init__()
        self.g = g; self.g.readonly()
        self.encoder = Encoder(g,in_feats,hidden_dims,activation,dropout)

        self.adj = g.adjacency_matrix().to_dense()
        self.n_nodes = self.adj.shape[0]

    def ContrastiveLoss(self,g,z,neg_sample_size):
        '''
        Use negative sampling of true positive and negative edges as proposed by Kipf et al. 2016
        Note: In this paper they used an InnerProductDecoder
        '''
        # Sample a positive and negative subgraph 
        # Positive/negative subgraph includes only true/false edges, respectively
        pos_g, neg_g = edge_sampler(g,neg_sample_size,return_false_neg=False)
        # Calculate loss for positive and negative subgraph
        pos_score = score_func(pos_g,z)
        neg_score = score_func(neg_g,z)

        return torch.mean(NCE_loss(pos_score,neg_score,neg_sample_size))

    def forward(self,features,neg_sample_size=1):
        z = self.encoder(features)
        loss = self.ContrastiveLoss(self.g,z,neg_sample_size)

        return loss
