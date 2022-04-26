import torch.nn as nn
import math
from unsupervised_models.GCNEncoder import GCN
from unsupervised_models.utils import *

from power_spherical.power_spherical import HypersphericalUniform
from power_spherical.power_spherical import PowerSpherical

'''
pytorch implementation of variational graph autoencoder using deep graph library
distributions : (1) 'normal'
                (2) 'powerspherical' 
'''

class Encoder(nn.Module):
    def __init__(self,g,in_feats,hidden_dims,activation,dropout,distribution):
        super(Encoder, self).__init__()
        self.g = g
        self.distribution = distribution
        self.conv = GCN(g,in_feats,hidden_dims,activation,dropout,distribution)

    def forward(self, features):
        z_mean, z_var = self.conv(features)

        return z_mean, z_var

class VGAE(nn.Module):
    def __init__(self,g,in_feats,hidden_dims,activation,dropout,distribution='normal'):
        super(VGAE, self).__init__()
        self.g = g; self.g.readonly()
        self.distribution = distribution
        self.encoder = Encoder(g,in_feats,hidden_dims,activation,dropout,distribution)
        self.z_dim = hidden_dims[-1]

        self.adj = g.adjacency_matrix().to_dense()
        self.n_nodes = self.adj.shape[0]

    def reparameterize(self,z_mean,z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean,z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean),torch.ones_like(z_var))
        elif self.distribution == 'hypersphere':
            q_z = PowerSpherical(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim-1)
        else:
            raise NotImplemented
        
        return q_z, p_z

    def contrastiveloss(self,g,z,neg_sample_size):
        pos_g, neg_g = edge_sampler(g,neg_sample_size,return_false_neg=False)
        pos_score = score_func(pos_g,z)
        neg_score = score_func(neg_g,z)

        return torch.mean(NCE_loss(pos_score,neg_score,neg_sample_size))

    def forward(self,features,KL_weight=1.,neg_sample_size=1):
        z_mean, z_var = self.encoder(features)
        q_z, p_z = self.reparameterize(z_mean,z_var)
        z = q_z.rsample()
        loss_recon = self.contrastiveloss(self.g,z,neg_sample_size)

        if self.distribution == 'normal':
            loss_KL = KL_weight * (0.5/self.n_nodes) * torch.distributions.kl.kl_divergence(q_z,p_z).sum(-1).mean()
        elif self.distribution == 'hypersphere':
            loss_KL = KL_weight * (0.5/self.n_nodes) * torch.distributions.kl.kl_divergence(q_z,p_z).mean()
        else:
            raise NotImplemented

        loss = loss_recon + loss_KL

        return loss
