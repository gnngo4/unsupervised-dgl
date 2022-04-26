import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,g,in_feats,hidden_dims,activation,dropout,distribution='none'):
        super(GCN,self).__init__()
        self.g = g
        self.distribution = distribution
        self.dropout = nn.Dropout(p=dropout)
        n_layers = len(hidden_dims)
        self.layers = nn.ModuleList()
        # encoder
        if n_layers == 1:
            self.encode_z(in_feats,hidden_dims[0])
        elif n_layers >= 2:
            self.layers.append(GraphConv(in_feats,hidden_dims[0],activation=activation))
            for i in range(1,n_layers):
                if i != n_layers - 1:
                    self.layers.append(GraphConv(hidden_dims[i-1],hidden_dims[i],activation=activation))
                else:
                    self.encode_z(hidden_dims[i-1],hidden_dims[i])
        else:
            raise NotImplemented

    def encode_z(self,in_dim,z_dim):
        if self.distribution == 'normal':
            # Compute mean and std of the normal distribution
            self.gc_mean = GraphConv(in_dim,z_dim,activation=lambda x:x)
            self.gc_var = GraphConv(in_dim,z_dim,activation=lambda x:x)
        elif self.distribution == 'hypersphere':
            # Compute loc and scale of the power spherical distribution
            self.gc_mean = GraphConv(in_dim,z_dim,activation=lambda x:x)
            self.gc_var = GraphConv(in_dim,1,activation=lambda x:x)
        elif self.distribution == 'none':
            self.gc_output = GraphConv(in_dim,z_dim,activation=lambda x:x)
        else:
            raise NotImplemented

    def forward(self,features):
        z = features
        # hidden layer
        for i,layer in enumerate(self.layers):
            if i != 0:
                z = self.dropout(z)
            z = layer(self.g,z)
        # output layer
        z = self.dropout(z)
        if self.distribution == 'normal':
            z_mean = self.gc_mean(self.g,z)
            z_var = F.softplus(self.gc_var(self.g,z))

            return z_mean, z_var
        
        elif self.distribution == 'hypersphere':
            # compute loc and scale of the power spherical distribution
            z_mean = self.gc_mean(self.g,z)
            z_mean = z_mean / z_mean.norm(dim=-1,keepdim=True)
            # the `+1` prevent collapsing behaviors
            z_var = F.softplus(self.gc_var(self.g,z).reshape(-1,))+1

            return z_mean, z_var
        
        else:
            z = self.gc_output(self.g,z)

            return z
