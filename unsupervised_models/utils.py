import dgl
import torch
import torch.nn.functional as F

def edge_sampler(g, neg_sample_size, edges=None, return_false_neg=True):
    sampler = dgl.contrib.sampling.EdgeSampler(g, batch_size=int(g.number_of_edges()/10),
                                                seed_edges=edges,
                                                neg_sample_size=neg_sample_size,
                                                negative_mode='tail',
                                                shuffle=True,
                                                return_false_neg=return_false_neg)
    sampler = iter(sampler)
    
    return next(sampler)


def score_func(g,z):
    src_nid, dst_nid = g.all_edges(order='eid')
    # get the node IDs in the parent graph
    src_nid = g.parent_nid[src_nid]
    dst_nid = g.parent_nid[dst_nid]
    # read the node embeddings of the source nodes and destination nodes
    pos_heads = z[src_nid]
    pos_tails = z[dst_nid]
    # Cosine similarity
    
    return torch.sum(pos_heads * pos_tails, dim=1)


def NCE_loss(pos_score,neg_score,neg_sample_size):
    pos_score = F.logsigmoid(pos_score)
    neg_score = F.logsigmoid(-neg_score).reshape(-1, neg_sample_size)
    
    return -pos_score - torch.sum(neg_score,dim=1)
