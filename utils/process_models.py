import torch
import numpy as np
import os
import time 


'''
process.py - includes tools to process graph

TO DO:
'''

def train_model(model,features,hyperparameters,model_name, \
                n_epochs=200,patience=20,verbose=True,saved_dir='saved_models'):
    '''
    Embed features using unsupervised graph representational learning algorithm.
    Early stopping implemented.
    '''

    if 'nv' in model_name and 'gae' in model_name:
        print('Model: Normal Variational Graph Autoencoder')
        lr = hyperparameters['lr']
        weight_decay = hyperparameters['weight_decay']
        neg_sample_size = hyperparameters['neg_sample_size']
        KL_weight = hyperparameters['KL_weight']
    elif 'sv' in model_name and 'gae' in model_name:
        print('Model: Spherical Variational Graph Autoencoder')
        lr = hyperparameters['lr']
        weight_decay = hyperparameters['weight_decay']
        neg_sample_size = hyperparameters['neg_sample_size']
        KL_weight = hyperparameters['KL_weight']
    elif 'gae' in model_name:
        print('Model: Graph Autoencoder')
        lr = hyperparameters['lr']
        weight_decay = hyperparameters['weight_decay']
        neg_sample_size = hyperparameters['neg_sample_size']
    elif 'dgi' in model_name:
        print('Model: Deep Graph Infomax')
        lr = hyperparameters['lr']
        weight_decay = hyperparameters['weight_decay']
    else:
        NotImplemented


    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []

    for epoch in range(n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        optimizer.zero_grad()
        if 'nv' in model_name and 'gae' in model_name:
            loss = model(features,KL_weight=KL_weight,neg_sample_size=neg_sample_size)
        elif 'sv' in model_name and 'gae' in model_name:
            loss = model(features,KL_weight=KL_weight,neg_sample_size=neg_sample_size)
        elif 'gae' in model_name:
            loss = model(features, neg_sample_size=neg_sample_size)
        elif 'dgi' in model_name:
            loss = model(features)
        loss.backward()
        optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            if os.path.isdir(saved_dir):
                torch.save(model.state_dict(), os.path.join(saved_dir,model_name+'.pkl'))
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Early stopping! ".format(epoch, np.mean(dur), loss.item()))
            break
        elif epoch == (n_epochs-1):
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Last epoch! ".format(epoch, np.mean(dur), loss.item()))

        if epoch >= 3:
            dur.append(time.time() - t0)
        
        if verbose:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} ".format(epoch, np.mean(dur), loss.item()))
