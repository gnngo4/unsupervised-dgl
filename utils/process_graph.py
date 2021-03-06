import numpy as np
import nibabel as nib
import networkx as nx

from copy import copy

"""
Tools to convert fMRI data into a graph
"""

def read_nifti(nii_path):
    obj = nib.load(nii_path)
    img = obj.get_fdata()

    return img, obj

def corr(X,Y):
    '''
    Compute row-wise pearson correlation coefficient between X,Y
    '''
    def corr_(X,Y):
        mX = X - X.mean(1)[:,None]
        mY = Y - Y.mean(1)[:,None]
        ssX = (mX**2).sum(1)[:,None]
        ssY = (mY**2).sum(1)[None]

        C = np.dot(mX,mY.T)/np.sqrt(np.dot(ssX,ssY))
        C[np.isnan(C)] = 0

        del mX, mY, ssX, ssY

        return C

    return corr_(X,Y)

def get_threshold_value(masked_timeseries,thr,n_iters=10,n_samples=5000,verbose=False):
    ''' 
    Approximate threshold value from subsampled connectivity matrix taken from
    masked_timeseries
    This threshold leads to `thr` % sparsity of the subsampled connectivity matrix
    '''
    import random
    
    def get_threshold_ind(s_,thr_):
        S_flatten = s_.flatten()
        n_connections = S_flatten.shape[0]
        top_percentile = int(thr_ * n_connections)
        top_percentile_ind = np.argsort(S_flatten)[-top_percentile]
        top_percentile_val = S_flatten[top_percentile_ind]
        
        return top_percentile_val

    if verbose:
        print('Getting threshold value for approximately ',thr*100,'% sparsity.')

    n_voxels = masked_timeseries.shape[0]
    n_tps = masked_timeseries.shape[1]

    thr_list = []
    for i in range(n_iters):
        sample_inds = np.array(random.sample(range(n_voxels),n_samples))
        C = corr(masked_timeseries[sample_inds,:],masked_timeseries[sample_inds,:])
        thr_ = get_threshold_ind(C,thr)
        thr_list.append(thr_)

        if verbose:
            print('ITER: ',i+1,'/',n_iters,' || thr: ',thr_)

    return np.array(thr_list).mean()

def graph_metric_dict_to_numpy(metric):
    n_nodes = len(metric.keys())
    metric_ = np.zeros((n_nodes,))
    for i in metric:
        metric_[i] = metric[i]

    return metric_

def timeseries_catch22_to_nifti(fmri_data,fmri_zero_data,verbose=False):
    from catch22 import catch22_all
    n_voxels = fmri_data.shape[0]
    catch22_features = np.zeros((n_voxels,22))
    for i in range(n_voxels):
        if verbose:
            if i % 1000 == 0:
                print('   - Processed ',i,'/',n_voxels,'voxels .')
        catchOut = catch22_all(fmri_data[i,:].tolist())
        catch22_features[i,:] = np.array(catchOut['values'])

    # ZERO ANY NANs
    catch22_features[np.isnan(catch22_features)] = 0

    ts_features = {}
    for ix,i in enumerate(catchOut['names']):
        temp_data = copy(fmri_zero_data)
        ts_features[i] = catch22_features[:,ix]

    return ts_features

def preprocess_features(features):
    '''
    Row-normalize feature matrix
    '''
    from scipy import sparse

    rowsum = features.sum(1)
    r_inv = np.power(rowsum,-1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    features = r_mat_inv.dot(features)

    return features

def get_sparse_connectivity_graph(nifti_path,mask_path, \
        sparsity_threshold=.01, sparsity_iters=10, sparsity_samples=10000, \
        processing_factor=10000, binarize_adjacency=False, self_loop=False, \
        sbref='none', simple_features=False, graph_features=[], catch22_features=False, \
        save_nifti=False, smooth_suffix='', \
        verbose=False):
    '''
    Calculate sparse adjacency matrix from fmri data and mask
    '''
    import math
    from scipy import sparse

    # Load fmri and mask data
    if verbose:
        print('Loading fmri data:', nifti_path)
    fmri_data,fmri_obj = read_nifti(nifti_path)
    if verbose:
        print('Loading mask data:', mask_path)
    mask_data,mask_obj = read_nifti(mask_path)
    fmri_mean_data = fmri_data.mean(3)

    # Extract timeseries from mask
    fmri_zero_data = fmri_mean_data * 0 # USE `fmri_zero_data` to fill with new metrics
    mask_coords = np.where(mask_data == 1) # USE `mask_coords` to remap metrics back onto nifti image
    if verbose:
        print('Number of voxels in mask:',mask_coords[0].shape[0])

    # get timeseries from mask
    masked_timeseries = fmri_data[mask_coords]
    n_voxels = masked_timeseries.shape[0]
    n_tps = masked_timeseries.shape[1]

    if n_voxels < sparsity_samples:
        sparsity_samples = n_voxels
        sparsity_iters = 1

    # Get threshold value
    thr = get_threshold_value(masked_timeseries,sparsity_threshold, \
                                n_iters=sparsity_iters, \
                                n_samples=sparsity_samples, \
                                verbose=verbose)

    # Obtain sparse connectivity matrix from `masked_timeseries`
    n_rows = math.floor(processing_factor*processing_factor / (n_voxels))
    n_splits = math.floor(n_voxels/n_rows)

    if verbose:
        print('Processing sparse connectivity matrix.')
        print('Processing ',n_rows,' per iteration for a total of ',n_splits,' iterations.')

    begin_count = True
    for i in range(n_splits+1):
        if begin_count:
            x1 = ':'
            x2 = (i+1)*n_rows

            C_subset = corr(masked_timeseries[:x2,:],masked_timeseries)
            nan_locs = np.isnan(C_subset)
            n_nans = nan_locs.sum()
            C_subset[nan_locs] = 0
            C_subset[np.where(C_subset<thr)] = 0
            if binarize_adjacency:
                C_subset[np.where(C_subset>=thr)] = 1
            C_subset_sp = sparse.csr_matrix(C_subset)

            begin_count = False

        elif i == n_splits:
            x1 = (i*n_rows)
            x2 = ':'

            C_subset = corr(masked_timeseries[x1:,:],masked_timeseries)
            nan_locs = np.isnan(C_subset)
            n_nans += nan_locs.sum()
            C_subset[nan_locs] = 0
            C_subset[np.where(C_subset<thr)] = 0
            if binarize_adjacency:
                C_subset[np.where(C_subset>=thr)] = 1
            C_subset_sp = sparse.vstack([C_subset_sp,sparse.csr_matrix(C_subset)])

        else:
            x1 = (i*n_rows)
            x2 = ((i+1)*n_rows)

            C_subset = corr(masked_timeseries[x1:x2,:],masked_timeseries)
            nan_locs = np.isnan(C_subset)
            n_nans += nan_locs.sum()
            C_subset[nan_locs] = 0
            C_subset[np.where(C_subset<thr)] = 0
            if binarize_adjacency:
                C_subset[np.where(C_subset>=thr)] = 1
            C_subset_sp = sparse.vstack([C_subset_sp,sparse.csr_matrix(C_subset)])

        if verbose:
            n_processed_rows = C_subset_sp.shape[0]
            total_rows = C_subset_sp.shape[1]
            print('   - Processed ',n_processed_rows,'/',total_rows,'voxels .')

    if self_loop:
        if verbose:
            print('Adding self-loops.')
        C_subset_sp.setdiag(1)
    else:
        if verbose:
            print('Removing self-loops.')
        C_subset_sp.setdiag(0)

    if verbose:
        print('Converting adjacency matrix to graph.')
    G = nx.from_scipy_sparse_matrix(C_subset_sp)


    # Process node graph-based features
    if len(graph_features) == 0:
        if verbose:
            print('No node graph-features features specified.')
    else:
        if verbose:
            print('Processing node graph-based features.')
            print(graph_features)

        G_features = {}
        for feat in graph_features:
            if feat == 'degree_centrality':
                if verbose:
                    print('   Calculating ',feat,'.')
                degree_centrality = nx.degree_centrality(G)
                G_features[feat] = graph_metric_dict_to_numpy(degree_centrality)
            elif feat == 'eigenvector_centrality':
                if verbose:
                    print('   Calculating ',feat,'.')
                eigenvector_centrality = nx.eigenvector_centrality(G)
                G_features[feat] = graph_metric_dict_to_numpy(eigenvector_centrality)
            elif feat == 'degree':
                if verbose:
                    print('   Calculating ',feat,'.')

                degree = np.array(C_subset_sp.sum(1))
                degree = degree.reshape((degree.shape[0],))
                G_features[feat] = degree
            else:
                if verbose:
                    print(feat,' Not implemented.')

    # Process node timeseries-property features
    if catch22_features:
        if verbose:
            print('Processing node timeseries-property features using catch22.')
   
        masked_fmri_data = fmri_data[mask_coords]
        ts_features = timeseries_catch22_to_nifti(masked_fmri_data,fmri_zero_data,verbose=verbose)

        del masked_fmri_data
        
    else:
        if verbose:
            print('No node timeseries-property features specified.')

    # Combine features into 4D nifti-like numpy array
    feature_list = []
    feature_counter = 0
    if sbref == 'none':
        pass
    else:
        feature_list.append('sbref')
        sbref_data,sbref_obj = read_nifti(sbref)
        temp_data = copy(fmri_zero_data)
        temp_data[mask_coords] = sbref_data[mask_coords]
        if feature_counter == 0:
            concat_features = temp_data[:,:,:,np.newaxis]
        else:
            concat_features = np.concatenate((concat_features,temp_data[:,:,:,np.newaxis]),axis=3)
        feature_counter += 1

    if simple_features:
        feature_list.append('Tmean')
        feature_list.append('Tstd')
        Tmean = masked_timeseries.mean(1)
        Tstd = masked_timeseries.std(1)
        for i in [Tmean,Tstd]:
            temp_data = copy(fmri_zero_data)
            temp_data[mask_coords] = i
            if feature_counter == 0:
                concat_features = temp_data[:,:,:,np.newaxis]
            else:
                concat_features = np.concatenate((concat_features,temp_data[:,:,:,np.newaxis]),axis=3)
            feature_counter += 1

    if len(graph_features) != 0:
        for i in G_features:
            feature_list.append(i)
            temp_data = copy(fmri_zero_data)
            temp_data[mask_coords] = G_features[i]
            if feature_counter == 0:
                concat_features = temp_data[:,:,:,np.newaxis]
            else:
                concat_features = np.concatenate((concat_features,temp_data[:,:,:,np.newaxis]),axis=3)
            feature_counter += 1
    
    if catch22_features:
        for i in ts_features:
            feature_list.append(i)
            temp_data = copy(fmri_zero_data)
            temp_data[mask_coords] = ts_features[i]
            if feature_counter == 0:
                concat_features = temp_data[:,:,:,np.newaxis]
            else:
                concat_features = np.concatenate((concat_features,temp_data[:,:,:,np.newaxis]),axis=3)
            feature_counter += 1

    # Save masked features to nifti
    if save_nifti:
        txt_ = mask_path.replace('.nii.gz','_features.txt')
        features_nii_ = mask_path.replace('.nii.gz','_features'+smooth_suffix+'.nii.gz')

        if verbose:
            print('Saving features to',features_nii_,'.\nOrdering of features can be found in',txt_,'.')

        f = open(txt_,"w")
        for ix,i in enumerate(feature_list):
            f.write(str(ix)+' - '+i+'\n')
        f.close()

        temp_obj = nib.Nifti1Image(concat_features,mask_obj.affine)
        nib.save(temp_obj,features_nii_)

    return G, concat_features[mask_coords], preprocess_features(concat_features[mask_coords]), feature_list


def save_numpy_to_nifti(feature_array,nifti_path,mask_path,suffix,smooth_suffix='',verbose=False):
    '''
    Convert masked numpy array to nifti
    '''
    # Load fmri and mask data
    if verbose:
        print('Loading fmri data:', nifti_path)
    fmri_data,fmri_obj = read_nifti(nifti_path)
    if verbose:
        print('Loading mask data:', mask_path)
    mask_data,mask_obj = read_nifti(mask_path)
    fmri_mean_data = fmri_data.mean(3)

    # Extract timeseries from mask
    fmri_zero_data = fmri_mean_data * 0 # USE `fmri_zero_data` to fill with new metrics
    mask_coords = np.where(mask_data == 1) # USE `mask_coords` to remap metrics back onto nifti image
    fmri_zero_data[mask_coords] = feature_array
    fmri_zero_data_obj = nib.Nifti1Image(fmri_zero_data,mask_obj.affine)
    nib.save(fmri_zero_data_obj,mask_path.replace('.nii.gz','_'+suffix+smooth_suffix+'.nii.gz'))
