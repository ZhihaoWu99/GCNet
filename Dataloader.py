import torch
import random
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from torch_geometric.utils import to_undirected


def load_data(args):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['X']
    if ss.isspmatrix_csr(features):
        features = features.todense()
    if args.dataset in ['Citeseer', 'Cora', 'Pubmed', 'CoraFull']:
        features = normalize(features)
    features = torch.from_numpy(features).float().to(args.device)
    org_edge_index = to_undirected(torch.from_numpy(data['org']), num_nodes=features.shape[0]).to(args.device)
    knn_edge_index = to_undirected(torch.from_numpy(data['knn']), num_nodes=features.shape[0]).to(args.device)
    idx_train = data['train'].squeeze(0).tolist()
    idx_val = data['val'].squeeze(0).tolist()
    idx_test = data['test'].squeeze(0).tolist()
    idx_unlabeled = data['unl'].squeeze(0).tolist()
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    print("Train: {} | Val: {} | Test: {}".format(len(idx_train), len(idx_val), len(idx_test)))
    num_classes = len(np.unique(labels))
    labels = torch.from_numpy(labels).long().to(args.device)
    hidden_dims = [features.shape[1]] + [args.hdim] * (args.layer - 1) + [num_classes]
    print("Hidden dims: ", hidden_dims)

    return features, org_edge_index, knn_edge_index, hidden_dims, labels, idx_train, idx_test, idx_val, idx_unlabeled


def adj_to_edge_index(adj_matrix):
    edge_index = np.nonzero(adj_matrix)
    edge_index = torch.from_numpy(np.stack(edge_index)).long()
    return edge_index
