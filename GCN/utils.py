import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = set(labels)
    #print(classes)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    #print(classes_dict)
    labels_onehot = np.array(list(map(classes_dict.get,labels)),dtype=np.int32)
    return labels_onehot

def load_data(path = "./data/cora",dataset="cora"):
    print("loading {} dataset...".format(dataset))
    idx_features_labels = np.genfromtxt("{}/{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 变成稀疏矩阵的形式
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #print(idx_features_labels[:5])
    #print(features[:5])
    labels = encode_onehot(idx_features_labels[:,-1])
    #print(labels[:5])
    #print(idx_features_labels[:5])
    #print(features[:5])

    idx = np.array(idx_features_labels[:,0],dtype=np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    #print(idx_map)

    edges_unordered = np.genfromtxt("{}/{}.cites".format(path, dataset),
                                    dtype=np.int32)
    #print(edges_unordered)
    edges = np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    #print(edges[:5])
    #将边变成稀疏矩阵的形式
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    #print(adj)
    # keep symmetric
    adj = adj + np.multiply(adj.T,adj.T > adj) - np.multiply(adj,adj.T > adj)

    features = normalize(features)
    # D-1AD-1
    adj = normalize(adj+sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200,500)
    idx_test = range(500,1500)

    #print(features.shape[0]) #2708
    #print(max(np.where(labels)[1]))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    #print(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj,features,labels,idx_train,idx_val,idx_test

def normalize(mx):
    # 行归一化
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum,-1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices,values,shape)

def accuray(output,labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)

load_data()