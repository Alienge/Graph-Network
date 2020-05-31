import torch
import numpy as np
import scipy.sparse as sp
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


a = np.array([1,2,3])
b = np.array([4,5,6])

c = a + b.reshape((3,1))
print(c)


#a = torch.randint(1,100,(2,2,5))
#print(a)
#print(a.repeat(1,1,2))
#b = a.repeat(1,1,2).view(2,4,5)
#print(b)
#c = a.repeat(1,2,1)
#print(torch.cat([b,c],dim=2).view(2,))



#print(range(10))

#a = torch.randint(1,100,(3,3)).numpy()
#print(a)
#a = a + np.multiply(a.T,a.T > a) - np.multiply(a,a.T > a)
#print(a)





'''
a = torch.randint(0,2,(3,3))
b = sp.coo_matrix(a)
c = np.array(b.sum(1))
print(a)
print(b)
print(c)
'''
'''
a = torch.randn((3,3))
rowsum = np.array(a.sum(1))
r_inv = np.power(rowsum,-1).flatten()

r_inv[np.isinf(r_inv)] = 0.

r_mat_inv = sp.diags(r_inv)
print(r_inv)
print(r_mat_inv)
print(a)
features = r_mat_inv.dot(a)
print(features)

print(sparse_to_tuple(features))

'''
